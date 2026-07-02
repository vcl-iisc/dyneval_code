#!/usr/bin/env python3
"""Generate Janus-Pro-7B images for DYNEVAL-1K remaining prompts.

Run from the DYNEVAL project root (paths below are relative to cwd):

    python dyneval_code/generator_scripts/janus/infer.py --dry-run
    python dyneval_code/generator_scripts/janus/infer.py --limit 5

Reads:
    DYNEVAL-1K-REMAINING-PROMPTS.json

Saves:
    DYNEVAL-1K-IMAGES-PART2/janus-pro-7b/<filename from JSON>
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
import PIL.Image
import torch
from transformers import AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from janus.models import MultiModalityCausalLM, VLChatProcessor

DEFAULT_JSON = "DYNEVAL-1K-REMAINING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES-PART2", "janus-pro-7b")
DEFAULT_MODEL = "./JANUS-PRO-MODEL"


_TRIANGULAR_BFLOAT16_PATCHED = False


def patch_bfloat16_triangular_cuda():
    """Work around older PyTorch CUDA builds missing bf16 triu/tril kernels."""
    global _TRIANGULAR_BFLOAT16_PATCHED
    if _TRIANGULAR_BFLOAT16_PATCHED:
        return

    original_triu = torch.triu
    original_tril = torch.tril

    def _supports_bfloat16_cuda_triangular(op):
        if not torch.cuda.is_available():
            return True
        try:
            sample = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
            op(sample)
            return True
        except RuntimeError as exc:
            if "triu_tril_cuda_template" in str(exc) and "BFloat16" in str(exc):
                return False
            raise

    triu_supported = _supports_bfloat16_cuda_triangular(original_triu)
    tril_supported = _supports_bfloat16_cuda_triangular(original_tril)
    if triu_supported and tril_supported:
        _TRIANGULAR_BFLOAT16_PATCHED = True
        return

    def triu(input, diagonal=0, *, out=None):
        if input.is_cuda and input.dtype == torch.bfloat16:
            result = original_triu(input.float(), diagonal=diagonal).to(torch.bfloat16)
            if out is not None:
                return out.copy_(result)
            return result
        return original_triu(input, diagonal=diagonal, out=out)

    def tril(input, diagonal=0, *, out=None):
        if input.is_cuda and input.dtype == torch.bfloat16:
            result = original_tril(input.float(), diagonal=diagonal).to(torch.bfloat16)
            if out is not None:
                return out.copy_(result)
            return result
        return original_tril(input, diagonal=diagonal, out=out)

    torch.triu = triu
    torch.tril = tril
    _TRIANGULAR_BFLOAT16_PATCHED = True


def configure_torch():
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        patch_bfloat16_triangular_cuda()


def load_prompts(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", data)
    return sorted(prompts, key=lambda p: int(p["gid"]))


def filter_pending(prompts, output_dir, overwrite):
    if overwrite:
        return prompts
    return [
        entry for entry in prompts
        if not os.path.exists(os.path.join(output_dir, entry["filename"]))
    ]


def build_prompt(vl_chat_processor, text):
    conversation = [
        {"role": "<|User|>", "content": text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag


def load_model(model_path):
    configure_torch()
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_gpt, vl_chat_processor


@torch.inference_mode()
def generate_image(
    mmgpt,
    vl_chat_processor,
    prompt_text,
    temperature=1.0,
    cfg_weight=5.0,
    image_token_num_per_image=576,
    img_size=384,
    patch_size=16,
    seed=0,
):
    if seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    prompt = build_prompt(vl_chat_processor, prompt_text)
    parallel_size = 1

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int,
    ).cuda()
    outputs = None

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1,
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    return PIL.Image.fromarray(dec[0])


def write_manifest(manifest, entry, seconds, seed, cfg):
    with open(manifest, "a", encoding="utf-8") as mf:
        mf.write(json.dumps({
            "gid": entry["gid"],
            "filename": entry["filename"],
            "benchmark": entry.get("benchmark"),
            "category": entry.get("category"),
            "subcategory": entry.get("subcategory"),
            "prompt": entry["prompt"],
            "seconds": round(seconds, 2),
            "seed": seed,
            "model": cfg["model_path"],
            "cfg_weight": cfg["cfg_weight"],
            "temperature": cfg["temperature"],
            "img_size": cfg["img_size"],
        }, ensure_ascii=False) + "\n")


def write_failure(fail_log, entry, exc):
    with open(fail_log, "a", encoding="utf-8") as lf:
        lf.write(f"{entry['gid']}\t{entry['filename']}\t{entry['prompt']}\t{exc}\n")
        lf.write(traceback.format_exc() + "\n")


def generate_all(vl_gpt, vl_chat_processor, pending, cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    manifest = os.path.join(cfg["output_dir"], "generated_manifest.jsonl")
    fail_log = os.path.join(cfg["output_dir"], "generate_fail.log")

    ok = bad = 0
    total = len(pending)

    for i, entry in enumerate(pending, 1):
        dst = os.path.join(cfg["output_dir"], entry["filename"])
        seed = cfg["seed"] + int(entry["gid"]) if cfg["seed_per_gid"] else cfg["seed"]
        t0 = time.time()
        try:
            image = generate_image(
                vl_gpt,
                vl_chat_processor,
                entry["prompt"],
                temperature=cfg["temperature"],
                cfg_weight=cfg["cfg_weight"],
                img_size=cfg["img_size"],
                seed=seed,
            )
            image.save(dst)
            elapsed = time.time() - t0
            write_manifest(manifest, entry, elapsed, seed, cfg)
            ok += 1
            print(f"[{i}/{total}] {entry['filename']} OK"
                  f"  [{entry.get('benchmark')}]  {elapsed:.1f}s")
        except Exception as exc:
            bad += 1
            write_failure(fail_log, entry, exc)
            print(f"[{i}/{total}] {entry['filename']} FAILED: {exc}")

    return ok, bad


def build_cfg(args):
    return {
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "seed_per_gid": args.seed_per_gid,
        "temperature": args.temperature,
        "cfg_weight": args.cfg_weight,
        "img_size": args.img_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_json", default=DEFAULT_JSON,
                    help="DYNEVAL-1K remaining prompts JSON")
    ap.add_argument("--output_dir", default=DEFAULT_OUT,
                    help="where to save PNGs")
    ap.add_argument("--model_path", default=DEFAULT_MODEL,
                    help="Janus-Pro-7B model path or HF id")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed seed; use -1 for random per image")
    ap.add_argument("--seed_per_gid", action="store_true",
                    help="use seed + gid for deterministic per-image variation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.prompt_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(f"[Janus-Pro-7B] json={len(prompts)}"
          f"  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}  -> {args.output_dir}")
    print(f"  model={args.model_path}")
    print(f"  cfg_weight={args.cfg_weight}  temperature={args.temperature}"
          f"  img_size={args.img_size}  seed={args.seed}")

    if args.limit:
        pending = pending[:args.limit]
        print(f"  limited to {len(pending)} this run")

    if not pending:
        print("Nothing to do.")
        return

    if args.dry_run:
        for entry in pending[:10]:
            print(f"  gid={entry['gid']}  {entry['filename']}"
                  f"  [{entry.get('benchmark')}]  {entry['prompt'][:90]}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return

    print(f"Loading {args.model_path} ...")
    vl_gpt, vl_chat_processor = load_model(args.model_path)

    ok, bad = generate_all(vl_gpt, vl_chat_processor, pending, build_cfg(args))
    print(f"\nDone Janus-Pro-7B: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail.log in output_dir")


if __name__ == "__main__":
    main()
