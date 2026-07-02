#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Text-to-image generation for BAGEL (converted from inference.ipynb).

Batch mode reads DYNEVAL-style JSON and saves PNGs by filename.

Usage (from DYNEVAL project root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/Bagel/scripts/infer-image-gen.py \\
        --model-path ./weights/BAGEL-7B-MoT

    python dyneval_code/generator_scripts/Bagel/scripts/infer-image-gen.py \\
        --model-path ./weights/BAGEL-7B-MoT --dry-run --limit 5

    # single prompt
    python dyneval_code/generator_scripts/Bagel/scripts/infer-image-gen.py \\
        --model-path ./weights/BAGEL-7B-MoT \\
        --prompt "a car made of small cars" --output output.png
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Default to PyTorch SDPA (no flash-attn) for inference scripts.
os.environ.setdefault("BAGEL_USE_FLASH_ATTN", "0")

import numpy as np
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image

DEFAULT_JSON = "DYNEVAL-1K-REMAINING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES-PART2", "bagel")
DEFAULT_OFFLOAD = os.path.join(".tmp", "bagel_offload")

SAME_DEVICE_MODULES = [
    "language_model.model.embed_tokens",
    "time_embedder",
    "latent_pos_embed",
    "vae2llm",
    "llm2vae",
    "connector",
    "vit_pos_embed",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def build_device_map(model, max_mem_per_gpu):
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(SAME_DEVICE_MODULES[0], "cuda:0")
        for key in SAME_DEVICE_MODULES:
            device_map[key] = first_device if key in device_map else "cuda:0"
    else:
        first_device = device_map.get(SAME_DEVICE_MODULES[0])
        for key in SAME_DEVICE_MODULES:
            if key in device_map:
                device_map[key] = first_device
    return device_map


def load_inferencer(args):
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer

    attn_backend = "flash-attn" if os.environ.get("BAGEL_USE_FLASH_ATTN") == "1" else "PyTorch SDPA"
    print(f"Attention backend: {attn_backend}", flush=True)

    model_path = args.model_path
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    if getattr(llm_config, "pad_token_id", None) is None:
        llm_config.pad_token_id = getattr(llm_config, "eos_token_id", 151643)

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(args.image_size, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = build_device_map(model, args.max_mem_per_gpu)
    print(f"device_map: {device_map}", flush=True)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=args.offload_folder,
    ).eval()

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def inference_hyper(args):
    return {
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "cfg_interval": args.cfg_interval,
        "timestep_shift": args.timestep_shift,
        "num_timesteps": args.num_timesteps,
        "cfg_renorm_min": args.cfg_renorm_min,
        "cfg_renorm_type": args.cfg_renorm_type,
        "image_shapes": (args.image_size, args.image_size),
        "enable_taylorseer": args.enable_taylorseer,
        "max_think_token_n": args.max_think_token_n,
        "do_sample": args.do_sample,
    }


def generate_one(inferencer, prompt, args):
    hyper = inference_hyper(args)
    if args.think:
        output = inferencer(text=prompt, think=True, **hyper)
    else:
        output = inferencer(text=prompt, **hyper)
    return output


def write_manifest(manifest, entry, seconds, seed, args):
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
            "model_path": args.model_path,
            "think": args.think,
            "cfg_text_scale": args.cfg_text_scale,
            "num_timesteps": args.num_timesteps,
            "image_size": args.image_size,
        }, ensure_ascii=False) + "\n")


def write_failure(fail_log, entry, exc):
    with open(fail_log, "a", encoding="utf-8") as lf:
        lf.write(f"{entry['gid']}\t{entry['filename']}\t{entry['prompt']}\t{exc}\n")
        lf.write(traceback.format_exc() + "\n")


def generate_all(inferencer, pending, args):
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = os.path.join(args.output_dir, "generated_manifest.jsonl")
    fail_log = os.path.join(args.output_dir, "generate_fail.log")
    ok = bad = 0
    total = len(pending)

    for i, entry in enumerate(pending, 1):
        dst = os.path.join(args.output_dir, entry["filename"])
        prompt = entry["prompt"].strip()
        seed = args.seed + int(entry["gid"]) if args.seed_per_gid else args.seed
        set_seed(seed)
        t0 = time.time()
        try:
            output = generate_one(inferencer, prompt, args)
            image = output["image"]
            if image is None:
                raise RuntimeError("inferencer returned no image")
            image.save(dst)
            elapsed = time.time() - t0
            write_manifest(manifest, entry, elapsed, seed, args)
            ok += 1
            think_note = ""
            if args.think and output.get("text"):
                think_note = f"  think_chars={len(output['text'])}"
            print(f"[{i}/{total}] {entry['filename']} OK"
                  f"  [{entry.get('benchmark')}]  {elapsed:.1f}s{think_note}", flush=True)
        except Exception as exc:
            bad += 1
            write_failure(fail_log, entry, exc)
            print(f"[{i}/{total}] {entry['filename']} FAILED: {exc}", flush=True)

    return ok, bad


def parse_cfg_interval(value):
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("cfg_interval must be two floats, e.g. 0.4,1.0")
    return parts


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--use-flash-attn", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.use_flash_attn:
        os.environ["BAGEL_USE_FLASH_ATTN"] = "1"

    parser = argparse.ArgumentParser(description="BAGEL text-to-image (from inference.ipynb)")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to BAGEL-7B-MoT weights directory")
    parser.add_argument("--prompt-json", default=DEFAULT_JSON,
                        help="DYNEVAL prompts JSON for batch mode")
    parser.add_argument("--output-dir", default=DEFAULT_OUT,
                        help="Directory to save generated PNGs")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (skips JSON batch mode)")
    parser.add_argument("--output", type=str, default="output.png",
                        help="Output path for single --prompt mode")
    parser.add_argument("--think", action="store_true",
                        help="Enable think-before-generate mode")
    parser.add_argument("--max-think-token-n", type=int, default=1000)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--max-mem-per-gpu", type=str, default="40GiB")
    parser.add_argument("--offload-folder", type=str, default=DEFAULT_OFFLOAD)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.0)
    parser.add_argument("--cfg-interval", type=parse_cfg_interval, default=[0.4, 1.0])
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", type=str, default="global",
                        choices=["global", "channel", "text_channel"])
    parser.add_argument("--enable-taylorseer", action="store_true")
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Use flash-attn if installed (default: PyTorch SDPA only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-per-gid", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.prompt is not None:
        set_seed(args.seed)
        print(f"Single-prompt mode: {args.prompt[:90]}", flush=True)
        inferencer = load_inferencer(args)
        output = generate_one(inferencer, args.prompt, args)
        output["image"].save(args.output)
        if args.think and output.get("text"):
            print(output["text"], flush=True)
        print(f"Saved {args.output}", flush=True)
        return

    prompts = load_prompts(args.prompt_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)
    print(f"[BAGEL] json={len(prompts)}"
          f"  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}  -> {args.output_dir}", flush=True)
    print(f"  model={args.model_path}", flush=True)
    print(f"  think={args.think}  cfg_text_scale={args.cfg_text_scale}"
          f"  num_timesteps={args.num_timesteps}  size={args.image_size}"
          f"  seed={args.seed}", flush=True)

    if args.limit:
        pending = pending[:args.limit]
        print(f"  limited to {len(pending)} this run", flush=True)

    if not pending:
        print("Nothing to do.", flush=True)
        return

    if args.dry_run:
        for entry in pending[:10]:
            print(f"  gid={entry['gid']}  {entry['filename']}"
                  f"  [{entry.get('benchmark')}]  {entry['prompt'][:90]}", flush=True)
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more", flush=True)
        return

    print(f"Loading model from {args.model_path} ...", flush=True)
    t0 = time.time()
    inferencer = load_inferencer(args)
    print(f"Model ready in {time.time() - t0:.1f}s", flush=True)

    ok, bad = generate_all(inferencer, pending, args)
    print(f"\nDone BAGEL: generated={ok}  failed={bad}  -> {args.output_dir}", flush=True)
    if bad:
        print("Check generate_fail.log in output_dir", flush=True)


if __name__ == "__main__":
    main()
