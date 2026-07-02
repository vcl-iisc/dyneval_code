#!/usr/bin/env python3
"""Generate missing Emu3 images for the DYNEVAL-1K set.

Reads Emu3-MISSING-PROMPTS.json (904 prompts as of 2026-06-22).
Already have: 96 (37 GenEval + 59 DPG-Bench). Missing: 904.

Saves:
    <output_dir>/<gid:04d>.png   e.g. 0026.png, 0031.png

Copy to main machine:
    DYNEVAL-1K-IMAGES/Emu3/

Usage (from DYNEVAL project root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/emu3/image-gen.py --dry-run
    python dyneval_code/generator_scripts/emu3/image-gen.py --limit 5
    python dyneval_code/generator_scripts/emu3/image-gen.py --gpus 0,1,2,3

Remote server:
    python dyneval_code/generator_scripts/emu3/image-gen.py \\
        --batch_json Emu3-MISSING-PROMPTS.json \\
        --output_dir ./emu3_outputs \\
        --emu_hub BAAI/Emu3-Gen \\
        --vq_hub BAAI/Emu3-VisionTokenizer \\
        --gpus 0,1,2,3
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

DEFAULT_JSON = "Emu3-MISSING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES", "Emu3")
DEFAULT_EMU_HUB = "BAAI/Emu3-Gen"
DEFAULT_VQ_HUB = "BAAI/Emu3-VisionTokenizer"
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg "
    "artifacts, signature, watermark, username, blurry."
)


def resolve_attn_implementation(preferred):
    if preferred != "auto":
        return preferred
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_stack(emu_hub, vq_hub, attn_implementation):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import (
        LogitsProcessorList,
        PrefixConstrainedLogitsProcessor,
        UnbatchedClassifierFreeGuidanceLogitsProcessor,
    )
    from transformers.generation.configuration_utils import GenerationConfig

    from emu3.mllm.processing_emu3 import Emu3Processor

    device = "cuda:0"
    attn = resolve_attn_implementation(attn_implementation)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            emu_hub,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
            trust_remote_code=True,
        )
    except Exception:
        if attn == "flash_attention_2":
            attn = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                emu_hub,
                device_map=device,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn,
                trust_remote_code=True,
            )
        else:
            raise

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        emu_hub, trust_remote_code=True, padding_side="left"
    )
    image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(
        vq_hub, device_map=device, trust_remote_code=True
    ).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    generation_config = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=40960,
        do_sample=True,
        top_k=2048,
    )

    return {
        "model": model,
        "processor": processor,
        "generation_config": generation_config,
        "device": device,
        "attn_implementation": attn,
        "Image": Image,
        "torch": torch,
        "LogitsProcessorList": LogitsProcessorList,
        "PrefixConstrainedLogitsProcessor": PrefixConstrainedLogitsProcessor,
        "UnbatchedClassifierFreeGuidanceLogitsProcessor": UnbatchedClassifierFreeGuidanceLogitsProcessor,
    }


def generate_one(stack, prompt, guidance_scale, ratio, target_size):
    model = stack["model"]
    processor = stack["processor"]
    generation_config = stack["generation_config"]
    device = stack["device"]
    torch = stack["torch"]
    Image = stack["Image"]
    LogitsProcessorList = stack["LogitsProcessorList"]
    PrefixConstrainedLogitsProcessor = stack["PrefixConstrainedLogitsProcessor"]
    UnbatchedClassifierFreeGuidanceLogitsProcessor = stack["UnbatchedClassifierFreeGuidanceLogitsProcessor"]

    full_prompt = prompt + POSITIVE_PROMPT
    proc_kwargs = dict(
        mode="G",
        ratio=ratio,
        image_area=model.config.image_area,
        return_tensors="pt",
        padding="longest",
    )

    pos_inputs = processor(text=full_prompt, **proc_kwargs)
    neg_inputs = processor(text=NEGATIVE_PROMPT, **proc_kwargs)

    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)
    logits_processor = LogitsProcessorList([
        UnbatchedClassifierFreeGuidanceLogitsProcessor(
            guidance_scale,
            model,
            unconditional_ids=neg_inputs.input_ids.to(device),
        ),
        PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
    ])

    with torch.inference_mode():
        outputs = model.generate(
            pos_inputs.input_ids.to(device),
            generation_config,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(device),
        )

    for item in processor.decode(outputs[0]):
        if isinstance(item, Image.Image):
            if target_size and item.size != target_size:
                item = item.resize(target_size, Image.LANCZOS)
            return item

    raise RuntimeError("Emu3 decode returned no image")


def load_prompts(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", data)
    return sorted(prompts, key=lambda p: int(p["gid"]))


def filter_pending(prompts, output_dir, overwrite):
    if overwrite:
        return prompts
    return [
        p for p in prompts
        if not os.path.exists(os.path.join(output_dir, p["filename"]))
    ]


def shard(items, n_shards, shard_id):
    return items[shard_id::n_shards]


def worker_main(worker_id, gpu_id, pending, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    tag = f"gpu{gpu_id}"
    manifest = os.path.join(cfg["output_dir"], f"generated_manifest.{tag}.jsonl")
    fail_log = os.path.join(cfg["output_dir"], f"generate_fail.{tag}.log")

    print(f"[{tag}] worker {worker_id}: {len(pending)} prompts")
    if not pending:
        return 0, 0

    t0 = time.time()
    print(f"[{tag}] loading Emu3 ({cfg['emu_hub']}) attn={cfg['attn_implementation']} ...")
    stack = load_stack(cfg["emu_hub"], cfg["vq_hub"], cfg["attn_implementation"])
    print(
        f"[{tag}] model ready in {time.time() - t0:.1f}s "
        f"(attn={stack['attn_implementation']})"
    )

    target_size = (cfg["width"], cfg["height"]) if cfg["resize"] else None
    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        t1 = time.time()
        try:
            generate_one(
                stack,
                prompt=prompt,
                guidance_scale=cfg["guidance_scale"],
                ratio=cfg["ratio"],
                target_size=target_size,
            ).save(dst)
            ok += 1
            elapsed = time.time() - t1
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({
                    "gid": entry["gid"],
                    "filename": filename,
                    "benchmark": entry["benchmark"],
                    "prompt": prompt,
                    "seconds": round(elapsed, 2),
                    "gpu": gpu_id,
                }, ensure_ascii=False) + "\n")
            print(
                f"[{tag} {i}/{len(pending)}] {filename} OK"
                f"  gid={entry['gid']}  [{entry['benchmark']}]  {elapsed:.1f}s"
            )
        except Exception as exc:
            bad += 1
            with open(fail_log, "a", encoding="utf-8") as lf:
                lf.write(f"{entry['gid']}\t{filename}\t{prompt}\t{exc}\n")
                lf.write(traceback.format_exc() + "\n")
            print(f"[{tag} {i}/{len(pending)}] {filename} FAILED: {exc}")

    print(f"[{tag}] done: generated={ok} failed={bad}")
    return ok, bad


def run_single_gpu(pending, cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    manifest = os.path.join(cfg["output_dir"], "generated_manifest.jsonl")
    fail_log = os.path.join(cfg["output_dir"], "generate_fail.log")

    if cfg.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    print(f"Loading Emu3 ({cfg['emu_hub']}) attn={cfg['attn_implementation']} ...")
    stack = load_stack(cfg["emu_hub"], cfg["vq_hub"], cfg["attn_implementation"])
    target_size = (cfg["width"], cfg["height"]) if cfg["resize"] else None

    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        t1 = time.time()
        try:
            generate_one(
                stack,
                prompt=prompt,
                guidance_scale=cfg["guidance_scale"],
                ratio=cfg["ratio"],
                target_size=target_size,
            ).save(dst)
            ok += 1
            elapsed = time.time() - t1
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({
                    "gid": entry["gid"],
                    "filename": filename,
                    "benchmark": entry["benchmark"],
                    "prompt": prompt,
                    "seconds": round(elapsed, 2),
                }, ensure_ascii=False) + "\n")
            print(
                f"[{i}/{len(pending)}] {filename} OK"
                f"  gid={entry['gid']}  [{entry['benchmark']}]  {elapsed:.1f}s"
            )
        except Exception as exc:
            bad += 1
            with open(fail_log, "a", encoding="utf-8") as lf:
                lf.write(f"{entry['gid']}\t{filename}\t{prompt}\t{exc}\n")
                lf.write(traceback.format_exc() + "\n")
            print(f"[{i}/{len(pending)}] {filename} FAILED: {exc}")

    return ok, bad


def parse_gpus(gpu_str):
    if not gpu_str:
        return None
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]


def build_cfg(args, gpu=None):
    return {
        "emu_hub": args.emu_hub,
        "vq_hub": args.vq_hub,
        "output_dir": args.output_dir,
        "guidance_scale": args.guidance_scale,
        "ratio": args.ratio,
        "width": args.width,
        "height": args.height,
        "resize": args.resize,
        "attn_implementation": args.attn_implementation,
        "gpu": gpu,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_json", default=DEFAULT_JSON)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--emu_hub", default=DEFAULT_EMU_HUB)
    ap.add_argument("--vq_hub", default=DEFAULT_VQ_HUB)
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--ratio", default="1:1")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument(
        "--resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resize decoded image to width x height (default: 1024x1024)",
    )
    ap.add_argument(
        "--attn_implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids, e.g. 0,1,2,3")
    args = ap.parse_args()

    prompts = load_prompts(args.batch_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(f"[Emu3] json={len(prompts)}  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}")
    print(f"  emu_hub={args.emu_hub}")
    print(f"  vq_hub={args.vq_hub}")
    print(f"  ratio={args.ratio}  guidance={args.guidance_scale}  attn={args.attn_implementation}")
    print(f"  output -> {args.output_dir}")

    if args.limit:
        pending = pending[: args.limit]
        print(f"  limited to {len(pending)} this run")

    if not pending:
        print("Nothing to do.")
        return

    if args.dry_run:
        gpus = parse_gpus(args.gpus) or [0]
        for rank, gpu in enumerate(gpus):
            part = shard(pending, len(gpus), rank)
            print(f"  GPU {gpu}: {len(part)} prompts")
        for p in pending[:10]:
            print(f"  gid={p['gid']}  {p['filename']}  [{p['benchmark']}]  {p['prompt'][:80]}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return

    gpus = parse_gpus(args.gpus)
    if gpus and len(gpus) > 1:
        cfg = build_cfg(args)
        ctx = mp.get_context("spawn")
        jobs = [
            (rank, gpu, shard(pending, len(gpus), rank), cfg)
            for rank, gpu in enumerate(gpus)
            if shard(pending, len(gpus), rank)
        ]
        print(f"Launching {len(jobs)} GPU workers on {gpus} ...")
        with ctx.Pool(len(jobs)) as pool:
            results = pool.starmap(worker_main, jobs)
        ok = sum(r[0] for r in results)
        bad = sum(r[1] for r in results)
    else:
        gpu = gpus[0] if gpus else None
        ok, bad = run_single_gpu(pending, build_cfg(args, gpu=gpu))

    print(f"\nDone Emu3: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail*.log in output_dir")


if __name__ == "__main__":
    main()
