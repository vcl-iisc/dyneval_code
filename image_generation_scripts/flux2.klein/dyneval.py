#!/usr/bin/env python3
"""Generate FLUX.1-dev images for DYNEVAL-1K remaining prompts.

Reads DYNEVAL-1K-REMAINING-PROMPTS.json and saves each image using the
filename from the JSON (e.g. 1001.png) under:
    DYNEVAL-1K-IMAGES-PART2/FLUX1.DEV/

Usage:
    python3 flux1.py --dry-run
    python3 flux1.py --limit 5
    python3 flux1.py --overwrite
"""
import argparse
import json
import os
import time
import traceback

DEFAULT_JSON = "DYNEVAL-1K-REMAINING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES-PART2", "FLUX2.klein")
DEFAULT_MODEL = "./FLUX2-KLEIN-MODEL"


def configure_torch():
    import torch

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


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


def load_pipeline(model_name_or_path, dtype, device, cpu_offload):
    import torch
    from diffusers import DiffusionPipeline

    configure_torch()
    pipe = DiffusionPipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map="cuda" if not cpu_offload else None,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    elif device != "cuda":
        pipe.to(device)
    return pipe


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
            "model": cfg["model_name_or_path"],
            "num_inference_steps": cfg["num_inference_steps"],
            "guidance_scale": cfg["guidance_scale"],
            "height": cfg["height"],
            "width": cfg["width"],
        }, ensure_ascii=False) + "\n")


def write_failure(fail_log, entry, exc):
    with open(fail_log, "a", encoding="utf-8") as lf:
        lf.write(f"{entry['gid']}\t{entry['filename']}\t{entry['prompt']}\t{exc}\n")
        lf.write(traceback.format_exc() + "\n")


def generate_all(pipe, pending, cfg):
    import torch

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
            with torch.inference_mode():
                image = pipe(
                    prompt=entry["prompt"],
                    height=cfg["height"],
                    width=cfg["width"],
                    guidance_scale=cfg["guidance_scale"],
                    num_inference_steps=cfg["num_inference_steps"],
                    generator=torch.Generator(device=cfg["device"]).manual_seed(seed),
                ).images[0]
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
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "seed_per_gid": args.seed_per_gid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_json", default=DEFAULT_JSON,
                    help="DYNEVAL-1K remaining prompts JSON")
    ap.add_argument("--output_dir", default=DEFAULT_OUT,
                    help="where to save PNGs")
    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                    default="bfloat16")
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=3.5)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed_per_gid", action="store_true",
                    help="use seed + gid for deterministic per-image variation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cpu_offload", action="store_true",
                    help="reduce VRAM use, usually slower")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.prompt_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(f"[FLUX.1-dev] json={len(prompts)}"
          f"  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}  -> {args.output_dir}")
    print(f"  model={args.model_name_or_path}")
    print(f"  steps={args.num_inference_steps}  guidance={args.guidance_scale}"
          f"  size={args.width}x{args.height}  seed={args.seed}")

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

    import torch

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"Loading {args.model_name_or_path} on {args.device} ...")
    pipe = load_pipeline(
        args.model_name_or_path,
        dtype=dtype,
        device=args.device,
        cpu_offload=args.cpu_offload,
    )

    ok, bad = generate_all(pipe, pending, build_cfg(args))
    print(f"\nDone FLUX.1-dev: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail.log in output_dir")


if __name__ == "__main__":
    main()
