#!/usr/bin/env python3
"""Generate Stable Diffusion 1.5 images for DYNEVAL-1K remaining prompts.

Reads DYNEVAL-1K-REMAINING-PROMPTS.json and saves each image using the
filename from the JSON (e.g. 1001.png) under:
    DYNEVAL-1K-IMAGES-PART2/sd1.5/

Uses all visible CUDA GPUs by default (one model replica per GPU). Override
with --gpus 0,1 or force a single GPU with --gpus 0.

Usage:
    python3 sd1_5.py --dry-run
    python3 sd1_5.py --limit 5
    python3 sd1_5.py --gpus 0,1,2,3
    python3 sd1_5.py --overwrite
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

DEFAULT_JSON = "DYNEVAL-1K-REMAINING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES-PART2", "playground")
DEFAULT_MODEL = "playgroundai/playground-v2.5-1024px-aesthetic"


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


def shard(items, n_shards, shard_id):
    return items[shard_id::n_shards]


def parse_gpus(gpu_str):
    if gpu_str is not None:
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]
    import torch

    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def load_pipeline(model_name_or_path, dtype, cpu_offload):
    import torch
    from diffusers import DiffusionPipeline

    configure_torch()
    pipe = DiffusionPipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def write_manifest(manifest, entry, seconds, seed, cfg):
    row = {
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
    }
    if cfg.get("gpu") is not None:
        row["gpu"] = cfg["gpu"]
    with open(manifest, "a", encoding="utf-8") as mf:
        mf.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_failure(fail_log, entry, exc):
    with open(fail_log, "a", encoding="utf-8") as lf:
        lf.write(f"{entry['gid']}\t{entry['filename']}\t{entry['prompt']}\t{exc}\n")
        lf.write(traceback.format_exc() + "\n")


def generate_entries(pipe, pending, cfg, manifest, fail_log, tag=None):
    import torch

    ok = bad = 0
    total = len(pending)
    device = torch.device("cuda")

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
                    generator=torch.Generator(device=device).manual_seed(seed),
                ).images[0]
            image.save(dst)
            elapsed = time.time() - t0
            write_manifest(manifest, entry, elapsed, seed, cfg)
            ok += 1
            prefix = f"[{tag} " if tag else "["
            print(f"{prefix}{i}/{total}] {entry['filename']} OK"
                  f"  [{entry.get('benchmark')}]  {elapsed:.1f}s")
        except Exception as exc:
            bad += 1
            write_failure(fail_log, entry, exc)
            prefix = f"[{tag} " if tag else "["
            print(f"{prefix}{i}/{total}] {entry['filename']} FAILED: {exc}")

    return ok, bad


def build_cfg(args, gpu=None):
    return {
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "seed_per_gid": args.seed_per_gid,
        "cpu_offload": args.cpu_offload,
        "dtype": args.dtype,
        "gpu": gpu,
    }


def worker_main(worker_id, gpu_id, pending, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    tag = f"gpu{gpu_id}"
    manifest = os.path.join(cfg["output_dir"], f"generated_manifest.{tag}.jsonl")
    fail_log = os.path.join(cfg["output_dir"], f"generate_fail.{tag}.log")

    print(f"[{tag}] worker {worker_id}: {len(pending)} prompts")
    if not pending:
        return 0, 0

    import torch

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg["dtype"]]

    t0 = time.time()
    print(f"[{tag}] loading {cfg['model_name_or_path']} ...")
    pipe = load_pipeline(
        cfg["model_name_or_path"],
        dtype=dtype,
        cpu_offload=cfg["cpu_offload"],
    )
    print(f"[{tag}] model ready in {time.time() - t0:.1f}s")

    cfg = dict(cfg, gpu=gpu_id)
    ok, bad = generate_entries(pipe, pending, cfg, manifest, fail_log, tag=tag)
    print(f"[{tag}] done: generated={ok} failed={bad}")
    return ok, bad


def run_single_gpu(pending, cfg, gpu=None):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cfg = dict(cfg, gpu=gpu)
        manifest = os.path.join(cfg["output_dir"], f"generated_manifest.gpu{gpu}.jsonl")
        fail_log = os.path.join(cfg["output_dir"], f"generate_fail.gpu{gpu}.log")
        tag = f"gpu{gpu}"
    else:
        manifest = os.path.join(cfg["output_dir"], "generated_manifest.jsonl")
        fail_log = os.path.join(cfg["output_dir"], "generate_fail.log")
        tag = None

    import torch

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[cfg["dtype"]]

    print(f"Loading {cfg['model_name_or_path']} ...")
    pipe = load_pipeline(
        cfg["model_name_or_path"],
        dtype=dtype,
        cpu_offload=cfg["cpu_offload"],
    )
    return generate_entries(pipe, pending, cfg, manifest, fail_log, tag=tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_json", default=DEFAULT_JSON,
                    help="DYNEVAL-1K remaining prompts JSON")
    ap.add_argument("--output_dir", default=DEFAULT_OUT,
                    help="where to save PNGs")
    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
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
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids (default: all visible CUDA devices)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    prompts = load_prompts(args.prompt_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)
    gpus = parse_gpus(args.gpus)

    print(f"[SD-1.5] json={len(prompts)}"
          f"  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}  -> {args.output_dir}")
    print(f"  model={args.model_name_or_path}")
    print(f"  steps={args.num_inference_steps}  guidance={args.guidance_scale}"
          f"  size={args.width}x{args.height}  seed={args.seed}")
    if gpus:
        print(f"  gpus={gpus}")
    else:
        print("  gpus=none (CUDA unavailable; will fail at load time)")

    if args.limit:
        pending = pending[:args.limit]
        print(f"  limited to {len(pending)} this run")

    if not pending:
        print("Nothing to do.")
        return

    if args.dry_run:
        if gpus:
            for rank, gpu in enumerate(gpus):
                part = shard(pending, len(gpus), rank)
                print(f"  GPU {gpu}: {len(part)} prompts")
        for entry in pending[:10]:
            print(f"  gid={entry['gid']}  {entry['filename']}"
                  f"  [{entry.get('benchmark')}]  {entry['prompt'][:90]}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return

    if args.cpu_offload and len(gpus) > 1:
        print("WARNING: --cpu_offload with multiple GPUs is unsupported; using GPU 0 only.")
        gpus = [gpus[0]]

    cfg = build_cfg(args)

    if len(gpus) > 1:
        ctx = mp.get_context("spawn")
        jobs = []
        for rank, gpu in enumerate(gpus):
            part = shard(pending, len(gpus), rank)
            if part:
                jobs.append((rank, gpu, part, cfg))
        print(f"Launching {len(jobs)} GPU workers on {gpus} ...")
        with ctx.Pool(len(jobs)) as pool:
            results = pool.starmap(worker_main, jobs)
        ok = sum(r[0] for r in results)
        bad = sum(r[1] for r in results)
    else:
        gpu = gpus[0] if gpus else None
        ok, bad = run_single_gpu(pending, cfg, gpu=gpu)

    print(f"\nDone SD-1.5: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail*.log in output_dir")


if __name__ == "__main__":
    main()
