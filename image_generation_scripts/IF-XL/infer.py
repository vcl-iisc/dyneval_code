#!/usr/bin/env python3
"""Generate missing DeepFloyd IF-XL images for the DYNEVAL-1K set.

Reads DeepFloyd-IF-XL-MISSING-PROMPTS.json (405 EvalMuse prompts; 595/1000 exist).
Already have: 129 GenEval + 466 DPG-Bench. Missing: 405 EvalMuse.

Saves:
    <output_dir>/<gid:04d>.png   e.g. 0026.png

Copy to main machine:
    DYNEVAL-1K-IMAGES/DeepFloyd IF-XL/

Usage (from DYNEVAL project root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/IF-XL/infer.py --dry-run
    python dyneval_code/generator_scripts/IF-XL/infer.py --limit 5
    python dyneval_code/generator_scripts/IF-XL/infer.py --gpus 0,1,2,3

Remote server:
    python dyneval_code/generator_scripts/IF-XL/infer.py \\
        --batch_json DeepFloyd-IF-XL-MISSING-PROMPTS.json \\
        --output_dir ./ifxl_outputs \\
        --gpus 0,1,2,3

If OOM:
    python3 generate_deepfloyd_ifxl_missing.py --cpu_offload
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

DEFAULT_JSON = "DeepFloyd-IF-XL-MISSING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES", "DeepFloyd IF-XL")
MODEL_ID = "DeepFloyd/IF-I-XL-v1.0"


def load_pipeline(model_id, cpu_offload=False, torch_dtype="bfloat16"):
    import torch
    from diffusers import DiffusionPipeline

    dtype = getattr(torch, torch_dtype)
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def pipe_device(pipe):
    if hasattr(pipe, "device"):
        return pipe.device
    for name in ("transformer", "unet", "text_encoder", "vae"):
        module = getattr(pipe, name, None)
        if module is not None:
            return next(module.parameters()).device
    import torch
    return torch.device("cuda")


def generate_one(pipe, prompt, seed):
    import torch

    generator = torch.Generator(device=pipe_device(pipe)).manual_seed(seed)
    return pipe(prompt, generator=generator).images[0]


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
    print(f"[{tag}] loading {cfg['model_id']} ...")
    pipe = load_pipeline(
        cfg["model_id"],
        cpu_offload=cfg["cpu_offload"],
        torch_dtype=cfg["torch_dtype"],
    )
    print(f"[{tag}] model ready in {time.time() - t0:.1f}s")

    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        t1 = time.time()
        try:
            generate_one(pipe, prompt, cfg["seed"]).save(dst)
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
            print(f"[{tag} {i}/{len(pending)}] {filename} OK"
                  f"  gid={entry['gid']}  [{entry['benchmark']}]  {elapsed:.1f}s")
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

    if cfg["gpu"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    print(f"Loading {cfg['model_id']} ...")
    pipe = load_pipeline(
        cfg["model_id"],
        cpu_offload=cfg["cpu_offload"],
        torch_dtype=cfg["torch_dtype"],
    )

    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        t1 = time.time()
        try:
            generate_one(pipe, prompt, cfg["seed"]).save(dst)
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
            print(f"[{i}/{len(pending)}] {filename} OK"
                  f"  gid={entry['gid']}  [{entry['benchmark']}]  {elapsed:.1f}s")
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
        "model_id": args.model_id,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "cpu_offload": args.cpu_offload,
        "torch_dtype": args.torch_dtype,
        "gpu": gpu,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_json", default=DEFAULT_JSON)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--torch_dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cpu_offload", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids, e.g. 0,1,2,3")
    args = ap.parse_args()

    prompts = load_prompts(args.batch_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(f"[DeepFloyd IF-XL] json={len(prompts)}  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}")
    print(f"  model={args.model_id}  dtype={args.torch_dtype}")
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

    print(f"\nDone DeepFloyd IF-XL: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail*.log in output_dir")


if __name__ == "__main__":
    main()
