#!/usr/bin/env python3
"""Generate remaining DYNEVAL-1K FIBO images from the shared prompts JSON.

Reads prompts from:
    DYNEVAL-UPDATED-1K-EXPERIMENT/prompts/DYNEVAL-1K-FIXED-PROMPTS-SHYAM-REMAINING.json

Saves images to:
    DYNEVAL-1K-IMAGES/Fibo/<filename>

Uses all available GPUs by default (one worker per GPU).

Usage (from DYNEVAL project root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/FIBO/infer.py --dry-run
    python dyneval_code/generator_scripts/FIBO/infer.py --limit 5
    python dyneval_code/generator_scripts/FIBO/infer.py
    python dyneval_code/generator_scripts/FIBO/infer.py --gpus 0,1
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

import torch
from diffusers import BriaFiboPipeline
from diffusers.modular_pipelines import modular_pipeline as modular_pipeline_module
from diffusers.modular_pipelines import ModularPipelineBlocks

DEFAULT_JSON = os.path.join(
    "DYNEVAL-UPDATED-1K-EXPERIMENT/prompts",
    "DYNEVAL-1K-FIXED-PROMPTS-SHYAM-REMAINING.json",
)
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES", "Fibo")
VLM_MODEL = "briaai/FIBO-VLM-prompt-to-JSON"
IMAGE_MODEL = "briaai/FIBO"


def get_default_negative_prompt(existing_json: dict) -> str:
    negative_prompt = ""
    style_medium = existing_json.get("style_medium", "").lower()
    if style_medium in ["photograph", "photography", "photo"]:
        negative_prompt = """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
    return negative_prompt


def patch_modular_diffusers_requirements():
    """Accept older remote modular_config requirements encoded as list pairs."""
    original_validate = modular_pipeline_module._validate_requirements

    def validate_requirements_compat(reqs):
        if isinstance(reqs, list):
            reqs = {
                name: version if str(version).startswith(("=", "<", ">", "~", "!")) else f"=={version}"
                for name, version in reqs
            }
        return original_validate(reqs)

    modular_pipeline_module._validate_requirements = validate_requirements_compat


def load_pipelines(cpu_offload=False):
    patch_modular_diffusers_requirements()
    torch.set_grad_enabled(False)

    vlm_pipe = ModularPipelineBlocks.from_pretrained(
        VLM_MODEL,
        trust_remote_code=True,
    )
    vlm_pipe = vlm_pipe.init_pipeline()

    pipe = BriaFiboPipeline.from_pretrained(
        IMAGE_MODEL,
        torch_dtype=torch.bfloat16,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return vlm_pipe, pipe


def generate_one(vlm_pipe, pipe, prompt, num_steps, guidance_scale):
    output = vlm_pipe(prompt=prompt)
    json_prompt_generate = output.values["json_prompt"]
    parsed_json_prompt = json.loads(json_prompt_generate)
    negative_prompt = get_default_negative_prompt(parsed_json_prompt)

    results_generate = pipe(
        prompt=json_prompt_generate,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )
    return results_generate.images[0], json_prompt_generate


def load_prompts(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", data)
    return sorted(prompts, key=lambda p: int(p["gid"]))


def filter_pending(prompts, output_dir, overwrite):
    if overwrite:
        return prompts
    return [
        p
        for p in prompts
        if not os.path.exists(os.path.join(output_dir, p["filename"]))
    ]


def shard(items, n_shards, shard_id):
    return items[shard_id::n_shards]


def json_prompt_path(output_dir, entry):
    stem = os.path.splitext(entry["filename"])[0]
    return os.path.join(output_dir, f"{stem}_json_prompt.json")


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
    print(f"[{tag}] loading FIBO pipelines ...")
    vlm_pipe, pipe = load_pipelines(cpu_offload=cfg["cpu_offload"])
    print(f"[{tag}] models ready in {time.time() - t0:.1f}s")

    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        json_dst = json_prompt_path(cfg["output_dir"], entry)
        t1 = time.time()
        try:
            image, json_prompt_generate = generate_one(
                vlm_pipe,
                pipe,
                prompt,
                cfg["num_inference_steps"],
                cfg["guidance_scale"],
            )
            image.save(dst)
            with open(json_dst, "w", encoding="utf-8") as jf:
                jf.write(json_prompt_generate)

            ok += 1
            elapsed = time.time() - t1
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(
                    json.dumps(
                        {
                            "gid": entry["gid"],
                            "filename": filename,
                            "benchmark": entry.get("benchmark"),
                            "model": IMAGE_MODEL,
                            "prompt": prompt,
                            "output_path": dst,
                            "json_prompt_path": json_dst,
                            "seconds": round(elapsed, 2),
                            "gpu": gpu_id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(
                f"[{tag} {i}/{len(pending)}] {filename} OK"
                f"  [{entry.get('benchmark', '')}]  {elapsed:.1f}s"
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

    if cfg["gpu"] is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    print("Loading FIBO pipelines ...")
    vlm_pipe, pipe = load_pipelines(cpu_offload=cfg["cpu_offload"])

    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        json_dst = json_prompt_path(cfg["output_dir"], entry)
        t1 = time.time()
        try:
            image, json_prompt_generate = generate_one(
                vlm_pipe,
                pipe,
                prompt,
                cfg["num_inference_steps"],
                cfg["guidance_scale"],
            )
            image.save(dst)
            with open(json_dst, "w", encoding="utf-8") as jf:
                jf.write(json_prompt_generate)

            ok += 1
            elapsed = time.time() - t1
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(
                    json.dumps(
                        {
                            "gid": entry["gid"],
                            "filename": filename,
                            "benchmark": entry.get("benchmark"),
                            "model": IMAGE_MODEL,
                            "prompt": prompt,
                            "output_path": dst,
                            "json_prompt_path": json_dst,
                            "seconds": round(elapsed, 2),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(
                f"[{i}/{len(pending)}] {filename} OK"
                f"  [{entry.get('benchmark', '')}]  {elapsed:.1f}s"
            )
        except Exception as exc:
            bad += 1
            with open(fail_log, "a", encoding="utf-8") as lf:
                lf.write(f"{entry['gid']}\t{filename}\t{prompt}\t{exc}\n")
                lf.write(traceback.format_exc() + "\n")
            print(f"[{i}/{len(pending)}] {filename} FAILED: {exc}")

    return ok, bad


def parse_gpus(gpu_str):
    if gpu_str:
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]
    count = torch.cuda.device_count()
    if count == 0:
        raise RuntimeError("No CUDA GPUs available.")
    return list(range(count))


def build_cfg(args, gpu=None):
    return {
        "output_dir": args.output_dir,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "cpu_offload": args.cpu_offload,
        "gpu": gpu,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_json", default=DEFAULT_JSON)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable CPU offload (slower; use if a single GPU OOMs)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU ids (default: all visible GPUs)",
    )
    args = ap.parse_args()

    prompts = load_prompts(args.batch_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(
        f"[FIBO] json={len(prompts)}  already done={len(prompts) - len(pending)}"
        f"  to generate={len(pending)}"
    )
    print(f"  output -> {args.output_dir}")

    if args.limit:
        pending = pending[: args.limit]
        print(f"  limited to {len(pending)} this run")

    if not pending:
        print("Nothing to do.")
        return

    gpus = parse_gpus(args.gpus)
    print(f"  gpus -> {gpus}")

    if args.dry_run:
        for rank, gpu in enumerate(gpus):
            part = shard(pending, len(gpus), rank)
            print(f"  GPU {gpu}: {len(part)} prompts")
        for p in pending[:10]:
            print(
                f"  gid={p['gid']}  {p['filename']}  [{p.get('benchmark', '')}]"
                f"  {p['prompt'][:80]}"
            )
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return

    if len(gpus) > 1:
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
        ok, bad = run_single_gpu(pending, build_cfg(args, gpu=gpus[0]))

    print(f"\nDone FIBO: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail*.log in output_dir")


if __name__ == "__main__":
    main()
