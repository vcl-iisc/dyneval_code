#!/usr/bin/env python3
"""Generate HiDream-I1 images for DYNEVAL-1K remaining prompts.

Reads DYNEVAL-1K-REMAINING-PROMPTS.json and saves each image using the
filename from the JSON (e.g. 1001.png) under:
    DYNEVAL-1K-IMAGES-PART2/HI-DREAM-I1/

Requires hi_diffusers (HiDream-I1 repo) and a local Llama-3.1-8B-Instruct
checkpoint passed via --llama_path or cached from Hugging Face.

Usage:
    python3 hi-dream.py --dry-run
    python3 hi-dream.py --limit 5
    python3 hi-dream.py --gpus 0,1,2,3
    python3 hi-dream.py --llama_path /path/to/Meta-Llama-3.1-8B-Instruct
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

DEFAULT_JSON = "DYNEVAL-1K-REMAINING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES-PART2", "HI-DREAM-I1")
DEFAULT_MODEL = "./HiDream-I1-Full-model"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MODEL_CONFIGS = {
    "dev": {
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": "FlashFlowMatchEulerDiscreteScheduler",
    },
    "full": {
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": "FlowUniPCMultistepScheduler",
    },
    "fast": {
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": "FlashFlowMatchEulerDiscreteScheduler",
    },
}


def get_scheduler_cls(name):
    from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler

    return {
        "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler,
        "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler,
    }[name]


def load_models(model_type, model_path, llama_path=None):
    import torch
    from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

    config = MODEL_CONFIGS[model_type]
    llama_model_name_or_path = llama_path or LLAMA_MODEL_NAME
    scheduler_cls = get_scheduler_cls(config["scheduler"])
    scheduler = scheduler_cls(
        num_train_timesteps=1000,
        shift=config["shift"],
        use_dynamic_shifting=False,
    )

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
        llama_model_name_or_path,
        use_fast=False,
    )
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        llama_model_name_or_path,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer

    return pipe, config


def generate_image(pipe, config, prompt, height, width, seed):
    import torch

    if seed < 0:
        seed = torch.randint(0, 1_000_000, (1,)).item()

    generator = torch.Generator("cuda").manual_seed(seed)
    images = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=config["guidance_scale"],
        num_inference_steps=config["num_inference_steps"],
        num_images_per_prompt=1,
        generator=generator,
    ).images
    return images[0], seed


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
    if gpu_str is None:
        return None
    gpus = [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]
    if gpus:
        return gpus
    import torch

    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return None


def write_manifest(manifest, entry, seconds, seed, cfg):
    with open(manifest, "a", encoding="utf-8") as mf:
        row = {
            "gid": entry["gid"],
            "filename": entry["filename"],
            "benchmark": entry.get("benchmark"),
            "category": entry.get("category"),
            "subcategory": entry.get("subcategory"),
            "prompt": entry["prompt"],
            "seconds": round(seconds, 2),
            "seed": seed,
            "model_type": cfg["model_type"],
            "model_path": cfg["model_path"],
        }
        if cfg.get("gpu") is not None:
            row["gpu"] = cfg["gpu"]
        mf.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_failure(fail_log, entry, exc):
    with open(fail_log, "a", encoding="utf-8") as lf:
        lf.write(f"{entry['gid']}\t{entry['filename']}\t{entry['prompt']}\t{exc}\n")
        lf.write(traceback.format_exc() + "\n")


def generate_entries(pipe, model_config, pending, cfg, manifest, fail_log, tag=None):
    ok = bad = 0
    total = len(pending)

    for i, entry in enumerate(pending, 1):
        dst = os.path.join(cfg["output_dir"], entry["filename"])
        seed = cfg["seed"] + int(entry["gid"]) if cfg["seed_per_gid"] else cfg["seed"]
        t0 = time.time()
        try:
            image, used_seed = generate_image(
                pipe,
                model_config,
                prompt=entry["prompt"],
                height=cfg["height"],
                width=cfg["width"],
                seed=seed,
            )
            image.save(dst)
            elapsed = time.time() - t0
            write_manifest(manifest, entry, elapsed, used_seed, cfg)
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
    print(f"[{tag}] loading HiDream model_type={cfg['model_type']} ...")
    print(f"[{tag}] model_path={cfg['model_path']}")
    pipe, model_config = load_models(
        cfg["model_type"],
        model_path=cfg["model_path"],
        llama_path=cfg["llama_path"],
    )
    print(f"[{tag}] model ready in {time.time() - t0:.1f}s")

    cfg = dict(cfg, gpu=gpu_id)
    ok, bad = generate_entries(
        pipe, model_config, pending, cfg, manifest, fail_log, tag=tag,
    )
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

    print(f"Loading HiDream model_type={cfg['model_type']} ...")
    print(f"model_path={cfg['model_path']}")
    pipe, model_config = load_models(
        cfg["model_type"],
        model_path=cfg["model_path"],
        llama_path=cfg["llama_path"],
    )

    return generate_entries(
        pipe, model_config, pending, cfg, manifest, fail_log, tag=tag,
    )


def build_cfg(args, gpu=None):
    return {
        "model_type": args.model_type,
        "model_path": args.model_path,
        "llama_path": args.llama_path,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "seed_per_gid": args.seed_per_gid,
        "height": args.height,
        "width": args.width,
        "gpu": gpu,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_json", default=DEFAULT_JSON,
                    help="DYNEVAL-1K remaining prompts JSON")
    ap.add_argument("--output_dir", default=DEFAULT_OUT,
                    help="where to save PNGs")
    ap.add_argument("--model_type", choices=["dev", "full", "fast"], default="full",
                    help="controls steps/guidance/scheduler")
    ap.add_argument("--model_path", default=DEFAULT_MODEL,
                    help="local HiDream checkpoint dir (must contain transformer/)")
    ap.add_argument("--llama_path", default=None,
                    help="local Llama path; default uses HF cache for Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed seed per image; use -1 for random")
    ap.add_argument("--seed_per_gid", action="store_true",
                    help="use seed + gid for deterministic per-image variation")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids; default uses all visible GPUs")
    args = ap.parse_args()

    prompts = load_prompts(args.prompt_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)
    cfg_info = MODEL_CONFIGS[args.model_type]

    print(f"[HiDream-I1] model_type={args.model_type}  path={args.model_path}")
    print(f"  steps={cfg_info['num_inference_steps']}  guidance={cfg_info['guidance_scale']}"
          f"  size={args.width}x{args.height}")
    print(f"  json={len(prompts)}  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}")
    print(f"  output -> {args.output_dir}")

    if args.limit:
        pending = pending[:args.limit]
        print(f"  limited to {len(pending)} this run")

    if not pending:
        print("Nothing to do.")
        return

    gpus = parse_gpus(args.gpus)

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
        ok, bad = run_single_gpu(pending, build_cfg(args), gpu=gpu)

    print(f"\nDone HiDream-I1: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail*.log in output_dir")


if __name__ == "__main__":
    main()
