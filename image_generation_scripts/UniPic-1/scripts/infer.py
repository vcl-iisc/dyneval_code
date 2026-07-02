#!/usr/bin/env python3
"""Generate UniPic text-to-image outputs for DYNEVAL-1K remaining prompts.

Reads:
    DYNEVAL-1K-REMAINING-PROMPTS.json

Saves:
    DYNEVAL-1K-IMAGES-PART2/unipic/<filename from JSON>

Usage (from DYNEVAL project root or UniPic-1 repo root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/UniPic-1/scripts/infer.py \\
        configs/models/qwen2_5_1_5b_kl16_mar_h.py \\
        --checkpoint checkpoint/pytorch_model.bin

    python dyneval_code/generator_scripts/UniPic-1/scripts/infer.py \\
        configs/models/qwen2_5_1_5b_kl16_mar_h.py \\
        --checkpoint checkpoint/pytorch_model.bin --dry-run --limit 5

    # single prompt (ignores JSON)
    python dyneval_code/generator_scripts/UniPic-1/scripts/infer.py \\
        configs/models/qwen2_5_1_5b_kl16_mar_h.py \\
        --checkpoint checkpoint/pytorch_model.bin \\
        --prompt 'A golden retriever in a park.' --output output.jpg

    # use all visible GPUs (default when 2+ GPUs are visible)
    CUDA_VISIBLE_DEVICES=0,1 python dyneval_code/generator_scripts/UniPic-1/scripts/infer.py \\
        configs/models/qwen2_5_1_5b_kl16_mar_h.py \\
        --checkpoint checkpoint/pytorch_model.bin

    # explicit GPU list
    python dyneval_code/generator_scripts/UniPic-1/scripts/infer.py \\
        configs/models/qwen2_5_1_5b_kl16_mar_h.py \\
        --checkpoint checkpoint/pytorch_model.bin --gpus 0,1
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from einops import rearrange
from mmengine.config import Config
from PIL import Image

from src.builder import BUILDER

DEFAULT_JSON = "DYNEVAL-1K-REMAINING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES-PART2", "unipic-1k-images")


def configure_torch(disable_cudnn=False):
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.enabled = not disable_cudnn
        # Warm up CUDA/cuDNN before the first real forward pass.
        torch.zeros(1, device="cuda").item()


def prepare_model_for_inference(model):
    """Disable training-only paths that break inference (checkpointing/cuDNN)."""
    if hasattr(model, "mar"):
        model.mar.grad_checkpointing = False
        if hasattr(model.mar, "gradient_checkpointing_disable"):
            model.mar.gradient_checkpointing_disable()
        net = getattr(getattr(model.mar, "diffloss", None), "net", None)
        if net is not None:
            net.grad_checkpointing = False
    if hasattr(model, "llm") and hasattr(model.llm, "gradient_checkpointing_disable"):
        model.llm.gradient_checkpointing_disable()
    model.eval()
    return model


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


def resolve_gpus(gpu_str):
    if gpu_str:
        return [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]
    n = torch.cuda.device_count()
    if n >= 2:
        return list(range(n))
    if n == 1:
        return [0]
    return []


def args_namespace(cfg):
    return argparse.Namespace(**cfg)


def build_cfg(args):
    return {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "model_dir": args.model_dir,
        "attn_implementation": args.attn_implementation,
        "disable_cudnn": args.disable_cudnn,
        "output_dir": args.output_dir,
        "cfg_prompt": args.cfg_prompt,
        "cfg": args.cfg,
        "temperature": args.temperature,
        "cfg_schedule": args.cfg_schedule,
        "num_iter": args.num_iter,
        "grid_size": args.grid_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "seed_per_gid": args.seed_per_gid,
    }


def worker_main(worker_id, gpu_id, pending, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tag = f"gpu{gpu_id}"
    manifest = os.path.join(cfg["output_dir"], f"generated_manifest.{tag}.jsonl")
    fail_log = os.path.join(cfg["output_dir"], f"generate_fail.{tag}.log")

    print(f"[{tag}] worker {worker_id}: {len(pending)} prompts", flush=True)
    if not pending:
        return 0, 0

    t0 = time.time()
    print(f"[{tag}] loading model from {cfg['checkpoint']} ...", flush=True)
    model = load_model(
        cfg["config"],
        cfg["checkpoint"],
        cfg["attn_implementation"],
        cfg["model_dir"],
        cfg["disable_cudnn"],
    )
    print(f"[{tag}] model ready in {time.time() - t0:.1f}s", flush=True)

    worker_args = args_namespace(cfg)
    ok, bad = generate_all(
        model, pending, worker_args, cfg,
        manifest=manifest, fail_log=fail_log, tag=tag,
    )
    print(f"[{tag}] done: generated={ok} failed={bad}", flush=True)
    return ok, bad


def run_single_gpu(pending, cfg, gpu=None):
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        tag = f"gpu{gpu}"
        manifest = os.path.join(cfg["output_dir"], f"generated_manifest.{tag}.jsonl")
        fail_log = os.path.join(cfg["output_dir"], f"generate_fail.{tag}.log")
    else:
        tag = None
        manifest = os.path.join(cfg["output_dir"], "generated_manifest.jsonl")
        fail_log = os.path.join(cfg["output_dir"], "generate_fail.log")

    print(f"Loading model from {cfg['checkpoint']} ...", flush=True)
    t0 = time.time()
    model = load_model(
        cfg["config"],
        cfg["checkpoint"],
        cfg["attn_implementation"],
        cfg["model_dir"],
        cfg["disable_cudnn"],
    )
    print(f"Model ready in {time.time() - t0:.1f}s", flush=True)
    return generate_all(
        model, pending, args_namespace(cfg), cfg,
        manifest=manifest, fail_log=fail_log, tag=tag,
    )


def resolve_model_paths(config, checkpoint_path, model_dir=None):
    """Point VAE/LLM/SigLIP paths at the directory that contains pytorch_model.bin."""
    if model_dir is None:
        model_dir = os.path.dirname(os.path.abspath(checkpoint_path))

    paths = {
        "kl16.ckpt": os.path.join(model_dir, "kl16.ckpt"),
        "Qwen2.5-1.5B-Instruct": os.path.join(model_dir, "Qwen2.5-1.5B-Instruct"),
        "siglip2-so400m-patch16-512": os.path.join(model_dir, "siglip2-so400m-patch16-512"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {name} under model dir {model_dir}. "
                f"Expected: {path}"
            )

    config.model.vae.ckpt_path = paths["kl16.ckpt"]
    config.model.llm.pretrained_model_name_or_path = paths["Qwen2.5-1.5B-Instruct"]
    config.model.siglip2.pretrained_model_name_or_path = paths["siglip2-so400m-patch16-512"]
    config.model.tokenizer.pretrained_model_name_or_path = paths["Qwen2.5-1.5B-Instruct"]
    return model_dir


def warn_cuda_env():
    if "CUDA_VISIBLE_DEVICE" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        print(
            "WARNING: CUDA_VISIBLE_DEVICE is ignored by CUDA/PyTorch. "
            "Use CUDA_VISIBLE_DEVICES instead.",
            flush=True,
        )


def load_model(config_path, checkpoint_path, attn_implementation="eager", model_dir=None,
               disable_cudnn=False):
    configure_torch(disable_cudnn=disable_cudnn)
    config = Config.fromfile(config_path)
    model_dir = resolve_model_paths(config, checkpoint_path, model_dir)
    print(f"Model components from: {model_dir}", flush=True)
    if attn_implementation:
        config.model.llm.attn_implementation = attn_implementation
    config.model.mar.grad_checkpointing = False
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return prepare_model_for_inference(model)


def generate_image(model, prompt, args, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    formatted = f"Generate an image: {prompt.strip()}"
    class_info = model.prepare_text_conditions(formatted, args.cfg_prompt)

    input_ids = class_info["input_ids"]
    attention_mask = class_info["attention_mask"]

    assert len(input_ids) == 2
    if args.cfg == 1.0:
        input_ids = input_ids[:1]
        attention_mask = attention_mask[:1]

    bsz = args.grid_size ** 2
    if args.cfg != 1.0:
        input_ids = torch.cat([
            input_ids[:1].expand(bsz, -1),
            input_ids[1:].expand(bsz, -1),
        ])
        attention_mask = torch.cat([
            attention_mask[:1].expand(bsz, -1),
            attention_mask[1:].expand(bsz, -1),
        ])
    else:
        input_ids = input_ids.expand(bsz, -1)
        attention_mask = attention_mask.expand(bsz, -1)

    m = n = args.image_size // 16
    samples = model.sample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_iter=args.num_iter,
        cfg=args.cfg,
        cfg_schedule=args.cfg_schedule,
        temperature=args.temperature,
        progress=False,
        image_shape=(m, n),
    )
    samples = rearrange(
        samples, "(m n) c h w -> (m h) (n w) c", m=args.grid_size, n=args.grid_size)
    samples = torch.clamp(
        127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(samples)


def write_manifest(manifest, entry, seconds, seed, cfg, gpu=None):
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
            "checkpoint": cfg["checkpoint"],
            "cfg": cfg["cfg"],
            "num_iter": cfg["num_iter"],
            "image_size": cfg["image_size"],
            "grid_size": cfg["grid_size"],
        }
        if gpu is not None:
            row["gpu"] = gpu
        mf.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_failure(fail_log, entry, exc):
    with open(fail_log, "a", encoding="utf-8") as lf:
        lf.write(f"{entry['gid']}\t{entry['filename']}\t{entry['prompt']}\t{exc}\n")
        lf.write(traceback.format_exc() + "\n")


def is_cudnn_init_error(exc):
    return "CUDNN_STATUS_NOT_INITIALIZED" in str(exc)


def disable_cudnn_after_failure():
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_all(model, pending, args, cfg, manifest=None, fail_log=None, tag=None):
    os.makedirs(args.output_dir, exist_ok=True)
    if manifest is None:
        manifest = os.path.join(args.output_dir, "generated_manifest.jsonl")
    if fail_log is None:
        fail_log = os.path.join(args.output_dir, "generate_fail.log")

    ok = bad = 0
    total = len(pending)
    gpu_id = tag.replace("gpu", "") if tag else None

    for i, entry in enumerate(pending, 1):
        dst = os.path.join(args.output_dir, entry["filename"])
        seed = args.seed + int(entry["gid"]) if args.seed_per_gid else args.seed
        t0 = time.time()
        prefix = f"[{tag} " if tag else "["
        try:
            try:
                image = generate_image(model, entry["prompt"], args, seed=seed)
            except RuntimeError as exc:
                if args.disable_cudnn or not is_cudnn_init_error(exc):
                    raise
                print(f"{prefix.rstrip()} cuDNN init failed; disabling cuDNN and retrying once.",
                      flush=True)
                args.disable_cudnn = True
                disable_cudnn_after_failure()
                image = generate_image(model, entry["prompt"], args, seed=seed)
            image.save(dst)
            elapsed = time.time() - t0
            write_manifest(manifest, entry, elapsed, seed, cfg, gpu=gpu_id)
            ok += 1
            print(f"{prefix}{i}/{total}] {entry['filename']} OK"
                  f"  [{entry.get('benchmark')}]  {elapsed:.1f}s", flush=True)
        except Exception as exc:
            bad += 1
            write_failure(fail_log, entry, exc)
            print(f"{prefix}{i}/{total}] {entry['filename']} FAILED: {exc}", flush=True)

    return ok, bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file path")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory with kl16.ckpt, Qwen2.5-1.5B-Instruct/, siglip2-so400m-patch16-512/ "
        "(default: parent directory of --checkpoint).",
    )
    parser.add_argument("--prompt_json", default=DEFAULT_JSON,
                        help="DYNEVAL-1K remaining prompts JSON")
    parser.add_argument("--output_dir", default=DEFAULT_OUT,
                        help="where to save PNGs")
    parser.add_argument("--prompt", type=str, default=None,
                        help="single prompt (skips JSON batch mode)")
    parser.add_argument("--output", type=str, default="output.jpg",
                        help="output path for single --prompt mode")
    parser.add_argument("--cfg_prompt", type=str, default="Generate an image.")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_schedule", type=str, default="constant")
    parser.add_argument("--num_iter", type=int, default=32)
    parser.add_argument("--grid_size", type=int, default=1,
                        help="grid side length (1 = single 1024x1024 image)")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="disable cuDNN; useful if the driver reports CUDNN_STATUS_NOT_INITIALIZED",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="LLM attention backend (default: eager, no flash-attn).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed_per_gid", action="store_true",
                        help="use seed + gid for deterministic per-image variation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpus",
        default=None,
        help="comma-separated GPU ids (default: use all visible GPUs when 2+ are available)",
    )
    args = parser.parse_args()
    warn_cuda_env()

    if args.prompt is not None:
        print(f"Single-prompt mode: {args.prompt[:90]}")
        print(f"  output={args.output}")
        model = load_model(
            args.config, args.checkpoint, args.attn_implementation, args.model_dir,
            args.disable_cudnn)
        image = generate_image(model, args.prompt, args, seed=args.seed)
        image.save(args.output)
        print(f"Saved {args.output}")
        return

    prompts = load_prompts(args.prompt_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(f"[UniPic] json={len(prompts)}"
          f"  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}  -> {args.output_dir}")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  cfg={args.cfg}  num_iter={args.num_iter}"
          f"  size={args.image_size}  grid_size={args.grid_size}  seed={args.seed}"
          f"  attn={args.attn_implementation}")

    gpus = resolve_gpus(args.gpus)
    if gpus:
        print(f"  gpus={gpus}  visible_cuda_devices={torch.cuda.device_count()}", flush=True)
        if len(gpus) > 1 and torch.cuda.device_count() < len(gpus):
            print(
                "WARNING: fewer visible CUDA devices than requested GPUs. "
                "Set CUDA_VISIBLE_DEVICES=0,1 (note the trailing S) to expose both GPUs.",
                flush=True,
            )

    if args.limit:
        pending = pending[:args.limit]
        print(f"  limited to {len(pending)} this run")

    if not pending:
        print("Nothing to do.")
        return

    if args.dry_run:
        if gpus and len(gpus) > 1:
            for rank, gpu in enumerate(gpus):
                part = shard(pending, len(gpus), rank)
                print(f"  GPU {gpu}: {len(part)} prompts", flush=True)
        for entry in pending[:10]:
            print(f"  gid={entry['gid']}  {entry['filename']}"
                  f"  [{entry.get('benchmark')}]  {entry['prompt'][:90]}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return

    cfg = build_cfg(args)
    if gpus and len(gpus) > 1:
        jobs = [
            (rank, gpu, shard(pending, len(gpus), rank), cfg)
            for rank, gpu in enumerate(gpus)
            if shard(pending, len(gpus), rank)
        ]
        print(f"Launching {len(jobs)} GPU workers on {gpus} ...", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(jobs)) as pool:
            results = pool.starmap(worker_main, jobs)
        ok = sum(r[0] for r in results)
        bad = sum(r[1] for r in results)
    else:
        gpu = gpus[0] if gpus else None
        ok, bad = run_single_gpu(pending, cfg, gpu=gpu)

    print(f"\nDone UniPic: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail*.log in output_dir")


if __name__ == "__main__":
    main()
