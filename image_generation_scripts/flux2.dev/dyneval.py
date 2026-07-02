#!/usr/bin/env python3
"""Generate FLUX.2-dev images for DYNEVAL-1K remaining prompts.

Uses the local FLUX.2-dev checkpoint by default. Pass --remote_encoder
with diffusers/FLUX.2-dev-bnb-4bit if you want the HF remote text encoder.

Reads:
    DYNEVAL-1K-REMAINING-PROMPTS.json

Saves:
    DYNEVAL-1K-IMAGES-PART2/flux2-dev/<filename from JSON>

Setup:
    huggingface-cli login
    # or: export HF_TOKEN="..."

By default, splits the model across all visible CUDA GPUs using
device_map="balanced" (needed when the full model does not fit on one GPU).
Use --device_map single to load everything on cuda:0 instead.

Usage:
    python3 flux2-dev.py --dry-run
    python3 flux2-dev.py --limit 5
    python3 flux2-dev.py --device_map balanced --max_memory 78GB,78GB
    python3 flux2-dev.py --overwrite
"""
import argparse
import io
import json
import os
import re
import sys
import time
import traceback

import requests
import torch
from huggingface_hub import get_token

DEFAULT_JSON = "DYNEVAL-1K-REMAINING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES-PART2", "flux2-dev")
DEFAULT_MODEL = "./FLUX2-DEV-MODEL"
REMOTE_ENCODER_URL = "https://remote-text-encoder-flux-2.huggingface.co/predict"
GENERATOR_DEVICE = "cuda:0"


def configure_torch():
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


def parse_max_memory(spec, n_gpus):
    """Parse '78GB,78GB' into {0: '78GB', 1: '78GB'} for accelerate."""
    if spec is None:
        return {i: "78GiB" for i in range(n_gpus)}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) == 1 and n_gpus > 1:
        parts = parts * n_gpus
    if len(parts) != n_gpus:
        sys.exit(
            f"--max_memory expects {n_gpus} values (one per GPU), got {len(parts)}: {spec!r}"
        )
    return {i: parts[i] for i in range(n_gpus)}


def resolve_device_map(requested, n_gpus):
    if requested == "auto":
        return "auto"
    if requested == "single":
        return "single"
    if requested == "balanced":
        if n_gpus < 2:
            print("WARNING: --device_map balanced needs 2+ GPUs; falling back to single GPU.")
            return "single"
        return "balanced"
    # default
    return "balanced" if n_gpus >= 2 else "single"


def print_gpu_memory(prefix=""):
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        print(f"{prefix}GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def response_excerpt(content, limit=700):
    text = content[:6000].decode("utf-8", errors="replace")
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def get_hf_token(explicit=None):
    token = explicit or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            token = get_token()
        except Exception:
            token = None
    if not token:
        sys.exit("Missing HuggingFace token. Run `huggingface-cli login` or set HF_TOKEN.")
    return token


def remote_text_encoder(prompt, device, hf_token, max_wait=900, retry_interval=10):
    deadline = time.time() + max_wait
    attempt = 0

    while True:
        attempt += 1
        response = requests.post(
            REMOTE_ENCODER_URL,
            json={"prompt": [prompt]},
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
            },
            timeout=300,
        )
        response.raise_for_status()
        if not response.content:
            raise RuntimeError("Remote text encoder returned an empty response.")

        content_type = response.headers.get("content-type", "unknown")
        is_html = "text/html" in content_type or response.content.lstrip().startswith(b"<")
        if is_html:
            excerpt = response_excerpt(response.content)
            if "Preparing Space" in excerpt and time.time() < deadline:
                remaining = int(deadline - time.time())
                print(
                    "Remote text encoder Space is still preparing; "
                    f"retrying in {retry_interval}s ({remaining}s left, attempt {attempt}) ..."
                )
                time.sleep(retry_interval)
                continue
            debug_path = os.path.abspath("remote_text_encoder_response.html")
            with open(debug_path, "wb") as f:
                f.write(response.content)
            raise RuntimeError(
                "Remote text encoder returned HTML instead of prompt embeddings. "
                f"Saved response to {debug_path}. "
                f"content-type={content_type}; excerpt={excerpt!r}"
            )

        try:
            prompt_embeds = torch.load(
                io.BytesIO(response.content),
                map_location=device,
                weights_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to deserialize remote text encoder response "
                f"({len(response.content)} bytes, content-type={content_type})."
            ) from exc
        return prompt_embeds.to(device)


def load_pipeline(model_name_or_path, remote_encoder, device_map, max_memory, attention_slicing):
    from diffusers import Flux2Pipeline

    configure_torch()
    kwargs = {"torch_dtype": torch.bfloat16}
    if remote_encoder:
        kwargs["text_encoder"] = None

    if device_map in ("balanced", "auto"):
        kwargs["device_map"] = device_map
        if max_memory:
            kwargs["max_memory"] = max_memory
        print(f"Loading {model_name_or_path} with device_map={device_map!r} ...")
        if max_memory:
            print(f"  max_memory={max_memory}")
        pipe = Flux2Pipeline.from_pretrained(model_name_or_path, **kwargs)
        if getattr(pipe, "hf_device_map", None):
            print(f"  component placement: {pipe.hf_device_map}")
    else:
        print(f"Loading {model_name_or_path} on {GENERATOR_DEVICE} ...")
        pipe = Flux2Pipeline.from_pretrained(model_name_or_path, **kwargs).to(GENERATOR_DEVICE)

    if attention_slicing:
        pipe.enable_attention_slicing()
        print("  attention_slicing=enabled")

    print_gpu_memory("  ")
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
            "device_map": cfg["device_map"],
        }, ensure_ascii=False) + "\n")


def write_failure(fail_log, entry, exc):
    with open(fail_log, "a", encoding="utf-8") as lf:
        lf.write(f"{entry['gid']}\t{entry['filename']}\t{entry['prompt']}\t{exc}\n")
        lf.write(traceback.format_exc() + "\n")


def generate_all(pipe, pending, cfg, hf_token=None):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    manifest = os.path.join(cfg["output_dir"], "generated_manifest.jsonl")
    fail_log = os.path.join(cfg["output_dir"], "generate_fail.log")
    device = GENERATOR_DEVICE

    ok = bad = 0
    total = len(pending)

    for i, entry in enumerate(pending, 1):
        dst = os.path.join(cfg["output_dir"], entry["filename"])
        seed = cfg["seed"] + int(entry["gid"]) if cfg["seed_per_gid"] else cfg["seed"]
        t0 = time.time()
        try:
            pipe_kwargs = {
                "height": cfg["height"],
                "width": cfg["width"],
                "generator": torch.Generator(device=device).manual_seed(seed),
                "num_inference_steps": cfg["num_inference_steps"],
                "guidance_scale": cfg["guidance_scale"],
            }
            if cfg["remote_encoder"]:
                pipe_kwargs["prompt_embeds"] = remote_text_encoder(
                    entry["prompt"],
                    device,
                    hf_token,
                    max_wait=cfg["remote_encoder_timeout"],
                    retry_interval=cfg["remote_encoder_retry_interval"],
                )
            else:
                pipe_kwargs["prompt"] = entry["prompt"]

            with torch.inference_mode():
                image = pipe(**pipe_kwargs).images[0]
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


def build_cfg(args, device_map):
    return {
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "device_map": device_map,
        "remote_encoder": args.remote_encoder,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "seed": args.seed,
        "seed_per_gid": args.seed_per_gid,
        "remote_encoder_timeout": args.remote_encoder_timeout,
        "remote_encoder_retry_interval": args.remote_encoder_retry_interval,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_json", default=DEFAULT_JSON,
                    help="DYNEVAL-1K remaining prompts JSON")
    ap.add_argument("--output_dir", default=DEFAULT_OUT,
                    help="where to save PNGs")
    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    ap.add_argument("--remote_encoder", action="store_true",
                    help="use HF remote text encoder (for bnb-4bit; needs HF_TOKEN)")
    ap.add_argument("--device_map", choices=("balanced", "auto", "single"), default=None,
                    help="how to place model weights (default: balanced when 2+ GPUs)")
    ap.add_argument("--max_memory", default=None,
                    help="per-GPU memory cap, comma-separated (default: 78GiB each)")
    ap.add_argument("--attention_slicing", action="store_true",
                    help="enable attention slicing to reduce peak VRAM")
    ap.add_argument("--hf_token", default=None)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=4.0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_per_gid", action="store_true",
                    help="use seed + gid for deterministic per-image variation")
    ap.add_argument("--remote_encoder_timeout", type=int, default=900,
                    help="seconds to wait for the HF remote text encoder Space to start")
    ap.add_argument("--remote_encoder_retry_interval", type=int, default=10,
                    help="seconds between remote text encoder startup polls")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map = resolve_device_map(args.device_map or "balanced", n_gpus)
    max_memory = parse_max_memory(args.max_memory, n_gpus) if device_map != "single" else None

    prompts = load_prompts(args.prompt_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(f"[FLUX.2-dev] json={len(prompts)}"
          f"  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}  -> {args.output_dir}")
    print(f"  model={args.model_name_or_path}")
    print(f"  remote_encoder={args.remote_encoder}")
    print(f"  device_map={device_map}  gpus={n_gpus}")
    if max_memory:
        print(f"  max_memory={max_memory}")
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

    hf_token = get_hf_token(args.hf_token) if args.remote_encoder else None

    t0 = time.time()
    pipe = load_pipeline(
        args.model_name_or_path,
        remote_encoder=args.remote_encoder,
        device_map=device_map,
        max_memory=max_memory,
        attention_slicing=args.attention_slicing,
    )
    print(f"Model ready in {time.time() - t0:.1f}s")

    ok, bad = generate_all(pipe, pending, build_cfg(args, device_map), hf_token)
    print(f"\nDone FLUX.2-dev: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail.log in output_dir")


if __name__ == "__main__":
    main()
