#!/usr/bin/env python3
"""Generate Qwen-Image outputs for DYNEVAL-1K remaining prompts.

Reads DYNEVAL-1K-REMAINING-PROMPTS.json and saves each image using the
filename from the JSON (e.g. 1001.png) under:
    DYNEVAL-1K-IMAGES-PART2/QWEN-IMAGE/

Usage:
    python3 qwenimage.py --dry-run
    python3 qwenimage.py --limit 5
    python3 qwenimage.py --aspect_ratio 1:1
    python3 qwenimage.py --overwrite
"""
import argparse
import json
import os
import time
import traceback

DEFAULT_JSON = "DYNEVAL-1K-REMAINING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES-PART2", "QWEN-IMAGE")
DEFAULT_MODEL = "Qwen/Qwen-Image"

POSITIVE_MAGIC = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图.",
}

ASPECT_RATIOS = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}


def configure_torch():
    import torch

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_device():
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16, "cuda"
    return torch.float32, "cpu"


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


def magic_lang(prompt, lang):
    if lang != "auto":
        return lang
    return "zh" if any("\u4e00" <= c <= "\u9fff" for c in prompt) else "en"


def build_prompt(prompt, lang="auto", use_magic=True):
    if not use_magic:
        return prompt
    return prompt + POSITIVE_MAGIC[magic_lang(prompt, lang)]


def load_pipeline(model_name_or_path, dtype, device, cpu_offload):
    from diffusers import DiffusionPipeline

    configure_torch()
    pipe = DiffusionPipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
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
            "true_cfg_scale": cfg["true_cfg_scale"],
            "negative_prompt": cfg["negative_prompt"],
            "aspect_ratio": cfg["aspect_ratio"],
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
    gen_device = cfg["device"] if cfg["device"] != "cpu" else "cpu"

    for i, entry in enumerate(pending, 1):
        dst = os.path.join(cfg["output_dir"], entry["filename"])
        seed = cfg["seed"] + int(entry["gid"]) if cfg["seed_per_gid"] else cfg["seed"]
        prompt = build_prompt(entry["prompt"], cfg["magic_lang"], cfg["use_magic"])
        t0 = time.time()
        try:
            with torch.inference_mode():
                image = pipe(
                    prompt=prompt,
                    negative_prompt=cfg["negative_prompt"],
                    width=cfg["width"],
                    height=cfg["height"],
                    num_inference_steps=cfg["num_inference_steps"],
                    true_cfg_scale=cfg["true_cfg_scale"],
                    generator=torch.Generator(device=gen_device).manual_seed(seed),
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


def build_cfg(args, width, height):
    return {
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "device": args.device,
        "num_inference_steps": args.num_inference_steps,
        "true_cfg_scale": args.true_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "aspect_ratio": args.aspect_ratio,
        "height": height,
        "width": width,
        "seed": args.seed,
        "seed_per_gid": args.seed_per_gid,
        "magic_lang": args.magic_lang,
        "use_magic": not args.no_magic,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_json", default=DEFAULT_JSON,
                    help="DYNEVAL-1K remaining prompts JSON")
    ap.add_argument("--output_dir", default=DEFAULT_OUT,
                    help="where to save PNGs")
    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None,
                    help="cuda or cpu (default: cuda if available else cpu)")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"),
                    default=None)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--true_cfg_scale", type=float, default=4.0)
    ap.add_argument("--negative_prompt", default=" ",
                    help="negative prompt; use a single space if unused")
    ap.add_argument("--aspect_ratio", choices=tuple(ASPECT_RATIOS),
                    default="1:1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seed_per_gid", action="store_true",
                    help="use seed + gid for deterministic per-image variation")
    ap.add_argument("--magic_lang", choices=("auto", "en", "zh"), default="auto",
                    help="language suffix appended to each prompt")
    ap.add_argument("--no_magic", action="store_true",
                    help="do not append the Qwen positive magic suffix")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--cpu_offload", action="store_true",
                    help="reduce VRAM use, usually slower")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import torch

    default_dtype, default_device = resolve_device()
    if args.device is None:
        args.device = default_device
    if args.dtype is None:
        args.dtype = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }[default_dtype]

    width, height = ASPECT_RATIOS[args.aspect_ratio]
    prompts = load_prompts(args.prompt_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(f"[Qwen-Image] json={len(prompts)}"
          f"  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}  -> {args.output_dir}")
    print(f"  model={args.model_name_or_path}")
    print(f"  steps={args.num_inference_steps}  true_cfg_scale={args.true_cfg_scale}"
          f"  aspect={args.aspect_ratio} ({width}x{height})  seed={args.seed}")

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

    ok, bad = generate_all(pipe, pending, build_cfg(args, width, height))
    print(f"\nDone Qwen-Image: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail.log in output_dir")


if __name__ == "__main__":
    main()
