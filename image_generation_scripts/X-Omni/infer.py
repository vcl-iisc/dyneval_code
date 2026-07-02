#!/usr/bin/env python3
"""Generate missing X-Omni images for the DYNEVAL-1K set.

Reads X-Omni-MISSING-PROMPTS.json (405 EvalMuse prompts; 595/1000 exist).
Already have: 129 GenEval + 466 DPG-Bench. Missing: 405 EvalMuse.

Saves:
    <output_dir>/<gid:04d>.png   e.g. 0026.png

Copy to main machine:
    DYNEVAL-1K-IMAGES/X-Omni/

Usage (from DYNEVAL project root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/X-Omni/infer.py --dry-run
    python dyneval_code/generator_scripts/X-Omni/infer.py --limit 5
    python dyneval_code/generator_scripts/X-Omni/infer.py --gpus 0,1,2,3

Remote server:
    python dyneval_code/generator_scripts/X-Omni/infer.py \\
        --batch_json X-Omni-MISSING-PROMPTS.json \\
        --output_dir ./xomni_outputs \\
        --model_name_or_path ./X-Omni \\
        --flux_model_name_or_path ./FLUX \\
        --gpus 0,1,2,3
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

DEFAULT_JSON = "X-Omni-MISSING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES", "X-Omni")


def parse_image_size(image_size):
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) == 1:
        return image_size[0], image_size[0]
    return image_size[0], image_size[1]


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    ).cuda()
    model.init_vision(args.flux_model_name_or_path)
    model.set_generation_mode("image")
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt, args):
    import torch
    from transformers.generation import GenerationConfig

    image_size = parse_image_size(args.image_size)
    token_h = image_size[0] // args.downsample_size
    token_w = image_size[1] // args.downsample_size
    image_prefix = f"<SOM>{token_h} {token_w}<IMAGE>"

    generation_config = GenerationConfig(
        max_new_tokens=token_h * token_w,
        do_sample=True,
        temperature=args.temperature,
        min_p=args.min_p,
        top_p=args.top_p,
        guidance_scale=args.cfg_scale,
        suppress_tokens=tokenizer.convert_tokens_to_ids(model.config.mm_special_tokens),
    )

    tokens = tokenizer(
        [prompt + image_prefix],
        return_tensors="pt",
        padding="longest",
        padding_side="left",
    )
    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()
    negative_ids = tokenizer.encode(
        image_prefix,
        add_special_tokens=False,
        return_tensors="pt",
    ).cuda().expand(1, -1)

    torch.manual_seed(args.seed)
    out_tokens = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        negative_prompt_ids=negative_ids,
    )

    out_tokens = torch.nn.functional.pad(
        out_tokens,
        (0, 1),
        value=tokenizer.convert_tokens_to_ids("<EOM>"),
    )
    torch.manual_seed(args.seed)
    _, images = model.mmdecode(tokenizer, out_tokens[0], skip_special_tokens=False)
    return images[0]


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

    args = argparse.Namespace(**cfg)
    t0 = time.time()
    print(f"[{tag}] loading X-Omni from {args.model_name_or_path} ...")
    model, tokenizer = load_model(args)
    print(f"[{tag}] model ready in {time.time() - t0:.1f}s")

    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        t1 = time.time()
        try:
            generate_one(model, tokenizer, prompt, args).save(dst)
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

    if cfg.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    args = argparse.Namespace(**cfg)
    print(f"Loading X-Omni from {args.model_name_or_path} ...")
    model, tokenizer = load_model(args)

    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        t1 = time.time()
        try:
            generate_one(model, tokenizer, prompt, args).save(dst)
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


def build_cfg(args, gpu=None):
    return {
        "model_name_or_path": args.model_name_or_path,
        "flux_model_name_or_path": args.flux_model_name_or_path,
        "output_dir": args.output_dir,
        "image_size": args.image_size,
        "downsample_size": args.downsample_size,
        "cfg_scale": args.cfg_scale,
        "temperature": args.temperature,
        "min_p": args.min_p,
        "top_p": args.top_p,
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "gpu": gpu,
    }


def parse_gpus(gpu_str):
    if not gpu_str:
        return None
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_json", default=DEFAULT_JSON)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--flux_model_name_or_path", type=str, required=True)
    ap.add_argument("--image-size", type=int, nargs="+", default=[1152],
                        dest="image_size")
    ap.add_argument("--downsample-size", type=int, default=16, dest="downsample_size")
    ap.add_argument("--cfg-scale", type=float, default=1.0, dest="cfg_scale")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--min-p", type=float, default=0.03, dest="min_p")
    ap.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        dest="attn_implementation",
        help="Attention backend for model loading. Defaults to sdpa to avoid requiring flash-attn.",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids, e.g. 0,1,2,3")
    args = ap.parse_args()

    prompts = load_prompts(args.batch_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(f"[X-Omni] json={len(prompts)}  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}")
    print(f"  model={args.model_name_or_path}")
    print(f"  flux={args.flux_model_name_or_path}")
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

    print(f"\nDone X-Omni: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail*.log in output_dir")


if __name__ == "__main__":
    main()
