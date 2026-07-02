#!/usr/bin/env python3
"""Generate GPT-Image-1.5 images for all 1000 DYNEVAL-1K prompts.

Reads GPT-Image-1.5-PROMPTS.json in gid order (0001.png .. 1000.png) and saves to:

    <root>/DYNEVAL-1K-IMAGES/GPT-Image-1.5/<gid:04d>.png

Same layout as every other model in the benchmark.

Setup:
    export OPENAI_API_KEY="sk-..."

Usage (from DYNEVAL project root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/GPT-Image/infer.py --dry-run
    python dyneval_code/generator_scripts/GPT-Image/infer.py --limit 5
    python dyneval_code/generator_scripts/GPT-Image/infer.py

Remote server (copy JSON + script; images go to standard DYNEVAL folder):
    python dyneval_code/generator_scripts/GPT-Image/infer.py \\
        --root . \\
        --batch_json GPT-Image-1.5-PROMPTS.json \\
        --api_key "$OPENAI_API_KEY"
"""
import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import traceback

ROOT = "."
DEFAULT_JSON = os.path.join(
    "DYNEVAL-UPDATED-1K-EXPERIMENT/prompts",
    "DYNEVAL-1K-FIXED-PROMPTS-SHYAM-REMAINING.json",
)
MAP = "DYNEVAL-1K-PROMPTS-MAP.tsv"
OUT_DIR = os.path.join("DYNEVAL-1K-IMAGES", "GPT-Image-1.5")
DEFAULT_MODEL = "gpt-image-1.5"
DEFAULT_SIZE = "1024x1024"


def resolve_paths(root, batch_json, output_dir, map_path):
    root = os.path.abspath(root)
    if batch_json and not os.path.isabs(batch_json):
        batch_json = os.path.join(root, batch_json)
    if not batch_json:
        batch_json = os.path.join(root, "GPT-Image-1.5-PROMPTS.json")
    if map_path and not os.path.isabs(map_path):
        map_path = os.path.join(root, map_path)
    if not map_path:
        map_path = os.path.join(root, "DYNEVAL-1K-PROMPTS-MAP.tsv")
    if output_dir:
        out_dir = output_dir if os.path.isabs(output_dir) else os.path.join(root, output_dir)
    else:
        out_dir = os.path.join(root, "DYNEVAL-1K-IMAGES", "GPT-Image-1.5")
    return root, batch_json, out_dir, map_path


def load_prompts_from_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", data)
    # Keep JSON order, but ensure stable gid ordering (0001 .. 1000)
    prompts = sorted(prompts, key=lambda p: int(p["gid"]))
    return prompts


def load_prompts_from_map(map_path):
    with open(map_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows.sort(key=lambda r: int(r["id"]))
    if len(rows) != 1000:
        print(f"Warning: expected 1000 prompts, got {len(rows)}", file=sys.stderr)
    prompts = []
    for row in rows:
        gid = int(row["id"])
        prompts.append({
            "gid": gid,
            "filename": f"{gid:04d}.png",
            "output_path": f"DYNEVAL-1K-IMAGES/GPT-Image-1.5/{gid:04d}.png",
            "benchmark": row["benchmark"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "orig_line": int(row["orig_line"]),
            "source_file": row["source_file"],
            "source_index": int(row["orig_line"]) - 1,
            "prompt": row["prompt"],
        })
    return prompts


def make_client(api_key=None, base_url=None):
    from openai import OpenAI

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit(
            "Missing API key. Set OPENAI_API_KEY or pass --api_key.\n"
            "Example: export OPENAI_API_KEY='sk-...'"
        )
    kwargs = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def generate_image(client, prompt, model, size, quality=None):
    kwargs = {
        "model": model,
        "prompt": prompt,
        "size": size,
    }
    if quality:
        kwargs["quality"] = quality

    result = client.images.generate(**kwargs)
    item = result.data[0]

    if getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)

    if getattr(item, "url", None):
        import urllib.request

        with urllib.request.urlopen(item.url, timeout=120) as resp:
            return resp.read()

    raise RuntimeError("API response had neither b64_json nor url")


def is_retryable(exc):
    name = exc.__class__.__name__
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError"}:
        return True
    msg = str(exc).lower()
    return any(x in msg for x in ("rate limit", "timeout", "temporarily", "503", "502"))


def summarize_openai_error(exc):
    msg = str(exc)
    fields = []
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else None

    request_id = getattr(exc, "request_id", None)
    if not request_id:
        response = getattr(exc, "response", None)
        if response is not None:
            request_id = response.headers.get("x-request-id")
    if not request_id:
        match = re.search(r"\breq_[A-Za-z0-9]+\b", msg)
        if match:
            request_id = match.group(0)
    if request_id:
        fields.append(f"request_id={request_id}")

    if isinstance(error, dict):
        for key in ("code", "type", "param"):
            value = error.get(key)
            if value is not None:
                fields.append(f"{key}={value}")
        details = error.get("moderation_details")
        if isinstance(details, dict):
            stage = details.get("moderation_stage")
            categories = details.get("categories")
            if stage:
                fields.append(f"moderation_stage={stage}")
            if categories:
                fields.append(f"categories={categories}")

    return " | ".join(fields) or msg


def generate_with_retries(client, prompt, model, size, quality, max_retries, retry_wait):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return generate_image(client, prompt, model, size, quality)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not is_retryable(exc):
                raise
            wait = retry_wait * attempt
            print(f"    retry {attempt}/{max_retries - 1} in {wait}s: {exc}")
            time.sleep(wait)
    raise last_exc


def build_pending(prompts, output_dir, gid_start, gid_end, overwrite):
    pending = []
    for entry in prompts:
        gid = int(entry["gid"])
        if gid < gid_start or gid > gid_end:
            continue
        filename = entry.get("filename") or f"{gid:04d}.png"
        dst = os.path.join(output_dir, filename)
        if not overwrite and os.path.exists(dst):
            continue
        pending.append({
            "gid": gid,
            "filename": filename,
            "benchmark": entry.get("benchmark", ""),
            "category": entry.get("category", ""),
            "subcategory": entry.get("subcategory", ""),
            "prompt": entry["prompt"],
            "dst": dst,
        })
    return pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT,
                    help="DYNEVAL project root (parent of DYNEVAL-1K-IMAGES/)")
    ap.add_argument("--batch_json", default=DEFAULT_JSON,
                    help="JSON prompt list in gid order (default)")
    ap.add_argument("--map", default=None,
                    help="fallback: read DYNEVAL-1K-PROMPTS-MAP.tsv instead of JSON")
    ap.add_argument("--output_dir", default=None,
                    help="override output dir (default: <root>/DYNEVAL-1K-IMAGES/GPT-Image-1.5)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="OpenAI image model, default gpt-image-1.5")
    ap.add_argument("--size", default=DEFAULT_SIZE,
                    help="image size, default 1024x1024")
    ap.add_argument("--quality", default=None,
                    help="optional quality flag if supported by the API")
    ap.add_argument("--api_key", default=None,
                    help="OpenAI API key (else OPENAI_API_KEY env var)")
    ap.add_argument("--base_url", default=None,
                    help="optional custom OpenAI-compatible base URL")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds to wait between successful requests")
    ap.add_argument("--max_retries", type=int, default=5,
                    help="retries per image on rate limit / transient errors")
    ap.add_argument("--retry_wait", type=float, default=10.0,
                    help="base seconds between retries")
    ap.add_argument("--limit", type=int, default=None,
                    help="generate at most N images this run")
    ap.add_argument("--gid_start", type=int, default=None,
                    help="only prompts with gid >= this (default: min gid in JSON)")
    ap.add_argument("--gid_end", type=int, default=None,
                    help="only prompts with gid <= this (default: max gid in JSON)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate even if PNG already exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _, batch_json, output_dir, map_path = resolve_paths(
        args.root, args.batch_json, args.output_dir, args.map or MAP,
    )

    if args.map:
        prompts = load_prompts_from_map(map_path)
        source = map_path
    elif os.path.isfile(batch_json):
        prompts = load_prompts_from_json(batch_json)
        source = batch_json
    elif os.path.isfile(map_path):
        prompts = load_prompts_from_map(map_path)
        source = map_path
        print(f"JSON not found, falling back to MAP: {map_path}", file=sys.stderr)
    else:
        sys.exit(f"Neither JSON nor MAP found.\n  json={batch_json}\n  map={map_path}")

    if not prompts:
        sys.exit(f"No prompts found in {source}")

    gids = [int(p["gid"]) for p in prompts]
    gid_start = args.gid_start if args.gid_start is not None else min(gids)
    gid_end = args.gid_end if args.gid_end is not None else max(gids)

    os.makedirs(output_dir, exist_ok=True)
    pending = build_pending(
        prompts, output_dir, gid_start, gid_end, args.overwrite,
    )

    total_in_range = sum(
        1 for p in prompts
        if gid_start <= int(p["gid"]) <= gid_end
    )
    print(f"[GPT-Image] model={args.model}  size={args.size}")
    print(f"  source={source}")
    print(f"  range={gid_start}-{gid_end}  total={total_in_range}"
          f"  already done={total_in_range - len(pending)}"
          f"  to generate={len(pending)}")
    print(f"  output -> {output_dir}")

    if args.limit:
        pending = pending[: args.limit]
        print(f"  limited to {len(pending)} this run")

    if not pending:
        print("Nothing to do.")
        return

    if args.dry_run:
        for item in pending[:20]:
            print(f"  {item['filename']}  [{item['benchmark']}]  {item['prompt'][:90]}")
        if len(pending) > 20:
            print(f"  ... and {len(pending) - 20} more")
        return

    client = make_client(api_key=args.api_key, base_url=args.base_url)
    manifest = os.path.join(output_dir, "generated_manifest.jsonl")
    fail_log = os.path.join(output_dir, "generate_fail.log")
    ok = bad = 0

    for i, item in enumerate(pending, 1):
        gid = item["gid"]
        filename = item["filename"]
        prompt = item["prompt"]
        dst = item["dst"]
        try:
            img_bytes = generate_with_retries(
                client,
                prompt=prompt,
                model=args.model,
                size=args.size,
                quality=args.quality,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
            with open(dst, "wb") as f:
                f.write(img_bytes)
            ok += 1
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({
                    "gid": gid,
                    "filename": filename,
                    "benchmark": item["benchmark"],
                    "model": args.model,
                    "size": args.size,
                    "prompt": prompt,
                }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(pending)}] {filename} OK  [{item['benchmark']}]")
            if args.sleep > 0 and i < len(pending):
                time.sleep(args.sleep)
        except Exception as exc:
            bad += 1
            summary = summarize_openai_error(exc)
            with open(fail_log, "a", encoding="utf-8") as lf:
                lf.write(f"{gid}\t{filename}\t{prompt}\t{summary}\t{exc}\n")
                lf.write(traceback.format_exc() + "\n")
            print(f"[{i}/{len(pending)}] {filename} FAILED: {summary}")

    print(f"\nDone GPT-Image: generated={ok}  failed={bad}  -> {output_dir}")
    if bad:
        print(f"Failures logged to {fail_log}")


if __name__ == "__main__":
    main()
