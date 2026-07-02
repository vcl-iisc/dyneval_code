#!/usr/bin/env python3
"""Generate NanoBanana (Gemini image) images for all 1000 DYNEVAL-1K prompts.

Uses the Google genai API (same as nanobanana-test.py):

    from google import genai
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[prompt],
    )

Reads NanoBanana-PROMPTS.json in gid order and saves:

    <root>/DYNEVAL-1K-IMAGES/NanoBanana/<gid:04d>.png

Setup:
    export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY

Usage (from DYNEVAL project root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/nanobanana/infer.py --dry-run
    python dyneval_code/generator_scripts/nanobanana/infer.py --limit 5
    python dyneval_code/generator_scripts/nanobanana/infer.py

Remote server:
    python dyneval_code/generator_scripts/nanobanana/infer.py \\
        --root . \\
        --batch_json NanoBanana-PROMPTS.json
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

ROOT = "."
DEFAULT_JSON = os.path.join(
    "DYNEVAL-UPDATED-1K-EXPERIMENT/prompts",
    "DYNEVAL-1K-FIXED-PROMPTS-SHYAM-REMAINING.json",
)
MAP = "DYNEVAL-1K-PROMPTS-MAP.tsv"
OUT_DIR = os.path.join("DYNEVAL-1K-IMAGES", "NanoBanana")
DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


def resolve_paths(root, batch_json, output_dir, map_path):
    root = os.path.abspath(root)
    if batch_json and not os.path.isabs(batch_json):
        batch_json = os.path.join(root, batch_json)
    if not batch_json:
        batch_json = os.path.join(root, "NanoBanana-PROMPTS.json")
    if map_path and not os.path.isabs(map_path):
        map_path = os.path.join(root, map_path)
    if not map_path:
        map_path = os.path.join(root, "DYNEVAL-1K-PROMPTS-MAP.tsv")
    if output_dir:
        out_dir = output_dir if os.path.isabs(output_dir) else os.path.join(root, output_dir)
    else:
        out_dir = os.path.join(root, "DYNEVAL-1K-IMAGES", "NanoBanana")
    return root, batch_json, out_dir, map_path


def load_prompts_from_json(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", data)
    return sorted(prompts, key=lambda p: int(p["gid"]))


def load_prompts_from_map(map_path):
    with open(map_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    rows.sort(key=lambda r: int(r["id"]))
    prompts = []
    for row in rows:
        gid = int(row["id"])
        prompts.append({
            "gid": gid,
            "filename": f"{gid:04d}.png",
            "output_path": f"DYNEVAL-1K-IMAGES/NanoBanana/{gid:04d}.png",
            "benchmark": row["benchmark"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "orig_line": int(row["orig_line"]),
            "source_file": row["source_file"],
            "source_index": int(row["orig_line"]) - 1,
            "prompt": row["prompt"],
        })
    return prompts


def make_client(api_key=None):
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit(
            "Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY, or pass --api_key.\n"
            "Example: export GEMINI_API_KEY='...'"
        )
    return genai.Client(api_key=key)


def extract_image(response):
    debug = []
    text_parts = []

    prompt_feedback = getattr(response, "prompt_feedback", None)
    if prompt_feedback is not None:
        debug.append(f"prompt_feedback={prompt_feedback}")

    parts = []
    candidates = getattr(response, "candidates", None) or []
    for i, cand in enumerate(candidates):
        debug.append(f"candidate[{i}].finish_reason={getattr(cand, 'finish_reason', None)}")
        safety_ratings = getattr(cand, "safety_ratings", None)
        if safety_ratings is not None:
            debug.append(f"candidate[{i}].safety_ratings={safety_ratings}")
        content = getattr(cand, "content", None)
        if content and getattr(content, "parts", None):
            parts.extend(content.parts)

    if not parts:
        direct_parts = getattr(response, "parts", None)
        if direct_parts:
            parts.extend(direct_parts)

    if not parts:
        detail = "; ".join(debug)
        raise RuntimeError("No response parts returned" + (f"; {detail}" if detail else ""))

    for part in parts:
        if getattr(part, "inline_data", None) is not None:
            image = part.as_image()
            if image is None:
                raise RuntimeError("inline_data present but as_image() returned None")
            return image
        text = getattr(part, "text", None)
        if text:
            text_parts.append(text[:500].replace("\n", "\\n"))

    details = []
    if text_parts:
        details.append(f"text_parts={text_parts}")
    details.extend(debug)
    raise RuntimeError(
        "No image inline_data in response"
        + (f"; {'; '.join(details)}" if details else "")
    )


def generate_image(client, prompt, model):
    try:
        from google.genai import types

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )
    except (ImportError, AttributeError, TypeError):
        response = client.models.generate_content(
            model=model,
            contents=[prompt],
        )
    return extract_image(response)


def is_retryable(exc):
    name = exc.__class__.__name__
    if name in {"ResourceExhausted", "ServiceUnavailable", "DeadlineExceeded", "InternalServerError"}:
        return True
    msg = str(exc).lower()
    return any(x in msg for x in (
        "rate", "quota", "timeout", "temporarily", "503", "502", "429", " overloaded",
    ))


def generate_with_retries(client, prompt, model, max_retries, retry_wait):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return generate_image(client, prompt, model)
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
            "prompt": entry["prompt"],
            "dst": dst,
        })
    return pending


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT,
                    help="DYNEVAL project root (parent of DYNEVAL-1K-IMAGES/)")
    ap.add_argument("--batch_json", default=DEFAULT_JSON,
                    help="JSON prompt list in gid order")
    ap.add_argument("--map", default=None,
                    help="fallback: read DYNEVAL-1K-PROMPTS-MAP.tsv instead of JSON")
    ap.add_argument("--output_dir", default=None,
                    help="override output dir (default: <root>/DYNEVAL-1K-IMAGES/NanoBanana)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Gemini image model id")
    ap.add_argument("--api_key", default=None,
                    help="Gemini API key (else GEMINI_API_KEY / GOOGLE_API_KEY)")
    ap.add_argument("--sleep", type=float, default=0.5,
                    help="seconds to wait between successful requests")
    ap.add_argument("--max_retries", type=int, default=5,
                    help="retries per image on rate limit / transient errors")
    ap.add_argument("--retry_wait", type=float, default=10.0,
                    help="base seconds between retries")
    ap.add_argument("--limit", type=int, default=None,
                    help="generate at most N images this run")
    ap.add_argument("--gid_start", type=int, default=None,
                    help="min gid to generate (default: min gid in JSON)")
    ap.add_argument("--gid_end", type=int, default=None,
                    help="max gid to generate (default: max gid in JSON)")
    ap.add_argument("--overwrite", action="store_true")
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
    print(f"[NanoBanana] model={args.model}")
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

    client = make_client(api_key=args.api_key)
    manifest = os.path.join(output_dir, "generated_manifest.jsonl")
    fail_log = os.path.join(output_dir, "generate_fail.log")
    ok = bad = 0

    for i, item in enumerate(pending, 1):
        gid = item["gid"]
        filename = item["filename"]
        prompt = item["prompt"]
        dst = item["dst"]
        try:
            image = generate_with_retries(
                client,
                prompt=prompt,
                model=args.model,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
            image.save(dst)
            ok += 1
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({
                    "gid": gid,
                    "filename": filename,
                    "benchmark": item["benchmark"],
                    "model": args.model,
                    "prompt": prompt,
                }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(pending)}] {filename} OK  [{item['benchmark']}]")
            if args.sleep > 0 and i < len(pending):
                time.sleep(args.sleep)
        except Exception as exc:
            bad += 1
            with open(fail_log, "a", encoding="utf-8") as lf:
                lf.write(f"{gid}\t{filename}\t{prompt}\t{exc}\n")
                lf.write(traceback.format_exc() + "\n")
            print(f"[{i}/{len(pending)}] {filename} FAILED: {exc}")

    print(f"\nDone NanoBanana: generated={ok}  failed={bad}  -> {output_dir}")
    if bad:
        print(f"Failures logged to {fail_log}")


if __name__ == "__main__":
    main()
