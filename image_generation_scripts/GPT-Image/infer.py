#!/usr/bin/env python3
"""Generate one GPT-Image image from a single prompt.

Setup:
    export OPENAI_API_KEY="sk-..."

Usage:
    python infer.py "a cinematic photo of a red chair beside a window"
"""
import argparse
import base64
import os
import re
import sys
import time

DEFAULT_OUT = "outputs"
DEFAULT_MODEL = "gpt-image-1.5"
DEFAULT_SIZE = "1024x1024"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", help="text prompt to generate")
    ap.add_argument("--output", default=None, help="full output image path")
    ap.add_argument("--output_dir", default=DEFAULT_OUT, help="directory used with --filename")
    ap.add_argument("--filename", default="output.png")
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
    ap.add_argument("--max_retries", type=int, default=5,
                    help="retries on rate limit / transient errors")
    ap.add_argument("--retry_wait", type=float, default=10.0,
                    help="base seconds between retries")
    args = ap.parse_args()

    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"[GPT-Image] model={args.model}  size={args.size}")
    print(f"  output -> {output}")

    client = make_client(api_key=args.api_key, base_url=args.base_url)
    img_bytes = generate_with_retries(
        client,
        prompt=args.prompt,
        model=args.model,
        size=args.size,
        quality=args.quality,
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
    )
    with open(output, "wb") as f:
        f.write(img_bytes)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
