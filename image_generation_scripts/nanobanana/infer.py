#!/usr/bin/env python3
"""Generate one NanoBanana (Gemini image) image from a single prompt.

Uses the Google genai API (same as nanobanana-test.py):

    from google import genai
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[prompt],
    )

Setup:
    export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY

Usage:
    python infer.py "a cinematic photo of a red chair beside a window"
"""
import argparse
import os
import sys
import time

DEFAULT_OUT = "outputs"
DEFAULT_MODEL = "gemini-3.1-flash-image-preview"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt", help="text prompt to generate")
    ap.add_argument("--output", default=None, help="full output image path")
    ap.add_argument("--output_dir", default=DEFAULT_OUT, help="directory used with --filename")
    ap.add_argument("--filename", default="output.png")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Gemini image model id")
    ap.add_argument("--api_key", default=None,
                    help="Gemini API key (else GEMINI_API_KEY / GOOGLE_API_KEY)")
    ap.add_argument("--max_retries", type=int, default=5,
                    help="retries on rate limit / transient errors")
    ap.add_argument("--retry_wait", type=float, default=10.0,
                    help="base seconds between retries")
    args = ap.parse_args()

    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"[NanoBanana] model={args.model}")
    print(f"  output -> {output}")

    client = make_client(api_key=args.api_key)
    image = generate_with_retries(
        client,
        prompt=args.prompt,
        model=args.model,
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
    )
    image.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
