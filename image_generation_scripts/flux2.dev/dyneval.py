#!/usr/bin/env python3
"""Generate one FLUX.2-dev image from a single prompt."""
import argparse
import io
import os
import re
import sys
import time

import requests
import torch
from huggingface_hub import get_token

DEFAULT_OUT = "outputs"
DEFAULT_MODEL = "./FLUX2-DEV-MODEL"
REMOTE_ENCODER_URL = "https://remote-text-encoder-flux-2.huggingface.co/predict"
GENERATOR_DEVICE = "cuda:0"


def configure_torch():
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def parse_max_memory(spec, n_gpus):
    if spec is None:
        return {i: "78GiB" for i in range(n_gpus)}
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) == 1 and n_gpus > 1:
        parts = parts * n_gpus
    if len(parts) != n_gpus:
        sys.exit(f"--max_memory expects {n_gpus} values, got {len(parts)}: {spec!r}")
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
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
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
                f"Saved response to {debug_path}. content-type={content_type}; excerpt={excerpt!r}"
            )

        try:
            prompt_embeds = torch.load(io.BytesIO(response.content), map_location=device, weights_only=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--output", default=None, help="full output image path")
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--filename", default="output.png")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--remote_encoder", action="store_true", help="use HF remote text encoder; needs HF_TOKEN")
    parser.add_argument("--device_map", choices=("balanced", "auto", "single"), default=None)
    parser.add_argument("--max_memory", default=None, help="per-GPU memory cap, comma-separated")
    parser.add_argument("--attention_slicing", action="store_true")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--remote_encoder_timeout", type=int, default=900)
    parser.add_argument("--remote_encoder_retry_interval", type=int, default=10)
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map = resolve_device_map(args.device_map or "balanced", n_gpus)
    max_memory = parse_max_memory(args.max_memory, n_gpus) if device_map != "single" else None
    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"remote_encoder={args.remote_encoder}")
    print(f"device_map={device_map}  gpus={n_gpus}")
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

    pipe_kwargs = {
        "height": args.height,
        "width": args.width,
        "generator": torch.Generator(device=GENERATOR_DEVICE).manual_seed(args.seed),
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
    }
    if args.remote_encoder:
        pipe_kwargs["prompt_embeds"] = remote_text_encoder(
            args.prompt,
            GENERATOR_DEVICE,
            hf_token,
            max_wait=args.remote_encoder_timeout,
            retry_interval=args.remote_encoder_retry_interval,
        )
    else:
        pipe_kwargs["prompt"] = args.prompt

    start = time.time()
    with torch.inference_mode():
        image = pipe(**pipe_kwargs).images[0]
    image.save(output)
    print(f"Saved {output} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
