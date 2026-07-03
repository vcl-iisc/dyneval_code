#!/usr/bin/env python3
"""Generate one HunyuanDiT image from a single prompt."""
import argparse
import os

import torch
from diffusers import HunyuanDiTPipeline

DEFAULT_OUT = "outputs"
DEFAULT_MODEL = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--output", default=None, help="full output image path")
    parser.add_argument("--output_dir", default=DEFAULT_OUT, help="directory used with --filename")
    parser.add_argument("--filename", default="output.png")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"Loading HunyuanDiT from {args.model_name_or_path} ...")
    pipe = HunyuanDiTPipeline.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
    ).to(args.device)

    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    result.images[0].save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
