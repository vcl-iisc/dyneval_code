#!/usr/bin/env python3
"""Generate one DeepFloyd IF-XL image from a single prompt."""
import argparse
import os
import time

DEFAULT_OUT = "outputs"
MODEL_ID = "DeepFloyd/IF-I-XL-v1.0"


def load_pipeline(model_id, cpu_offload=False, torch_dtype="bfloat16"):
    import torch
    from diffusers import DiffusionPipeline

    dtype = getattr(torch, torch_dtype)
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def pipe_device(pipe):
    if hasattr(pipe, "device"):
        return pipe.device
    for name in ("transformer", "unet", "text_encoder", "vae"):
        module = getattr(pipe, name, None)
        if module is not None:
            return next(module.parameters()).device
    import torch
    return torch.device("cuda")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--output", default=None, help="full output image path")
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--filename", default="output.png")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    import torch

    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"Loading {args.model_id} ...")
    pipe = load_pipeline(args.model_id, cpu_offload=args.cpu_offload, torch_dtype=args.torch_dtype)
    generator = torch.Generator(device=pipe_device(pipe)).manual_seed(args.seed)
    start = time.time()
    with torch.inference_mode():
        image = pipe(args.prompt, generator=generator).images[0]
    image.save(output)
    print(f"Saved {output} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
