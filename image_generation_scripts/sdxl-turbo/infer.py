#!/usr/bin/env python3
"""Generate one SDXL Turbo image from a single prompt."""
import argparse
import os
import time

DEFAULT_OUT = "outputs"
DEFAULT_MODEL = 'stabilityai/sdxl-turbo'


def configure_torch():
    import torch

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def resolve_output(output, output_dir, filename):
    if output:
        return output
    return os.path.join(output_dir, filename)


def load_pipeline(model_name_or_path, dtype, device, cpu_offload):
    import torch
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--output", default=None, help="full output image path")
    parser.add_argument("--output_dir", default=DEFAULT_OUT, help="directory used with --filename")
    parser.add_argument("--filename", default="output.png", help="output filename when --output is not set")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu_offload", action="store_true", help="reduce VRAM use, usually slower")
    args = parser.parse_args()

    import torch

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    output = resolve_output(args.output, args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"Loading {args.model_name_or_path} on {args.device} ...")
    pipe = load_pipeline(args.model_name_or_path, dtype, args.device, args.cpu_offload)

    generator_device = args.device if args.device != "cpu" else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    start = time.time()
    with torch.inference_mode():
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
        ).images[0]
    image.save(output)
    print(f"Saved {output} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
