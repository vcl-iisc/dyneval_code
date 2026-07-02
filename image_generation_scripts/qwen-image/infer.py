#!/usr/bin/env python3
"""Generate one Qwen-Image output from a single prompt."""
import argparse
import os
import time

DEFAULT_OUT = "outputs"
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


def magic_lang(prompt, lang):
    if lang != "auto":
        return lang
    return "zh" if any("一" <= c <= "鿿" for c in prompt) else "en"


def build_prompt(prompt, lang="auto", use_magic=True):
    if not use_magic:
        return prompt
    return prompt + POSITIVE_MAGIC[magic_lang(prompt, lang)]


def load_pipeline(model_name_or_path, dtype, device, cpu_offload):
    from diffusers import DiffusionPipeline

    configure_torch()
    pipe = DiffusionPipeline.from_pretrained(model_name_or_path, torch_dtype=dtype)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    return pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--output", default=None, help="full output image path")
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--filename", default="output.png")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None, help="cuda or cpu; defaults to cuda when available")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--true_cfg_scale", type=float, default=4.0)
    parser.add_argument("--negative_prompt", default=" ", help="negative prompt; use a single space if unused")
    parser.add_argument("--aspect_ratio", choices=tuple(ASPECT_RATIOS), default="1:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--magic_lang", choices=("auto", "en", "zh"), default="auto")
    parser.add_argument("--no_magic", action="store_true", help="do not append the Qwen positive suffix")
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    import torch

    default_dtype, default_device = resolve_device()
    if args.device is None:
        args.device = default_device
    if args.dtype is None:
        args.dtype = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[default_dtype]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    width, height = ASPECT_RATIOS[args.aspect_ratio]
    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"Loading {args.model_name_or_path} on {args.device} ...")
    pipe = load_pipeline(args.model_name_or_path, dtype=dtype, device=args.device, cpu_offload=args.cpu_offload)
    prompt = build_prompt(args.prompt, args.magic_lang, not args.no_magic)
    generator = torch.Generator(device=args.device if args.device != "cpu" else "cpu").manual_seed(args.seed)
    start = time.time()
    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=args.num_inference_steps,
            true_cfg_scale=args.true_cfg_scale,
            generator=generator,
        ).images[0]
    image.save(output)
    print(f"Saved {output} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
