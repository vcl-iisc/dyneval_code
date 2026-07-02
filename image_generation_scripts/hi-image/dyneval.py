#!/usr/bin/env python3
"""Generate one HiDream image from a single prompt."""
import argparse
import os
import time

DEFAULT_OUT = "outputs"
DEFAULT_MODEL = "./HiDream-I1-Full-model"
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

MODEL_CONFIGS = {
    "dev": {
        "guidance_scale": 0.0,
        "num_inference_steps": 28,
        "shift": 6.0,
        "scheduler": "FlashFlowMatchEulerDiscreteScheduler",
    },
    "full": {
        "guidance_scale": 5.0,
        "num_inference_steps": 50,
        "shift": 3.0,
        "scheduler": "FlowUniPCMultistepScheduler",
    },
    "fast": {
        "guidance_scale": 0.0,
        "num_inference_steps": 16,
        "shift": 3.0,
        "scheduler": "FlashFlowMatchEulerDiscreteScheduler",
    },
}


def get_scheduler_cls(name):
    from hi_diffusers.schedulers.flash_flow_match import FlashFlowMatchEulerDiscreteScheduler
    from hi_diffusers.schedulers.fm_solvers_unipc import FlowUniPCMultistepScheduler

    return {
        "FlowUniPCMultistepScheduler": FlowUniPCMultistepScheduler,
        "FlashFlowMatchEulerDiscreteScheduler": FlashFlowMatchEulerDiscreteScheduler,
    }[name]


def load_models(model_type, model_path, llama_path=None):
    import torch
    from hi_diffusers import HiDreamImagePipeline, HiDreamImageTransformer2DModel
    from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

    config = MODEL_CONFIGS[model_type]
    llama_model_name_or_path = llama_path or LLAMA_MODEL_NAME
    scheduler_cls = get_scheduler_cls(config["scheduler"])
    scheduler = scheduler_cls(num_train_timesteps=1000, shift=config["shift"], use_dynamic_shifting=False)

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_model_name_or_path, use_fast=False)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        llama_model_name_or_path,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    transformer = HiDreamImageTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe = HiDreamImagePipeline.from_pretrained(
        model_path,
        scheduler=scheduler,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        torch_dtype=torch.bfloat16,
    ).to("cuda", torch.bfloat16)
    pipe.transformer = transformer
    return pipe, config


def generate_image(pipe, config, prompt, height, width, seed):
    import torch

    if seed < 0:
        seed = torch.randint(0, 1_000_000, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)
    images = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=config["guidance_scale"],
        num_inference_steps=config["num_inference_steps"],
        num_images_per_prompt=1,
        generator=generator,
    ).images
    return images[0], seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--output", default=None, help="full output image path")
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--filename", default="output.png")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--model_type", choices=tuple(MODEL_CONFIGS), default="full")
    parser.add_argument("--llama_path", default=None, help="local Llama-3.1-8B-Instruct path; defaults to HF id")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0, help="use -1 for a random seed")
    args = parser.parse_args()

    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"Loading HiDream model_type={args.model_type} ...")
    print(f"model_path={args.model_path}")
    pipe, model_config = load_models(args.model_type, args.model_path, llama_path=args.llama_path)
    start = time.time()
    image, seed = generate_image(pipe, model_config, args.prompt, args.height, args.width, args.seed)
    image.save(output)
    print(f"Saved {output} in {time.time() - start:.1f}s (seed={seed})")


if __name__ == "__main__":
    main()
