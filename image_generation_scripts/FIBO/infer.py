#!/usr/bin/env python3
"""Generate one FIBO image from a single prompt."""
import argparse
import json
import os
import time

import torch
from diffusers import BriaFiboPipeline
from diffusers.modular_pipelines import ModularPipelineBlocks
from diffusers.modular_pipelines import modular_pipeline as modular_pipeline_module

DEFAULT_OUT = "outputs"
VLM_MODEL = "briaai/FIBO-VLM-prompt-to-JSON"
IMAGE_MODEL = "briaai/FIBO"


def get_default_negative_prompt(existing_json):
    style_medium = existing_json.get("style_medium", "").lower()
    if style_medium in ["photograph", "photography", "photo"]:
        return "{'style_medium':'digital illustration','artistic_style':'non-realistic'}"
    return ""


def patch_modular_diffusers_requirements():
    original_validate = modular_pipeline_module._validate_requirements

    def validate_requirements_compat(reqs):
        if isinstance(reqs, list):
            reqs = {
                name: version if str(version).startswith(("=", "<", ">", "~", "!")) else f"=={version}"
                for name, version in reqs
            }
        return original_validate(reqs)

    modular_pipeline_module._validate_requirements = validate_requirements_compat


def load_pipelines(cpu_offload=False):
    patch_modular_diffusers_requirements()
    torch.set_grad_enabled(False)

    vlm_pipe = ModularPipelineBlocks.from_pretrained(VLM_MODEL, trust_remote_code=True)
    vlm_pipe = vlm_pipe.init_pipeline()

    pipe = BriaFiboPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.bfloat16)
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return vlm_pipe, pipe


def generate_one(vlm_pipe, pipe, prompt, num_steps, guidance_scale):
    output = vlm_pipe(prompt=prompt)
    json_prompt = output.values["json_prompt"]
    parsed_json_prompt = json.loads(json_prompt)
    negative_prompt = get_default_negative_prompt(parsed_json_prompt)
    result = pipe(
        prompt=json_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )
    return result.images[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--output", default=None, help="full output image path")
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--filename", default="output.png")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--cpu_offload", action="store_true")
    args = parser.parse_args()

    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print("Loading FIBO pipelines ...")
    vlm_pipe, pipe = load_pipelines(cpu_offload=args.cpu_offload)
    start = time.time()
    image = generate_one(vlm_pipe, pipe, args.prompt, args.num_inference_steps, args.guidance_scale)
    image.save(output)
    print(f"Saved {output} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
