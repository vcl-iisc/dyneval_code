
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KOLORS_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT = "outputs"
DEFAULT_MODEL = "Kwai-Kolors/Kolors"


def load_pipe(model_name_or_path):
    sys.path.insert(0, KOLORS_ROOT)
    os.chdir(KOLORS_ROOT)

    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from kolors.models.modeling_chatglm import ChatGLMModel
    from kolors.models.tokenization_chatglm import ChatGLMTokenizer
    from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import (
        StableDiffusionXLPipeline,
    )

    text_encoder = ChatGLMModel.from_pretrained(
        model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    ).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(
        model_name_or_path,
        subfolder="text_encoder",
    )
    vae = AutoencoderKL.from_pretrained(
        model_name_or_path,
        subfolder="vae",
        revision=None,
    ).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_name_or_path,
        subfolder="scheduler",
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_name_or_path,
        subfolder="unet",
        revision=None,
    ).half()
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False,
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe


def generate(pipe, prompt, seed, height, width, num_inference_steps, guidance_scale):
    import torch

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        generator=torch.Generator(pipe.device).manual_seed(seed),
    ).images[0]
    return image


def main():
    ap = argparse.ArgumentParser(description="Generate one Kolors image from a single prompt.")
    ap.add_argument("prompt", help="text prompt to generate")
    ap.add_argument("--output", default=None, help="full output image path")
    ap.add_argument("--output_dir", default=DEFAULT_OUT, help="directory used with --filename")
    ap.add_argument("--filename", default="output.png")
    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL,
                    help="Kolors Hugging Face repo id or local weights directory")
    ap.add_argument("--seed", type=int, default=66,
                    help="same default as Kolors scripts/sample.py")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=5.0)
    args = ap.parse_args()

    output = args.output or os.path.join(args.output_dir, args.filename)
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print("Loading Kolors pipeline (once) ...")
    pipe = load_pipe(args.model_name_or_path)

    image = generate(
        pipe,
        args.prompt,
        args.seed,
        args.height,
        args.width,
        args.num_inference_steps,
        args.guidance_scale,
    )
    image.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
