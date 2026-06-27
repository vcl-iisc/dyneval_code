#!/usr/bin/env python3
import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OmniGen2 batch image generation script."
    )

    # Batch processing arguments
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to text file containing prompts (one per line)",
    )
    parser.add_argument(
        "--start_line", type=int, required=True, help="Starting line number (1-based)"
    )
    parser.add_argument(
        "--finish_line", type=int, required=True, help="Ending line number (1-based)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated images",
    )

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to transformer checkpoint.",
    )
    parser.add_argument(
        "--transformer_lora_path",
        type=str,
        default=None,
        help="Path to transformer LoRA checkpoint.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver++"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--num_inference_step", type=int, default=50, help="Number of inference steps."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for generation."
    )
    parser.add_argument("--height", type=int, default=1024, help="Output image height.")
    parser.add_argument("--width", type=int, default=1024, help="Output image width.")
    parser.add_argument(
        "--max_input_image_pixels",
        type=int,
        default=1048576,
        help="Maximum number of pixels for each input image.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for model weights.",
    )
    parser.add_argument(
        "--text_guidance_scale", type=float, default=4.0, help="Text guidance scale."
    )
    parser.add_argument(
        "--image_guidance_scale", type=float, default=2.0, help="Image guidance scale."
    )
    parser.add_argument(
        "--cfg_range_start", type=float, default=0.0, help="Start of the CFG range."
    )
    parser.add_argument(
        "--cfg_range_end", type=float, default=1.0, help="End of the CFG range."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation.",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to input image(s).",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload.",
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload.",
    )
    parser.add_argument(
        "--enable_group_offload", action="store_true", help="Enable group offload."
    )
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
        help="Enable teacache to speed up inference.",
    )
    parser.add_argument(
        "--teacache_rel_l1_thresh",
        type=float,
        default=0.05,
        help="Relative L1 threshold for teacache.",
    )
    parser.add_argument(
        "--enable_taylorseer", action="store_true", help="Enable TaylorSeer Caching."
    )
    return parser.parse_args()


def load_prompts_range(prompts_file, start_line, finish_line):
    """Load prompts from specified line range (1-based)"""
    with open(prompts_file, "r") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    # Convert to 0-based indexing
    start_idx = start_line - 1
    finish_idx = finish_line

    if start_idx < 0:
        start_idx = 0
    if finish_idx > len(all_prompts):
        finish_idx = len(all_prompts)

    prompts = all_prompts[start_idx:finish_idx]
    return prompts, start_idx


def get_existing_images(output_dir, start_idx, num_prompts):
    """Get set of already generated image indices"""
    existing = set()
    if not os.path.exists(output_dir):
        return existing

    for i in range(num_prompts):
        line_no = start_idx + i + 1  # Convert back to 1-based
        img_path = os.path.join(output_dir, f"{line_no}.png")
        if os.path.exists(img_path):
            existing.add(i)

    return existing


def load_pipeline(
    args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype
) -> OmniGen2Pipeline:
    pipeline = OmniGen2Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=weight_dtype,
        trust_remote_code=True,
    )

    if args.transformer_path:
        print(f"Transformer weights loaded from {args.transformer_path}")
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=weight_dtype,
        )
    else:
        pipeline.transformer = OmniGen2Transformer2DModel.from_pretrained(
            args.model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )

    if args.transformer_lora_path:
        print(f"LoRA weights loaded from {args.transformer_lora_path}")
        pipeline.load_lora_weights(args.transformer_lora_path)

    if args.enable_teacache and args.enable_taylorseer:
        print(
            "WARNING: enable_teacache and enable_taylorseer are mutually exclusive. enable_teacache will be ignored."
        )

    if args.enable_taylorseer:
        pipeline.enable_taylorseer = True
    elif args.enable_teacache:
        pipeline.transformer.enable_teacache = True
        pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh

    if args.scheduler == "dpmsolver++":
        try:
            from omnigen2.schedulers.scheduling_dpmsolver_multistep import (
                DPMSolverMultistepScheduler,
            )

            scheduler = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                solver_order=2,
                prediction_type="flow_prediction",
            )
            pipeline.scheduler = scheduler
        except ImportError:
            print(
                "WARNING: DPMSolver scheduler not available, using default euler scheduler"
            )

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(
            pipeline.transformer,
            onload_device=accelerator.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
        apply_group_offloading(
            pipeline.mllm,
            onload_device=accelerator.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
        apply_group_offloading(
            pipeline.vae,
            onload_device=accelerator.device,
            offload_type="block_level",
            num_blocks_per_group=2,
            use_stream=True,
        )
    else:
        pipeline = pipeline.to(accelerator.device)

    return pipeline


def preprocess(input_image_path: List[str] = []) -> List[Image.Image]:
    """Preprocess the input images."""
    # Process input images
    input_images = None

    if input_image_path:
        input_images = []
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [
                Image.open(os.path.join(input_image_path[0], f)).convert("RGB")
                for f in os.listdir(input_image_path[0])
            ]
        else:
            input_images = [
                Image.open(path).convert("RGB") for path in input_image_path
            ]

        input_images = [ImageOps.exif_transpose(img) for img in input_images]

    return input_images


def run_generation(
    args: argparse.Namespace,
    accelerator: Accelerator,
    pipeline: OmniGen2Pipeline,
    instruction: str,
    negative_prompt: str,
    input_images: List[Image.Image],
    seed_offset: int = 0,
):
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(
        args.seed + seed_offset
    )

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_step,
        max_sequence_length=1024,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
    return results


def main():
    """Main function to run the batch image generation process."""
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.dtype if args.dtype != "fp32" else "no"
    )

    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == "fp16":
        weight_dtype = torch.float16
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts for specified range
    prompts, start_idx = load_prompts_range(
        args.prompts_file, args.start_line, args.finish_line
    )

    if not prompts:
        print("❌ No prompts found in specified range")
        return

    # Check for existing images (resume logic)
    existing = get_existing_images(args.output_dir, start_idx, len(prompts))
    remaining_indices = [i for i in range(len(prompts)) if i not in existing]
    remaining_prompts = [prompts[i] for i in remaining_indices]

    print(f"📋 Total prompts in range: {len(prompts)}")
    print(f"✅ Already generated: {len(existing)}")
    print(f"🕗 Remaining to generate: {len(remaining_prompts)}")
    print(f"📐 Image size: {args.width}x{args.height}")

    if not remaining_prompts:
        print("🎉 All images already generated!")
        return

    # Load pipeline and process inputs
    print("🔄 Loading OmniGen2 model...")
    pipeline = load_pipeline(args, accelerator, weight_dtype)
    input_images = preprocess(args.input_image_path)
    print("✅ Model loaded")

    # Generate images
    for i, prompt_idx in enumerate(
        tqdm(remaining_indices, desc="Generating images", unit="image")
    ):
        line_no = start_idx + prompt_idx + 1  # Convert to 1-based line number
        prompt = prompts[prompt_idx]
        output_path = os.path.join(args.output_dir, f"{line_no}.png")

        # Skip if already exists (double-check for resume safety)
        if os.path.exists(output_path):
            continue

        try:
            # Generate image with seed offset for variety
            results = run_generation(
                args,
                accelerator,
                pipeline,
                prompt,
                args.negative_prompt,
                input_images,
                seed_offset=prompt_idx,
            )

            # Save the first (or only) generated image
            image = results.images[0]
            image.save(output_path)

            # If multiple images per prompt, save them with suffixes
            if len(results.images) > 1:
                for j, img in enumerate(results.images):
                    if j == 0:
                        continue  # Already saved above
                    img_path = os.path.join(args.output_dir, f"{line_no}_{j}.png")
                    img.save(img_path)

        except Exception as e:
            print(f"❌ Error generating line {line_no}: {e}")

    print("🎯 Generation completed!")


if __name__ == "__main__":
    main()
