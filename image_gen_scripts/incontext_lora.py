#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline
from tqdm import tqdm


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using In-Context LoRA (FLUX.1-dev + LoRA)"
    )
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
    parser.add_argument(
        "--model_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Base FLUX model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--lora_repo_id",
        type=str,
        default="ali-vilab/In-Context-LoRA",
        help="LoRA repository ID or path",
    )

    args = parser.parse_args()

    # Optimize CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    print(f"📐 Image size: 1024x1024")

    if not remaining_prompts:
        print("🎉 All images already generated!")
        return

    # Load FLUX pipeline
    print("🔄 Loading FLUX.1-dev model...")
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    # Load In-Context LoRA
    print(f"🔄 Loading In-Context LoRA from {args.lora_repo_id}...")
    pipe.load_lora_weights(args.lora_repo_id)

    # Set adapters if needed
    try:
        pipe.set_adapters(["default"])
    except Exception:
        pass

    # Optional: fuse LoRA for speed
    try:
        pipe.fuse_lora(lora_scale=1.0)
    except Exception as e:
        print(f"⚠️ LoRA fuse skipped: {e}")

    # Memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    print("✅ Model and LoRA loaded")

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
            image = pipe(
                prompt=prompt,
                height=1024,
                width=1024,
                num_inference_steps=50,
            ).images[0]

            image.save(output_path)

        except Exception as e:
            print(f"❌ Error generating line {line_no}: {e}")

    print("🎯 Generation completed!")


if __name__ == "__main__":
    main()
