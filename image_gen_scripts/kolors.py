#!/usr/bin/env python3
import argparse
import os

import numpy as np
import torch
from diffusers import KolorsPipeline
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
    parser = argparse.ArgumentParser(description="Generate images using Kolors model")
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

    args = parser.parse_args()

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
    print("📐 Image size: 1024x1024")

    if not remaining_prompts:
        print("🎉 All images already generated!")
        return

    # Load model
    print("🔄 Loading Kolors model...")
    try:
        pipe = KolorsPipeline.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")

        # Enable memory optimizations
        pipe.enable_attention_slicing()
        # pipe.enable_model_cpu_offload()

        # Ensure proper initialization
        pipe.scheduler.set_timesteps(50)
        print("✅ Model loaded and configured")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

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
            # Generate image without autocast to avoid precision issues
            # Use different seed for each image to ensure variety
            seed = 42 + prompt_idx
            image = pipe(
                prompt,
                height=1024,
                width=1024,
                guidance_scale=5.0,
                num_inference_steps=50,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]

            # Quick check if generated image is mostly black
            img_array = np.array(image)
            mean_brightness = np.mean(img_array)
            if mean_brightness < 10:
                print(
                    f"⚠️  Warning: Generated image {line_no} appears very dark (brightness: {mean_brightness:.2f})"
                )

            image.save(output_path)

        except Exception as e:
            print(f"❌ Error generating line {line_no}: {e}")

    print("🎯 Generation completed!")


if __name__ == "__main__":
    main()
