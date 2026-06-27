#!/usr/bin/env python3
import os
import json
import torch
import argparse
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from diffusers import AutoPipelineForText2Image


def seed_everything(seed: int):
    """Sets the seed for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from JSON captions using Kandinsky Model"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="kandinsky-community/kandinsky-3",
        help="Model name or path",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        required=True,
        help="JSON file containing captions with 'id' and 'caption' fields",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Directory to save images",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set the master seed
    seed_everything(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"🖼️  Outputs will be saved to: {args.outdir}")

    # Load JSON file
    print(f"Loading JSON file: {args.json_file}")
    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} items from JSON file.")

    # Load Model
    print(f"Loading model: {args.model_name_or_path}...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model_name_or_path,
        variant="fp16",
        torch_dtype=torch.float16
    )
    pipe.enable_model_cpu_offload()
    print("✅ Model loaded successfully.")

    # Initialize counters
    generated_count = 0
    skipped_count = 0

    # Process each item in JSON
    for item in tqdm(data, desc="Processing items"):
        img_id = item.get("id")
        caption = item.get("caption")

        if not img_id or not caption:
            print(f"⚠️  Skipping item: missing 'id' or 'caption' field.")
            skipped_count += 1
            continue

        save_path = os.path.join(args.outdir, f"{img_id}.png")

        # Check if output already exists
        if os.path.exists(save_path):
            print(f"⏩ Skipping {img_id}, output already exists.")
            skipped_count += 1
            continue

        print(f"\nProcessing {img_id} | Caption: \"{caption}\"")

        try:
            # Generate image
            torch.manual_seed(args.seed)
            
            image = pipe(caption).images[0]
            
            # Save image
            image.save(save_path)
            generated_count += 1
            print(f"✅ Image saved to {save_path}")

        except Exception as e:
            print(f"❌ Error processing {img_id}: {str(e)}")
            skipped_count += 1

    # Save Metadata
    meta_info = {
        "model": args.model_name_or_path,
        "json_file": os.path.abspath(args.json_file),
        "total_items": len(data),
        "generated": generated_count,
        "skipped": skipped_count,
        "seed": args.seed,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(args.outdir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    print(f"\n📝 Metadata saved to {meta_path}")

    print("\n--- Summary ---")
    print(f"Total items: {len(data)}")
    print(f"Generated: {generated_count}")
    print(f"Skipped: {skipped_count}")
    print(f"\n✅ Processing complete!")


if __name__ == "__main__":
    main()
