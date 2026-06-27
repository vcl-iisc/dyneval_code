#!/usr/bin/env python3
import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diffusers import Kandinsky3Pipeline
from PIL import Image
from tqdm import tqdm


def seed_everything(seed: int):
    """Sets the seed for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sanitize_filename(text, max_length=100):
    """Convert text to a valid filename."""
    # Remove or replace invalid characters
    text = text.replace(" ", "_")

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]

    # Remove leading/trailing whitespace and dots
    text = text.strip().strip(".")

    return text


def load_prompts_from_folder(folder_path):
    """Load all .txt files from folder and return dict mapping filename to list of prompts.
    Excludes files with 'train' or 'val' in their name, but includes files starting with 'complex'."""
    prompts_dict = {}
    txt_files = sorted(Path(folder_path).glob("*.txt"))

    if not txt_files:
        raise ValueError(f"No .txt files found in {folder_path}")

    for txt_file in txt_files:
        filename_lower = txt_file.stem.lower()

        # Skip files with 'train' or 'val' in the name (case-insensitive)
        # BUT allow files that start with 'complex'
        if filename_lower.startswith("complex"):
            # Include this file even if it has 'train' or 'val'
            pass
        elif "train" in filename_lower or "val" in filename_lower:
            print(f"⏩ Skipping {txt_file.name} (contains 'train' or 'val')")
            continue

        with open(txt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        if prompts:
            prompts_dict[txt_file.stem] = prompts
            print(f"📄 Loaded {len(prompts)} prompts from {txt_file.name}")

    if not prompts_dict:
        raise ValueError(
            f"No valid .txt files found in {folder_path} (after filtering)"
        )

    return prompts_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from text files using Kandinsky Model"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="kandinsky-community/kandinsky-3",
        help="Model name or path",
    )
    parser.add_argument(
        "--prompts_folder",
        type=str,
        required=True,
        help="Folder containing .txt files with prompts (one per line)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Directory to save images",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # Set the master seed
    seed_everything(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    print(f"🖼️  Outputs will be saved to: {args.outdir}")

    # Load all prompts from folder
    print(f"\nLoading prompts from folder: {args.prompts_folder}")
    prompts_dict = load_prompts_from_folder(args.prompts_folder)
    print(
        f"✅ Loaded {len(prompts_dict)} text files with {sum(len(p) for p in prompts_dict.values())} total prompts.\n"
    )

    # Load Model
    print(f"Loading model: {args.model_name_or_path}...")
    pipe = Kandinsky3Pipeline.from_pretrained(
        args.model_name_or_path, variant="fp16", torch_dtype=torch.float16
    )
    pipe.to("cuda")
    print("✅ Model loaded successfully.\n")

    # Initialize counters
    total_generated = 0
    total_skipped = 0

    # Process each text file
    for txt_filename, prompts in prompts_dict.items():
        print(f"\n{'=' * 60}")
        print(f"Processing: {txt_filename}.txt ({len(prompts)} prompts)")
        print(f"{'=' * 60}")

        # Create subfolder for this text file
        subfolder = os.path.join(args.outdir, txt_filename)
        os.makedirs(subfolder, exist_ok=True)

        file_generated = 0
        file_skipped = 0

        # Process each prompt in the file
        for idx, prompt in enumerate(tqdm(prompts, desc=f"Processing {txt_filename}")):
            # Create filename: sanitized_prompt_index.png
            sanitized_prompt = sanitize_filename(prompt)
            filename = f"{sanitized_prompt}_{idx:06d}.png"
            save_path = os.path.join(subfolder, filename)

            # Check if output already exists
            if os.path.exists(save_path):
                file_skipped += 1
                continue

            try:
                # Set seed for this specific prompt
                torch.manual_seed(args.seed + idx)

                # Generate image
                image = pipe(prompt).images[0]

                # Save image
                image.save(save_path)
                file_generated += 1

            except Exception as e:
                print(f"❌ Error processing prompt {idx}: {str(e)}")
                file_skipped += 1

        print(
            f"\n✅ {txt_filename}: Generated {file_generated}, Skipped {file_skipped}"
        )
        total_generated += file_generated
        total_skipped += file_skipped

        # Save metadata for this text file
        meta_info = {
            "txt_file": f"{txt_filename}.txt",
            "total_prompts": len(prompts),
            "generated": file_generated,
            "skipped": file_skipped,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_path = os.path.join(subfolder, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta_info, f, indent=4)

    # Save overall metadata
    overall_meta = {
        "model": args.model_name_or_path,
        "prompts_folder": os.path.abspath(args.prompts_folder),
        "total_txt_files": len(prompts_dict),
        "total_prompts": sum(len(p) for p in prompts_dict.values()),
        "total_generated": total_generated,
        "total_skipped": total_skipped,
        "seed": args.seed,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    overall_meta_path = os.path.join(args.outdir, "overall_meta.json")
    with open(overall_meta_path, "w") as f:
        json.dump(overall_meta, f, indent=4)
    print(f"\n📝 Overall metadata saved to {overall_meta_path}")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print("=" * 60)
    print(f"Total text files: {len(prompts_dict)}")
    print(f"Total prompts: {sum(len(p) for p in prompts_dict.values())}")
    print(f"Total generated: {total_generated}")
    print(f"Total skipped: {total_skipped}")
    print(f"\n✅ Processing complete!")


if __name__ == "__main__":
    main()
