#!/usr/bin/env python3
import os
import json
import torch
import argparse
import random
import numpy as np
import multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from diffusers import HunyuanDiTPipeline


def seed_everything(seed: int):
    """Sets the seed for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gpu_worker(gpu_id, task_queue, args):
    """Worker function for each GPU process"""
    print(f"🚀 GPU {gpu_id}: Loading HunyuanDiT model...")

    device = torch.device(f"cuda:{gpu_id}")

    # Load HunyuanDiT pipeline on specific GPU
    pipe = HunyuanDiTPipeline.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(device)

    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    print(f"✅ GPU {gpu_id}: Model loaded successfully")

    # Process tasks from queue
    generated_count = 0
    skipped_count = 0

    while not task_queue.empty():
        try:
            item = task_queue.get_nowait()
        except:
            break

        img_id = item.get("id")
        caption = item.get("caption")

        if not img_id or not caption:
            print(f"⏩ GPU {gpu_id}: Skipping invalid item {img_id}")
            skipped_count += 1
            continue

        save_path = os.path.join(args.outdir, f"{img_id}.png")

        try:
            print(f"🖼️ GPU {gpu_id}: Generating for {img_id}")

            # Set unique seed for each generation (deterministic based on id)
            gen_seed = args.seed + hash(str(img_id)) % 1000000
            torch.manual_seed(gen_seed)
            torch.cuda.manual_seed_all(gen_seed)

            # Generate image
            with torch.no_grad():
                result = pipe(prompt=caption)
                image = result.images[0]

            # Save image
            image.save(save_path)
            generated_count += 1
            print(f"✅ GPU {gpu_id}: Saved {save_path}")

        except Exception as e:
            print(f"❌ GPU {gpu_id}: Error processing {img_id}: {str(e)}")
            skipped_count += 1

    print(
        f"🏁 GPU {gpu_id}: Finished. Generated: {generated_count}, Skipped: {skipped_count}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from JSON captions using HunyuanDiT (Multi-GPU)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to HunyuanDiT model",
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
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    # === DETECT GPUS ===
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        print("⚠️ Only one GPU detected — using single GPU.")
    else:
        print(f"✅ Using {num_gpus} GPUs for generation in parallel")

    # Set seed
    seed_everything(args.seed)

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    print(f"🖼️ Outputs will be saved to: {args.outdir}")

    # Load JSON file
    print(f"📄 Loading JSON file: {args.json_file}")
    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ Loaded {len(data)} items from JSON file.")

    # Filter out existing images before distributing to GPUs
    print(f"🔍 Checking for existing images...")
    filtered_data = []
    existing_count = 0

    for item in data:
        img_id = item.get("id")
        if img_id:
            save_path = os.path.join(args.outdir, f"{img_id}.png")
            if os.path.exists(save_path):
                existing_count += 1
            else:
                filtered_data.append(item)
        else:
            filtered_data.append(item)  # Keep items without valid id for error handling

    print(
        f"✅ Found {existing_count} existing images, {len(filtered_data)} items remaining to process"
    )

    # Create task queue with filtered data
    task_queue = mp.Queue()
    for item in filtered_data:
        task_queue.put(item)

    # Start worker processes
    procs = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=gpu_worker, args=(gpu_id, task_queue, args))
        p.start()
        procs.append(p)

    # Wait for all processes to complete
    for p in procs:
        p.join()

    # Save metadata
    total_items = len(data)
    generated_files = [f for f in os.listdir(args.outdir) if f.endswith(".png")]
    total_generated = len(generated_files)
    total_existing = existing_count
    total_skipped = total_items - total_generated

    meta_info = {
        "model": os.path.abspath(args.model_path),
        "json_file": os.path.abspath(args.json_file),
        "total_items": total_items,
        "generated": total_generated,
        "existing": total_existing,
        "skipped": total_skipped,
        "seed": args.seed,
        "num_gpus": num_gpus,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(args.outdir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)
    print(f"\n📝 Metadata saved to {meta_path}")

    print("\n" + "=" * 50)
    print("GENERATION SUMMARY")
    print("=" * 50)
    print(f"Total items:     {total_items}")
    print(f"Existing:        {total_existing}")
    print(f"Generated:       {total_generated}")
    print(f"Skipped:         {total_skipped}")
    print(f"GPUs used:       {num_gpus}")
    print("=" * 50)
    print("🎯 All prompts processed (parallelized across GPUs).")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # ✅ crucial line
    main()

