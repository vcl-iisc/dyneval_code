# coding=utf-8
import argparse
from pathlib import Path
import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# ARGPARSE
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)

    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--images_per_prompt", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    return parser.parse_args()


# =====================================================
# GPU WORKER
# =====================================================
def generate_on_gpu(gpu_id, prompts_subset, args):
    """Generate images on a specific GPU."""
    torch.cuda.set_device(gpu_id)

    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to(f"cuda:{gpu_id}")

    pipe.set_progress_bar_config(disable=True)

    skipped = 0
    for stem, _ in prompts_subset:
        if all(os.path.exists(os.path.join(args.output_dir, f"{stem}_{k}.png")) for k in range(args.images_per_prompt)):
            skipped += 1

    print(f"🖥️ GPU {gpu_id}: {len(prompts_subset)} prompts | {skipped} already done | {len(prompts_subset) - skipped} remaining")

    for stem, prompt in tqdm(prompts_subset, desc=f"GPU {gpu_id}", position=gpu_id):
        for k in range(args.images_per_prompt):
            out_path = os.path.join(args.output_dir, f"{stem}_{k}.png")

            if os.path.exists(out_path):
                continue

            try:
                result = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_steps,
                )
                image = result.images[0]
                image.save(out_path)

            except Exception as e:
                tqdm.write(f"❌ GPU {gpu_id} | {stem}_{k}.png failed: {e}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)

    # =====================================================
    # LOAD PROMPTS
    # =====================================================
    with open(args.prompts_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    total_prompts = len(lines)

    start_idx = args.start_index
    end_idx = args.end_index if args.end_index != -1 else total_prompts

    assert 0 <= start_idx < total_prompts, "❌ start_index out of range"
    assert start_idx < end_idx <= total_prompts, "❌ end_index out of range"

    selected_lines = lines[start_idx:end_idx]

    all_prompts = []
    for i, prompt in enumerate(selected_lines, start=start_idx):
        stem = f"prompt{i}"
        all_prompts.append((stem, prompt))

    print(f"📋 Loaded prompts [{start_idx}:{end_idx}] → {len(all_prompts)} prompts")
    print(f"📐 Image size: {args.width}x{args.height}")
    print(f"🔁 Images per prompt: {args.images_per_prompt}")
    print(f"🧠 Model: {args.model_path}")
    print(f"📂 Output dir: {args.output_dir}")

    num_gpus = torch.cuda.device_count()
    print(f"🖥️ Found {num_gpus} GPUs")

    if num_gpus == 0:
        raise RuntimeError("❌ No GPUs found!")

    elif num_gpus == 1:
        generate_on_gpu(0, all_prompts, args)

    else:
        mp.set_start_method("spawn", force=True)
        chunks = [all_prompts[i::num_gpus] for i in range(num_gpus)]

        procs = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=generate_on_gpu, args=(gpu_id, chunks[gpu_id], args))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    print("\n🎯 All prompts processed successfully!")
