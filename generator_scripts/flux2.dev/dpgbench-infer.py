from pathlib import Path
import glob
#!/usr/bin/env python3
import os
import torch, random
from diffusers import Flux2Pipeline
from tqdm import tqdm
import multiprocessing as mp
import argparse

# =====================================================
# CONFIGURATION
# =====================================================

parser = argparse.ArgumentParser(description="FLUX2 DPG-Bench Evaluation")

parser.add_argument(
    "--prompts_file",
    type=str,
    default="../prompts/dpgbench_prompts.txt",
    help="Path to the prompts text file"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="../output/flux2-dev-dpgbench-results",
    help="Directory to save generated outputs"
)

parser.add_argument(
    "--model_path",
    type=str,
    default="/storage/users/anirban/dheerajbaiju/FLUX2-dev-model",
    help="Local path to FLUX2-dev model"
)

args = parser.parse_args()

PROMPTS_FILE = args.prompts_file
OUTPUT_DIR = args.output_dir
REPO_ID = args.model_path

START_INDEX = 0  
NUM_STEPS = 50      
IMAGES_PER_PROMPT = 4
HEIGHT, WIDTH = 1024, 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# WORKER
# =====================================================

def generate_on_gpu(gpu_id, prompts_subset, start_idx):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    print(f"🖥️ Loading FLUX2 pipeline on {device}...")
    pipe = Flux2Pipeline.from_pretrained(
        REPO_ID,
        torch_dtype=torch.bfloat16
    )
    pipe.to(device)

    # Safe memory optimization supported by FLUX2
    pipe.enable_attention_slicing()

    skipped = 0
    for i in range(len(prompts_subset)):
        stem = prompts_subset[i][0]
        if os.path.exists(os.path.join(OUTPUT_DIR, f"{stem}_0.png")):
            skipped += 1

    print(f"🖥️ GPU {gpu_id}: {len(prompts_subset)} prompts ({skipped} already done, {len(prompts_subset) - skipped} remaining)")

    for i, (stem, prompt) in enumerate(tqdm(prompts_subset, desc=f"GPU {gpu_id}", unit="prompt", position=gpu_id)):
        for jk in range(IMAGES_PER_PROMPT):
            out_path = os.path.join(OUTPUT_DIR, f"{stem}_{jk}.png")

            if os.path.exists(out_path):
                continue

            try:
                result = pipe(
                    prompt=prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=4.0,
                    generator=torch.Generator(device=device).manual_seed(42+jk),
                )
                result.images[0].save(out_path)

            except Exception as e:
                import traceback
                traceback.print_exc()
                tqdm.write(f"❌ GPU {gpu_id} Error {stem}_0.png: {e}")

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prompt_files = sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.txt")))
    all_prompts = []
    for p_file in prompt_files:
        stem = Path(p_file).stem
        with open(p_file, "r") as f:
            text = f.read().strip()
        all_prompts.append((stem, text))

    prompts = all_prompts[START_INDEX:]
    total_prompts = len(prompts)

    print(f"📋 Loaded {total_prompts} prompts (from index {START_INDEX} to {START_INDEX + total_prompts - 1})")

    num_gpus = torch.cuda.device_count()
    print(f"🖥️ Found {num_gpus} GPUs")
    print(f"📐 Image size: {WIDTH}x{HEIGHT}")
    print(f"🔧 Inference steps: {NUM_STEPS}")

    if num_gpus < 2:
        print("⚠️ Less than 2 GPUs found, running on single GPU")
        generate_on_gpu(0, prompts, START_INDEX)
    else:
        mid = len(prompts) // 2
        gpu0_prompts = prompts[:mid]
        gpu1_prompts = prompts[mid:]

        gpu0_start = START_INDEX
        gpu1_start = START_INDEX + mid

        print(f"📊 GPU 0: prompts {gpu0_start}-{gpu0_start + len(gpu0_prompts) - 1} ({len(gpu0_prompts)})")
        print(f"📊 GPU 1: prompts {gpu1_start}-{gpu1_start + len(gpu1_prompts) - 1} ({len(gpu1_prompts)})")

        mp.set_start_method("spawn", force=True)

        p0 = mp.Process(target=generate_on_gpu, args=(0, gpu0_prompts, gpu0_start))
        p1 = mp.Process(target=generate_on_gpu, args=(1, gpu1_prompts, gpu1_start))

        p0.start()
        p1.start()

        p0.join()
        p1.join()

    print("\n🎯 All prompts processed successfully!")
