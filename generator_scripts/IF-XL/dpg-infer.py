from pathlib import Path
import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_FILE = "/home/anirban/dheerajbaiju/dheeraj/dyneval-models/dpgbench-final-prompt-order.txt"
OUTPUT_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/if-xl-dpg-results"
MODEL_PATH = "/storage/users/anirban/dheerajbaiju/IF-I-XL-v1.0-model"

START_INDEX = 0  
NUM_STEPS = 28      
IMAGES_PER_PROMPT = 4
HEIGHT, WIDTH = 512, 512
GUIDANCE_SCALE = 3.5

# 🔥 New: batching config
BATCH_SIZE = 2   # number of prompts per forward pass (tune this based on VRAM)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_on_gpu(gpu_id, prompts_subset):
    """Generate images on a specific GPU using batched inference."""
    torch.cuda.set_device(gpu_id)

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(f"cuda:{gpu_id}")
    pipe.set_progress_bar_config(disable=True)

    print(f"🖥️ GPU {gpu_id}: Loaded model, {len(prompts_subset)} prompts")

    def is_prompt_done(stem):
        for k in range(IMAGES_PER_PROMPT):
            if not os.path.exists(os.path.join(OUTPUT_DIR, f"{stem}_{k}.png")):
                return False
        return True

    # Filter out already completed prompts
    remaining = [(stem, prompt) for stem, prompt in prompts_subset if not is_prompt_done(stem)]
    print(f"🧹 GPU {gpu_id}: {len(remaining)} remaining after skipping completed")

    # Batched generation
    for i in tqdm(range(0, len(remaining), BATCH_SIZE), desc=f"GPU {gpu_id}", position=gpu_id):
        batch = remaining[i:i + BATCH_SIZE]
        stems = [s for s, _ in batch]
        prompts = [p for _, p in batch]

        try:
            result = pipe(
                prompt=prompts,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                num_images_per_prompt=IMAGES_PER_PROMPT,
            )

            images = result.images  # length = BATCH_SIZE * IMAGES_PER_PROMPT

            idx = 0
            for stem in stems:
                for k in range(IMAGES_PER_PROMPT):
                    out_path = os.path.join(OUTPUT_DIR, f"{stem}_{k}.png")
                    if not os.path.exists(out_path):
                        images[idx].save(out_path)
                    idx += 1

        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            tqdm.write(f"🔥 GPU {gpu_id} OOM at batch starting {i}: {e}")
        except Exception as e:
            tqdm.write(f"❌ GPU {gpu_id} Error at batch starting {i}: {e}")


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Load prompts
    all_prompts = []
    with open(PROMPTS_FILE, "r") as f:
        for idx, line in enumerate(f):
            prompt = line.strip()
            if not prompt:
                continue
            stem = f"prompt{idx:05d}"
            all_prompts.append((stem, prompt))

    prompts = all_prompts[START_INDEX:]
    total_prompts = len(prompts)

    print(f"📋 Loaded {total_prompts} prompts (from index {START_INDEX} onwards)")
    print(f"📐 Image size: {WIDTH}x{HEIGHT}")
    print(f"🔧 Inference steps: {NUM_STEPS}")
    print(f"🎚️ Guidance scale: {GUIDANCE_SCALE}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"🖼️ Images per prompt: {IMAGES_PER_PROMPT}")

    num_gpus = torch.cuda.device_count()
    print(f"🖥️ Found {num_gpus} GPUs")

    if num_gpus < 2:
        print("⚠️ Only one GPU found, running single-GPU mode")
        generate_on_gpu(0, prompts)
    else:
        mid = len(prompts) // 2
        gpu0_prompts = prompts[:mid]
        gpu1_prompts = prompts[mid:]

        print(f"📊 GPU 0: {len(gpu0_prompts)} prompts")
        print(f"📊 GPU 1: {len(gpu1_prompts)} prompts")

        mp.set_start_method("spawn", force=True)

        p0 = mp.Process(target=generate_on_gpu, args=(0, gpu0_prompts))
        p1 = mp.Process(target=generate_on_gpu, args=(1, gpu1_prompts))

        p0.start()
        p1.start()

        p0.join()
        p1.join()

    print("\n🎯 All prompts processed successfully!")
