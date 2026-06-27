from pathlib import Path
import os
import torch
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_FILE = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark/prompts/evalmuse/evalmuse-test-prompts.txt"
OUTPUT_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/EVALMUSE/sdxl-turbo"
MODEL_PATH = "stabilityai/sdxl-turbo"

START_INDEX = 0  
NUM_STEPS = 1      
IMAGES_PER_PROMPT = 1
HEIGHT, WIDTH = 1024, 1024
GUIDANCE_SCALE = 0.0   # Turbo is trained for CFG=0

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_on_gpu(gpu_id, prompts_subset):
    """Generate images on a specific GPU."""
    torch.cuda.set_device(gpu_id)

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        guidance_scale=GUIDANCE_SCALE
    )
    pipe.to(f"cuda:{gpu_id}")
    pipe.set_progress_bar_config(disable=True)

    skipped_prompts = 0
    for stem, _ in prompts_subset:
        all_exist = True
        for k in range(IMAGES_PER_PROMPT):
            if not os.path.exists(os.path.join(OUTPUT_DIR, f"{stem}_{k}.png")):
                all_exist = False
                break
        if all_exist:
            skipped_prompts += 1

    print(f"🖥️ GPU {gpu_id}: Loaded model, {len(prompts_subset)} prompts "
          f"({skipped_prompts} already done, {len(prompts_subset) - skipped_prompts} remaining)")

    for stem, prompt in tqdm(prompts_subset, desc=f"GPU {gpu_id}", unit="prompt", position=gpu_id):
        all_exist = True
        for k in range(IMAGES_PER_PROMPT):
            if not os.path.exists(os.path.join(OUTPUT_DIR, f"{stem}_{k}.png")):
                all_exist = False
                break
        if all_exist:
            continue

        for k in range(IMAGES_PER_PROMPT):
            out_path = os.path.join(OUTPUT_DIR, f"{stem}_{k}.png")
            if os.path.exists(out_path):
                continue

            try:
                result = pipe(
                    prompt=prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                )
                result.images[0].save(out_path)
            except Exception as e:
                tqdm.write(f"❌ GPU {gpu_id} Error {stem}_{k}: {e}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load prompts from single file
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
