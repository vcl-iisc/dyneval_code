from pathlib import Path
import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_FILE = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark/prompts/evalmuse/evalmuse-test-prompts.txt"
OUTPUT_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/EVALMUSE/sdxl"
MODEL_PATH = "stabilityai/sdxl-turbo"

START_INDEX = 0  
NUM_STEPS = 1                 # SDXL-Turbo is 1-step model (2–4 max if you want)
IMAGES_PER_PROMPT = 1
HEIGHT, WIDTH = 1024, 1024   # SDXL native resolution

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_prompts(txt_file):
    with open(txt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def generate_on_gpu(gpu_id, prompts_subset, global_start_idx):
    torch.cuda.set_device(gpu_id)

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,   # safer than bf16 for SDXL-Turbo
    )
    pipe.to(f"cuda:{gpu_id}")

    print(f"🖥️ GPU {gpu_id}: Loaded model, {len(prompts_subset)} prompts")

    for local_i, prompt in enumerate(tqdm(prompts_subset, desc=f"GPU {gpu_id}", unit="prompt", position=gpu_id)):
        global_i = global_start_idx + local_i

        for k in range(IMAGES_PER_PROMPT):
            output_path = os.path.join(OUTPUT_DIR, f"{global_i}_{k}.png")
            if os.path.exists(output_path):
                continue

            try:
                image = pipe(
                    prompt=prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_STEPS,
                ).images[0]

                image.save(output_path)

            except Exception as e:
                tqdm.write(f"❌ GPU {gpu_id} Error prompt {global_i}, img {k}: {e}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    prompts = load_prompts(PROMPTS_FILE)
    prompts = prompts[START_INDEX:]
    total_prompts = len(prompts)

    print(f"📋 Loaded {total_prompts} prompts (starting from index {START_INDEX})")

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

        print(f"📊 GPU 0: prompts {gpu0_start}-{gpu0_start + len(gpu0_prompts) - 1}")
        print(f"📊 GPU 1: prompts {gpu1_start}-{gpu1_start + len(gpu1_prompts) - 1}")

        mp.set_start_method("spawn", force=True)

        p0 = mp.Process(target=generate_on_gpu, args=(0, gpu0_prompts, gpu0_start))
        p1 = mp.Process(target=generate_on_gpu, args=(1, gpu1_prompts, gpu1_start))

        p0.start()
        p1.start()
        p0.join()
        p1.join()

    print("\n🎯 All prompts processed successfully!")
