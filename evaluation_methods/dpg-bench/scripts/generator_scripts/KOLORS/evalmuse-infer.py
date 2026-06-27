from pathlib import Path
import glob
import os
import gc
import torch
from diffusers import KolorsPipeline
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# GLOBAL TORCH SAFETY (CRITICAL FOR KOLORS)
# =====================================================
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark/prompts/dpg_bench"
OUTPUT_DIR = "./KOLORS-EVALMUSE-TRAIN-first-half-RESULTS"
MODEL_PATH = "/home/anirban/dheerajbaiju/dheeraj/instance-seg/KOLORS/Kolors-model"

START_INDEX = 0  # Start from prompt 0 (0-indexed)
END_INDEX = 2990  # End at prompt 2990 (will generate 02990.png)
NUM_STEPS = 18
HEIGHT, WIDTH = 512, 512

RELOAD_EVERY = 100                 # hard pipeline reset
MAX_PROMPT_CHARS = 300             # prevents conditioning bugs
FALLBACK_PROMPT = "a high quality photo"
NEGATIVE_PROMPT = ""               # 🔥 REQUIRED FOR KOLORS (MUST NOT BE None)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# PROMPT SANITIZATION
# =====================================================
def sanitize_prompt(prompt: str) -> str:
    if prompt is None:
        return FALLBACK_PROMPT

    prompt = prompt.strip()
    if len(prompt) == 0:
        return FALLBACK_PROMPT

    prompt = prompt[:MAX_PROMPT_CHARS]
    prompt = prompt.encode("utf-8", errors="ignore").decode("utf-8")

    return prompt

# =====================================================
# PIPELINE LOADER
# =====================================================
def load_pipeline(gpu_id: int):
    pipe = KolorsPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
    ).to(f"cuda:{gpu_id}")

    return pipe

# =====================================================
# GPU WORKER
# =====================================================
def generate_on_gpu(gpu_id: int, prompts_subset, start_idx: int):
    torch.cuda.set_device(gpu_id)

    pipe = load_pipeline(gpu_id)

    print(f"🖥️ GPU {gpu_id}: Loaded model ({len(prompts_subset)} prompts)")

    for i, raw_prompt in enumerate(
        tqdm(prompts_subset, desc=f"GPU {gpu_id}", unit="prompt", position=gpu_id)
    ):
        output_path = os.path.join(OUTPUT_DIR, f"{idx:05d}.png")

        if os.path.exists(output_path):
            continue

        prompt = sanitize_prompt(raw_prompt)

        try:
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,  # 🔥 MANDATORY
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_STEPS,
                )

            result.images[0].save(output_path)

        except Exception as e:
            tqdm.write(f"⚠️ GPU {gpu_id}: Error at {idx:05d}, recovering ({e})")

            # hard recovery
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
            pipe = load_pipeline(gpu_id)

            # guaranteed-safe retry
            result = pipe(
                prompt=FALLBACK_PROMPT,
                negative_prompt=NEGATIVE_PROMPT,  # 🔥 MANDATORY
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_STEPS,
            )
            result.images[0].save(output_path)

        # periodic hard reset
        if (i + 1) % RELOAD_EVERY == 0:
            tqdm.write(f"🔄 GPU {gpu_id}: Hard reload at step {i + 1}")
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
            pipe = load_pipeline(gpu_id)

    del pipe
    torch.cuda.empty_cache()
    gc.collect()

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        all_prompts = [line.rstrip("\n") for line in f]

    # Get prompts from START_INDEX to END_INDEX (inclusive)
    prompts = all_prompts[START_INDEX:END_INDEX + 1]
    total = len(prompts)

    print(f"📋 Loaded {total} prompts (from index {START_INDEX} to {END_INDEX}, generating 00000.png to {END_INDEX:05d}.png)")

    num_gpus = torch.cuda.device_count()
    print(f"🖥️ Found {num_gpus} GPUs")

    if num_gpus < 2:
        print("⚠️ Running on single GPU")
        generate_on_gpu(0, prompts, START_INDEX)
    else:
        mid = total // 2
        gpu0_start = START_INDEX
        gpu1_start = START_INDEX + mid
        p0 = mp.Process(target=generate_on_gpu, args=(0, prompts[:mid], gpu0_start))
        p1 = mp.Process(target=generate_on_gpu, args=(1, prompts[mid:], gpu1_start))

        p0.start()
        p1.start()
        p0.join()
        p1.join()

    print("\n🎯 All prompts processed successfully!")
