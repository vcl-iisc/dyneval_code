import os
import torch
from diffusers import FluxPipeline
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_FILE = "/home/anirban/dheerajbaiju/dheeraj/dyneval-models/dpgbench-final-prompt-order.txt"
OUTPUT_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/incontextllora-dpgbench-results"
MODEL_PATH = "/storage/users/anirban/dheerajbaiju/FLUX.1-dev-model"

# HF LoRA repo (NOT a local path)
LORA_REPO_ID = "ali-vilab/In-Context-LoRA"

START_INDEX = 0
NUM_STEPS = 50
HEIGHT, WIDTH = 1024, 1024
IMAGES_PER_PROMPT = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_on_gpu(gpu_id, prompts_subset, start_idx):
    torch.cuda.set_device(gpu_id)

    # Load FLUX pipeline
    pipe = FluxPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(f"cuda:{gpu_id}")

    # ======================
    # Load In-Context LoRA
    # ======================
    pipe.load_lora_weights(LORA_REPO_ID)

    # Some FLUX builds need this explicitly
    try:
        pipe.set_adapters(["default"])
    except Exception:
        pass

    # Optional: fuse LoRA for speed
    try:
        pipe.fuse_lora(lora_scale=1.0)
    except Exception as e:
        print(f"⚠️ GPU {gpu_id}: LoRA fuse skipped: {e}")

    # Memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # Optional compile (safe to fail)
    try:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    except Exception as e:
        print(f"⚠️ GPU {gpu_id}: torch.compile failed: {e}")

    print(f"🖥️ GPU {gpu_id}: Model + LoRA loaded. Generating {len(prompts_subset)} prompts.")

    for i, prompt in enumerate(tqdm(prompts_subset, desc=f"GPU {gpu_id}", unit="prompt", position=gpu_id)):
        idx = start_idx + i

        for k in range(IMAGES_PER_PROMPT):
            out_name = f"prompt{idx}_{k}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            if os.path.exists(out_path):
                continue

            try:
                image = pipe(
                    prompt=prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_STEPS,
                ).images[0]

                image.save(out_path)

            except Exception as e:
                tqdm.write(f"❌ GPU {gpu_id} Error prompt{idx}_{k}: {e}")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load prompts
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    prompts = all_prompts[START_INDEX:]
    print(f"📋 Loaded {len(prompts)} prompts")

    num_gpus = torch.cuda.device_count()
    print(f"🖥️ Found {num_gpus} GPUs")
    print(f"📐 Image size: {WIDTH}x{HEIGHT}")
    print(f"🔧 Inference steps: {NUM_STEPS}")
    print(f"🖼️ Images per prompt: {IMAGES_PER_PROMPT}")

    if num_gpus < 2:
        print("⚠️ Running on single GPU")
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
