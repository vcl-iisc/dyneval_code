import os
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_FILE = "/home/anirban/dheerajbaiju/dheeraj/dyneval-models/dpgbench-final-prompt-order.txt"
OUTPUT_DIR = "./longcat-dpg-results"
MODEL_PATH = "/home/anirban/yashwanthm/dheeraj/LongCat/LongCat-Image-model"

START_INDEX = 0  
NUM_STEPS = 50      
IMAGES_PER_PROMPT = 4
HEIGHT, WIDTH = 1024, 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_on_gpu(gpu_id, prompts_subset, start_idx):
    torch.cuda.set_device(gpu_id)

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(f"cuda:{gpu_id}")

    print(f"🖥️ GPU {gpu_id}: Model loaded")

    for i, prompt in enumerate(tqdm(prompts_subset, desc=f"GPU {gpu_id}", position=gpu_id)):
        idx = start_idx + i

        for k in range(IMAGES_PER_PROMPT):
            out_path = os.path.join(OUTPUT_DIR, f"prompt{idx}_{k}.png")
            if os.path.exists(out_path):
                continue

            try:
                image = pipe(
                    prompt=prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=4.0,
                ).images[0]

                image.save(out_path)

            except Exception as e:
                tqdm.write(f"❌ GPU {gpu_id} Error prompt{idx}_{k}: {e}")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with open(PROMPTS_FILE, "r") as f:
        prompts = [l.strip() for l in f if l.strip()]

    num_gpus = torch.cuda.device_count()
    print(f"🖥️ Found {num_gpus} GPUs")

    if num_gpus < 2:
        generate_on_gpu(0, prompts, START_INDEX)
    else:
        mid = len(prompts) // 2
        mp.set_start_method("spawn", force=True)

        p0 = mp.Process(target=generate_on_gpu, args=(0, prompts[:mid], START_INDEX))
        p1 = mp.Process(target=generate_on_gpu, args=(1, prompts[mid:], START_INDEX + mid))

        p0.start()
        p1.start()
        p0.join()
        p1.join()

    print("🎯 Done!")
