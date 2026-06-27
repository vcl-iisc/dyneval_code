from pathlib import Path
import glob
import os
import torch
from diffusers import FluxPipeline
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark/prompts/dpg_bench"
OUTPUT_DIR = "./INCONTEXTLLORA-EVALMUSE-TEST-IMAGES"
MODEL_PATH = "/storage/users/anirban/dheerajbaiju/FLUX.1-dev-model"
LORA_PATH = "ali-vilab/In-Context-LoRA"

START_INDEX = 0  # Start from 2001st prompt (0-indexed)
NUM_STEPS = 18     # Reduced steps for faster generation (FLUX works well with fewer steps)
HEIGHT, WIDTH = 512, 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_on_gpu(gpu_id, prompts_subset, start_idx):
    """Generate images on a specific GPU."""
    torch.cuda.set_device(gpu_id)
    
    # Load pipeline with optimizations
    pipe = FluxPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    pipe.to(f"cuda:{gpu_id}")
    pipe.load_lora_weights(LORA_PATH)
    
    # Enable memory optimizations
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    
    # Compile for faster inference (PyTorch 2.0+)
    try:
        pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    except Exception as e:
        print(f"⚠️ GPU {gpu_id}: torch.compile not available or failed: {e}")

    
    # Count already generated images for resume info
    skipped = 0
    for i in range(len(prompts_subset)):
        stem = prompts_subset[i][0]
        if os.path.exists(os.path.join(OUTPUT_DIR, f"{idx:05d}.png")):
            skipped += 1
    
    print(f"🖥️ GPU {gpu_id}: Loaded model, {len(prompts_subset)} prompts ({skipped} already done, {len(prompts_subset) - skipped} remaining)")
    
    for i, (stem, prompt) in enumerate(tqdm(prompts_subset, desc=f"GPU {gpu_id}", unit="prompt", position=gpu_id)):
        output_filename = f"{idx:05d}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        if os.path.exists(output_path):
            continue

        try:
            result = pipe(
                prompt=prompt,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_STEPS,
            )
            image = result.images[0]
            image.save(output_path)

        except Exception as e:
            tqdm.write(f"❌ GPU {gpu_id} Error {output_filename}: {e}")

if __name__ == "__main__":
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Load prompts
    prompt_files = sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.txt")))
    all_prompts = []
    for p_file in prompt_files:
        stem = Path(p_file).stem
        with open(p_file, "r") as f:
            text = f.read().strip()
        all_prompts.append((stem, text))
    
    # Get prompts from START_INDEX to end
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
        # Split prompts between GPUs
        mid = len(prompts) // 2
        gpu0_prompts = prompts[:mid]
        gpu1_prompts = prompts[mid:]
        
        # Calculate actual indices
        gpu0_start = START_INDEX
        gpu1_start = START_INDEX + mid
        
        print(f"📊 GPU 0: prompts {gpu0_start}-{gpu0_start + len(gpu0_prompts) - 1} ({len(gpu0_prompts)} prompts)")
        print(f"📊 GPU 1: prompts {gpu1_start}-{gpu1_start + len(gpu1_prompts) - 1} ({len(gpu1_prompts)} prompts)")
        
        # Start processes for each GPU
        mp.set_start_method('spawn', force=True)
        
        p0 = mp.Process(target=generate_on_gpu, args=(0, gpu0_prompts, gpu0_start))
        p1 = mp.Process(target=generate_on_gpu, args=(1, gpu1_prompts, gpu1_start))
        
        p0.start()
        p1.start()
        
        p0.join()
        p1.join()
    
    print("\n🎯 All prompts processed successfully!")