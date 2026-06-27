from pathlib import Path
import glob
import os
import torch
from flux_klein.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.utils import load_image
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark/prompts/dpg_bench"
OUTPUT_DIR = "./flux2-klein-9B-dpgbench-results"
MODEL_PATH = "/storage/users/anirban/dheerajbaiju/FLUX.2-klein-9B-model"

START_INDEX = 0  
NUM_STEPS = 4      
IMAGES_PER_PROMPT = 1
HEIGHT, WIDTH = 1024, 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_on_gpu(gpu_id, prompts_subset, start_idx):
    """Generate images on a specific GPU."""
    torch.cuda.set_device(gpu_id)
    
    # Load pipeline with optimizations
    # Load pipeline with optimizations
    # Load pipeline with optimizations
    pipe = Flux2KleinPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    
    # Enable memory optimizations
    # Enable memory optimizations
    # pipe.enable_vae_slicing()
    # pipe.enable_vae_tiling()
    
    # Compile for faster inference (PyTorch 2.0+)
    # try:
    #     pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
    # except Exception as e:
    #     print(f"⚠️ GPU {gpu_id}: torch.compile not available or failed: {e}")
    
    # Count already generated images for resume info
    skipped_prompts = 0
    for i in range(len(prompts_subset)):
        stem = prompts_subset[i][0]
        # check if all 4 images exist
        all_exist = True
        for k in range(IMAGES_PER_PROMPT):
            
            fname = f"{stem}_{k}.png"
                
            if not os.path.exists(os.path.join(OUTPUT_DIR, fname)):
                all_exist = False
                break
        if all_exist:
            skipped_prompts += 1
    
    print(f"🖥️ GPU {gpu_id}: Loaded model, {len(prompts_subset)} prompts ({skipped_prompts} already done, {len(prompts_subset) - skipped_prompts} remaining)")
    
    for i, (stem, prompt) in enumerate(tqdm(prompts_subset, desc=f"GPU {gpu_id}", unit="prompt", position=gpu_id)):
        
        # Check if all images for this prompt exist
        all_exist = True
        for k in range(IMAGES_PER_PROMPT):
            fname = f"{stem}_{k}.png"

            if not os.path.exists(os.path.join(OUTPUT_DIR, fname)):
                all_exist = False
                break
        
        if all_exist:
            continue

        for k in range(IMAGES_PER_PROMPT):
            output_filename = f"{stem}_{k}.png"
                
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            if os.path.exists(output_path):
                continue
                
            try:
                result = pipe(
                    prompt=prompt,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=1.0,
                )
                image = result.images[0]
                image.save(output_path)

            except Exception as e:
                import traceback
                traceback.print_exc()
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
