import os
import glob
import re
from pathlib import Path
from PIL import Image
from transformers import AutoModelForCausalLM
import torch

# =====================================
# CUDA OPTIMIZATIONS
# =====================================
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 for cudnn
# =====================================
# CONFIGURATION
# =====================================
MODEL_PATH = "../HunyuanImage-3.0-model"  # Local model path
PROMPTS_DIR = "/storage/users/anirban/yashwanthm/HunyuanImage-3.0/ELLA/dpg_bench/prompts"  # Directory with prompt txt files
OUTPUT_DIR = "dpg-bench-remianing"  # Output folder for remaining images
SKIP_LIST_FILE = "/storage/users/anirban/yashwanthm/HunyuanImage-3.0/files_with_3_dpg_already_done.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load list of files already done on another server
already_done_files = set()
if os.path.exists(SKIP_LIST_FILE):
    with open(SKIP_LIST_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line != '.':
                already_done_files.add(line)
    print(f"Loaded {len(already_done_files)} files to skip from skip list.")

# =====================================
# LOAD MODEL
# =====================================
print("Loading Hunyuan model...")

# Check available GPUs
num_gpus = torch.cuda.device_count()
print(f"Found {num_gpus} GPU(s) available")

kwargs = dict(
    attn_implementation="sdpa",  # Using PyTorch native SDPA (flash_attn not installed)
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,  # Explicit bfloat16 for faster computation
    device_map="auto",  # Automatically distribute across all GPUs
    moe_impl="eager",  # flashinfer failed to compile, using eager
)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
model.load_tokenizer(MODEL_PATH)

# Note: torch.compile is NOT compatible with Hunyuan's custom cache
# Other optimizations (flash_attention_2, flashinfer, bfloat16, TF32) are still active

print("Model loaded successfully.")

# =====================================
# READ ALL PROMPTS FROM DIRECTORY
# =====================================
prompt_files = sorted(glob.glob(os.path.join(PROMPTS_DIR, "*.txt")))
print(f"Found {len(prompt_files)} prompt files to process.")

# =====================================
# GENERATE IMAGES (from end of list, one image per prompt with _3 suffix)
# =====================================
# Use inference_mode for faster generation (no gradient tracking)
with torch.inference_mode():
    # Process prompts in reverse order (from end to beginning)
    for prompt_file in reversed(prompt_files):
        # Read prompt from file
        with open(prompt_file, 'r') as f:
            prompt = f.read().strip()
        
        # Get filename without extension
        prompt_filename = Path(prompt_file).stem
        
        # Generate one image per prompt with _3 suffix
        output_filename = os.path.join(OUTPUT_DIR, f"{prompt_filename}_3.png")
        output_basename = f"{prompt_filename}_3.png"
        
        # Check if file is in skip list (already done on another server)
        if output_basename in already_done_files:
            print(f"Skipping {output_basename}, already done on another server.")
            continue
            
        # Check if file already exists to support resuming
        if os.path.exists(output_filename):
            print(f"Skipping {output_filename}, already exists.")
            continue
            
        print(f"\n>>> Generating for {prompt_filename}: {prompt[:100]}...")
        
        # stream=True required for compatibility with this transformers version
        image = model.generate_image(prompt=prompt, image_size="1024x1024", stream=True)
        
        # Save image
        image.save(output_filename)
        print(f"  ✓ Saved: {output_filename}")

print(f"\n✅ Generation completed. All images saved in: {OUTPUT_DIR}")
