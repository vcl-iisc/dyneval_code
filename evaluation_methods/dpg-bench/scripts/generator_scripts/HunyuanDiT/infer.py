import os
import torch
from diffusers import HunyuanDiTPipeline
from torchvision.utils import make_grid
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================
# =====================================================
# CONFIGURATION
# =====================================================
PROMPT_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark/prompts/dpg_bench"
OUTPUT_DIR = "./dpg-bench-outputs"           # folder to save generated images
MODEL_PATH = "hunyuandit-model-path"

IMAGES_PER_PROMPT = 4
GUIDANCE_SCALE = 7.5
NUM_STEPS = 60
HEIGHT, WIDTH = 1024, 1024
NEGATIVE_PROMPT = "blurry, low resolution"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
pipe = HunyuanDiTPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to("cuda")

pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

# =====================================================
# COLLECT PROMPTS
# =====================================================
prompt_files = sorted([f for f in os.listdir(PROMPT_DIR) if f.endswith(".txt")])
total_prompts = len(prompt_files)

if total_prompts == 0:
    print("⚠️ No .txt files found in prompt directory!")
    exit()

# =====================================================
# MAIN GENERATION LOOP WITH RESUME + STATUS BAR
# =====================================================
for file in tqdm(prompt_files, desc="🧠 Generating images", unit="prompt"):
    stem = os.path.splitext(file)[0]
    
    # Check if all images exist
    all_exist = True
    for k in range(IMAGES_PER_PROMPT):
        if not os.path.exists(os.path.join(OUTPUT_DIR, f"{stem}_{k}.png")):
            all_exist = False
            break
            
    if all_exist:
        continue

    # Read prompt
    prompt_path = os.path.join(PROMPT_DIR, file)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    tqdm.write(f"🎨 Generating for prompt: {file}")

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            num_images_per_prompt=IMAGES_PER_PROMPT
        )

        images = result.images
        for k, img in enumerate(images):
             img.save(os.path.join(OUTPUT_DIR, f"{stem}_{k}.png"))
        
        tqdm.write(f"✅ Saved images for: {stem}")

    except Exception as e:
        tqdm.write(f"❌ Error generating {file}: {e}")

print("\n🎯 All prompts processed successfully!")
