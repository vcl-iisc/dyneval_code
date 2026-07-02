import os
import torch
from diffusers import HunyuanDiTPipeline
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_FILE = "prompts/all_prompts.txt"
OUTPUT_DIR = "./hunyuandit-evalmuse"
MODEL_PATH = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"  # Or your local model path

NUM_STEPS = 50
HEIGHT, WIDTH = 1024, 1024
GUIDANCE_SCALE = 5.0
NEGATIVE_PROMPT = "blurry, low quality"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading HunyuanDiT model...")
pipe = HunyuanDiTPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to("cuda")
print("Model loaded!")

# =====================================================
# LOAD PROMPTS
# =====================================================
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    all_prompts = [line.strip() for line in f if line.strip()]

total_prompts = len(all_prompts)
if total_prompts == 0:
    print("⚠️ No prompts found!")
    exit()

print(f"📋 Loaded {total_prompts} prompts")

# =====================================================
# MAIN GENERATION LOOP WITH RESUME SUPPORT
# =====================================================
for idx, prompt in enumerate(tqdm(all_prompts, desc="Generating", unit="prompt")):
    output_filename = f"{idx:05d}.png"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Skip if already exists (resume support)
    if os.path.exists(output_path):
        continue

    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        )
        image = result.images[0]
        image.save(output_path)

    except Exception as e:
        tqdm.write(f"❌ Error {output_filename}: {e}")

print("\n🎯 All prompts processed!")