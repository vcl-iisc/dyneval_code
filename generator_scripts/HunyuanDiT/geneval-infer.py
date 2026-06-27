import os
import torch
from diffusers import HunyuanDiTPipeline
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================
PROMPTS_FILE = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark/prompts/evalmuse/evalmuse-test-prompts.txt"
OUTPUT_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/EVALMUSE/hunyuandit"
MODEL_PATH = "Tencent-Hunyuan/HunyuanDiT-v1.2"

IMAGES_PER_PROMPT = 1
GUIDANCE_SCALE = 7.5
NUM_STEPS = 50
HEIGHT, WIDTH = 1024, 1024
NEGATIVE_PROMPT = "blurry, low resolution"

START_INDEX = 0
MAX_PROMPTS = None   # None = all prompts

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD PROMPTS FROM TXT
# =====================================================
with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
    all_prompts = [line.strip() for line in f if line.strip()]

if MAX_PROMPTS is not None:
    prompts = all_prompts[START_INDEX:START_INDEX + MAX_PROMPTS]
else:
    prompts = all_prompts[START_INDEX:]

print(f"Loaded {len(prompts)} prompts from TXT")

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
# MAIN GENERATION LOOP (SAVE IMAGES SEPARATELY)
# =====================================================
for idx, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="🧠 Generating images", unit="prompt"):
    prompt_idx = START_INDEX + idx

    try:
        for img_i in range(IMAGES_PER_PROMPT):
            output_filename = f"prompt{prompt_idx}_{img_i}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            # Resume support per image
            if os.path.exists(output_path):
                tqdm.write(f"⏭️ Skipping {output_filename} (already exists)")
                continue

            result = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE_SCALE
            )

            image = result.images[0]
            image.save(output_path)

            tqdm.write(f"✅ Saved: {output_path}")

    except Exception as e:
        tqdm.write(f"❌ Error generating prompt{prompt_idx}: {e}")

print("\n🎯 All prompts processed successfully!")
