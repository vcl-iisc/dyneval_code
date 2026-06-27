import os
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image
from tqdm import tqdm

def load_prompts(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f if line.strip()]

def main():
    try:
        # =====================
        # CONFIG
        # =====================
        MODEL_ID = "kandinsky-community/kandinsky-3"
        PROMPTS_FILE = "/home/anirban/dheerajbaiju/dheeraj/dyneval-models/dpgbench-final-prompt-order.txt"   # <-- your .txt file (one prompt per line)
        OUTDIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/kandinsky3-dpg-results"
        IMAGES_PER_PROMPT = 4
        NUM_STEPS = 25
        SEED = 0

        # =====================
        # LOAD PROMPTS
        # =====================
        prompts = load_prompts(PROMPTS_FILE)
        print(f"📄 Loaded {len(prompts)} prompts")

        # =====================
        # LOAD PIPELINE
        # =====================
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            variant="fp16",
            torch_dtype=torch.float16
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)

        # Speed-friendly flags
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        os.makedirs(OUTDIR, exist_ok=True)

        # =====================
        # GENERATE
        # =====================
        generator = torch.Generator(device=device).manual_seed(SEED)

        for idx, prompt in enumerate(tqdm(prompts, desc="🖼️ Generating")):
            with torch.inference_mode():
                result = pipe(
                    [prompt] * IMAGES_PER_PROMPT,   # batch of same prompt
                    num_inference_steps=NUM_STEPS,
                    generator=generator
                )

            for img_idx, image in enumerate(result.images):
                out_path = os.path.join(OUTDIR, f"prompt{idx}_{img_idx}.png")
                image.save(out_path)

        print("succ")
        print(f"✅ All images saved to: {OUTDIR}")

    except Exception as e:
        print("❌ failed")
        print("Error:", str(e))


if __name__ == "__main__":
    main()
