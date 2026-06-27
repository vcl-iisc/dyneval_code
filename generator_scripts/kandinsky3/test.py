import os
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

def main():
    try:
        # =====================
        # CONFIG
        # =====================
        MODEL_ID = "kandinsky-community/kandinsky-3"
        OUT_PATH = "kandinsky_output.png"
        PROMPT = (
            "A photograph of the inside of a subway train. "
            "There are raccoons sitting on the seats. "
            "One of them is reading a newspaper. "
            "The window shows the city in the background."
        )
        NUM_STEPS = 25
        SEED = 0

        # =====================
        # LOAD PIPELINE
        # =====================
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID,
            variant="fp16",
            torch_dtype=torch.float16
        )

        # Device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.to(device)

        # ⚠️ Do NOT enable xformers (your build has no CUDA support)
        # pipe.enable_xformers_memory_efficient_attention()  ❌ REMOVE

        # Optional: speed-friendly flags (safe)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # =====================
        # GENERATE
        # =====================
        generator = torch.Generator(device=device).manual_seed(SEED)

        with torch.inference_mode():
            result = pipe(
                PROMPT,
                num_inference_steps=NUM_STEPS,
                generator=generator
            )

        image = result.images[0]

        # =====================
        # SAVE IMAGE
        # =====================
        os.makedirs("outputs", exist_ok=True)
        out_path = os.path.join("outputs", OUT_PATH)
        image.save(out_path)

        print("succ")
        print(f"✅ Image saved to: {out_path}")

    except Exception as e:
        print("❌ failed")
        print("Error:", str(e))


if __name__ == "__main__":
    main()
