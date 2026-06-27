import json
import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from diffusers import BriaFiboPipeline
from diffusers.modular_pipelines import ModularPipelineBlocks
from tqdm import tqdm


# ================================
# CONFIG
# ================================
PROMPTS_FILE = "/home/anirban/dheerajbaiju/dheeraj/dyneval-models/geneval-final-prompt-order.txt"
OUTDIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/fibo-geneval-results"
IMAGES_PER_PROMPT = 4
DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_STEPS = 50
GUIDANCE_SCALE = 5
FIBO_VLM_DIR = "/home/anirban/arunabhsingh25/.cache/huggingface/hub/models--briaai--FIBO-VLM-prompt-to-JSON"
FIBO_DIR = "/home/anirban/arunabhsingh25/.cache/huggingface/hub/models--briaai--FIBO"


# ================================
# GLOBAL SPEED FLAGS
# ================================
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
os.makedirs(OUTDIR, exist_ok=True)


# ================================
# UTILS
# ================================
def hf_snapshot_dir(model_dir: str, revision: str = "main") -> str:
    model_path = Path(model_dir)
    refs_path = model_path / "refs" / revision
    if refs_path.exists():
        commit = refs_path.read_text().strip()
        snapshot_path = model_path / "snapshots" / commit
        if snapshot_path.exists():
            return str(snapshot_path)
    return str(model_path)


def get_default_negative_prompt(existing_json: dict) -> str:
    negative_prompt = ""
    style_medium = existing_json.get("style_medium", "").lower()
    if style_medium in ["photograph", "photography", "photo"]:
        negative_prompt = """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
    return negative_prompt


def load_prompts(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_completed_prompt_indices(outdir, images_per_prompt=4):
    completed = set()
    files = list(Path(outdir).glob("prompt*_*.png"))
    counter = {}

    for f in files:
        try:
            pid = int(f.stem.split("_")[0].replace("prompt", ""))
            counter[pid] = counter.get(pid, 0) + 1
        except:
            pass

    for k, v in counter.items():
        if v >= images_per_prompt:
            completed.add(k)

    return completed


def to_pil(img):
    if isinstance(img, Image.Image):
        return img

    if isinstance(img, np.ndarray):
        # Squeeze extra batch dims
        while img.ndim > 3:
            img = img[0]

        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return Image.fromarray(img)

    raise TypeError(f"Unsupported image type: {type(img)}")


# ================================
# LOAD VLM PIPELINE
# ================================
print("🔄 Loading FIBO VLM...")
vlm_pipe = ModularPipelineBlocks.from_pretrained(
    hf_snapshot_dir(FIBO_VLM_DIR),
    trust_remote_code=True
).init_pipeline()


# ================================
# LOAD IMAGE PIPELINE
# ================================
print("🔄 Loading FIBO diffusion pipeline...")
pipe = BriaFiboPipeline.from_pretrained(
    hf_snapshot_dir(FIBO_DIR),
    torch_dtype=DTYPE,
).to(DEVICE)


# ================================
# MAIN
# ================================
def main():
    prompts = load_prompts(PROMPTS_FILE)
    completed = get_completed_prompt_indices(OUTDIR, IMAGES_PER_PROMPT)

    print(f"📄 Loaded {len(prompts)} prompts")
    print(f"✅ Found {len(completed)} completed prompts (resume enabled)")

    for idx, prompt in enumerate(tqdm(prompts, desc="🖼️ Generating images")):
        if idx in completed:
            continue

        json_prompt_path = os.path.join(OUTDIR, f"prompt{idx}_json_prompt.json")

        # -------- Prompt → JSON (cached) --------
        if os.path.exists(json_prompt_path):
            json_prompt_generate = open(json_prompt_path).read()
        else:
            output = vlm_pipe(prompt=prompt)
            json_prompt_generate = output.values["json_prompt"]
            with open(json_prompt_path, "w") as f:
                f.write(json_prompt_generate)

        negative_prompt = get_default_negative_prompt(json.loads(json_prompt_generate))

        # -------- Batched image generation --------
        results = pipe(
            prompt=[json_prompt_generate] * IMAGES_PER_PROMPT,
            negative_prompt=[negative_prompt] * IMAGES_PER_PROMPT,
            num_inference_steps=NUM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        )

        for img_idx, img in enumerate(results.images):
            out_path = os.path.join(OUTDIR, f"prompt{idx}_{img_idx}.png")
            if os.path.exists(out_path):
                continue

            image = to_pil(img)
            image.save(out_path)

    print("🎉 All generations complete!")


if __name__ == "__main__":
    main()
