# coding=utf-8
import json
import os
import argparse
import torch
from tqdm import tqdm
from diffusers import BriaFiboPipeline
from diffusers.modular_pipelines import ModularPipelineBlocks

# =====================================================
# ARGPARSE
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True, help="Single txt file, one prompt per line")
    parser.add_argument("--output_dir", type=str, required=True)

    # Keep current defaults
    parser.add_argument("--vlm_model", type=str, default="briaai/FIBO-VLM-prompt-to-JSON")
    parser.add_argument("--image_model", type=str, default="briaai/FIBO")

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1)

    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--images_per_prompt", type=int, default=1)
    return parser.parse_args()


# -------------------------------
# Helpers
# -------------------------------
def get_default_negative_prompt(existing_json: dict) -> str:
    negative_prompt = ""
    style_medium = existing_json.get("style_medium", "").lower()
    if style_medium in ["photograph", "photography", "photo"]:
        negative_prompt = """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
    return negative_prompt


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.set_grad_enabled(False)
    assert torch.cuda.device_count() >= 1, "❌ No GPU found"

    # -------------------------------
    # Load VLM pipeline (prompt -> JSON)
    # -------------------------------
    print("🧠 Loading VLM...")
    vlm_pipe = ModularPipelineBlocks.from_pretrained(args.vlm_model, trust_remote_code=True)
    vlm_pipe = vlm_pipe.init_pipeline()

    # -------------------------------
    # Load FIBO image pipeline
    # -------------------------------
    print("🎨 Loading FIBO image pipeline...")
    pipe = BriaFiboPipeline.from_pretrained(
        args.image_model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    # -------------------------------
    # Load prompts
    # -------------------------------
    with open(args.prompts_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    total_prompts = len(lines)
    print(f"📋 Total prompts loaded: {total_prompts}")

    start_idx = args.start_index
    end_idx = args.end_index if args.end_index != -1 else total_prompts

    if not (0 <= start_idx < total_prompts):
        raise ValueError(f"❌ start_index {start_idx} out of range [0, {total_prompts-1}]")
    if not (start_idx < end_idx <= total_prompts):
        raise ValueError(f"❌ end_index {end_idx} out of range ({start_idx+1} .. {total_prompts})")

    selected = lines[start_idx:end_idx]

    # -------------------------------
    # Resume logic (filter fully completed)
    # -------------------------------
    filtered = []
    already_done = 0

    for i, prompt in enumerate(selected, start=start_idx):
        all_done = True
        for k in range(args.images_per_prompt):
            out_path = os.path.join(args.output_dir, f"prompt{i}_{k}.png")
            if not os.path.exists(out_path):
                all_done = False
                break
        if all_done:
            already_done += 1
        else:
            filtered.append((i, prompt))

    print(f"📋 Selected prompts [{start_idx}:{end_idx}] → {len(selected)}")
    print(f"✅ Already completed: {already_done}")
    print(f"🚀 Remaining to generate: {len(filtered)}")

    if len(filtered) == 0:
        print("🎉 Nothing left to generate.")
        exit(0)

    generated = skipped = errors = 0
    pbar = tqdm(filtered)

    # -------------------------------
    # Generation loop
    # -------------------------------
    for idx, prompt in pbar:
        for k in range(args.images_per_prompt):
            img_path = os.path.join(args.output_dir, f"prompt{idx}_{k}.png")
            json_path = os.path.join(args.output_dir, f"prompt{idx}_{k}.json")

            # -------- RESUME LOGIC --------
            if os.path.exists(img_path):
                skipped += 1
                continue

            try:
                output = vlm_pipe(prompt=prompt)
                json_prompt_generate = output.values["json_prompt"]

                negative_prompt = get_default_negative_prompt(json.loads(json_prompt_generate))

                results_generate = pipe(
                    prompt=json_prompt_generate,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    negative_prompt=negative_prompt,
                )
                results_generate.images[0].save(img_path)

                with open(json_path, "w") as f:
                    f.write(json_prompt_generate)

                generated += 1

            except Exception as e:
                errors += 1
                tqdm.write(f"❌ Error on prompt{idx}_{k}: {e}")

        pbar.set_postfix(done=generated, skip=skipped, err=errors)

    print("\n🎯 Done.")
