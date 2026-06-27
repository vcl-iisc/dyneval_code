# coding=utf-8

from pathlib import Path
import glob
import os
import argparse
import torch
from diffusers import Flux2KleinPipeline
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# ARGPARSE
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_path", type=str, required=True, help="Directory of .txt files OR single .txt file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default='/home/anirban/arunabhsingh25/models/FLUX.2-KLEIN-MODEL')

    parser.add_argument("--start_index", type=int, default=20001)
    parser.add_argument("--end_index", type=int, default=22000)

    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--images_per_prompt", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    return parser.parse_args()


def load_prompts_file_or_dir(prompts_path):
    all_prompts = []

    if os.path.isdir(prompts_path):
        files = sorted(glob.glob(os.path.join(prompts_path, "*.txt")))
        if len(files) == 0:
            raise RuntimeError(f"❌ No .txt files found in directory: {prompts_path}")
        for p_file in files:
            stem = Path(p_file).stem
            with open(p_file, "r") as f:
                text = f.read().strip()
            all_prompts.append((stem, text))

    elif os.path.isfile(prompts_path):
        with open(prompts_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, text in enumerate(lines):
            stem = f"prompt{i}"
            all_prompts.append((stem, text))

    else:
        raise RuntimeError(f"❌ prompts_path does not exist: {prompts_path}")

    return all_prompts


# =====================================================
# GPU WORKER
# =====================================================
def generate_on_gpu(gpu_id, prompts_subset, args):
    torch.cuda.set_device(gpu_id)

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    pipe.enable_model_cpu_offload(gpu_id=gpu_id)

    pipe.set_progress_bar_config(disable=True)
    print(f"🖥️ GPU {gpu_id}: Model loaded")

    for stem, prompt in tqdm(prompts_subset, desc=f"GPU {gpu_id}", unit="prompt", position=gpu_id):
        for k in range(args.images_per_prompt):
            out_path = os.path.join(args.output_dir, f"{stem}_{k}.png")

            # -------- RESUME LOGIC --------
            if os.path.exists(out_path):
                continue

            try:
                result = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                )
                result.images[0].save(out_path)

            except Exception as e:
                import traceback
                traceback.print_exc()
                tqdm.write(f"❌ GPU {gpu_id} Error {stem}_{k}: {e}")


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)

    # =====================================================
    # LOAD PROMPTS
    # =====================================================
    all_prompts = load_prompts_file_or_dir(args.prompts_path)
    total_prompts = len(all_prompts)
    print(f"📋 Total prompts loaded: {total_prompts}")

    start_idx = args.start_index
    end_idx = args.end_index if args.end_index != -1 else total_prompts

    if not (0 <= start_idx < total_prompts):
        raise ValueError(f"❌ start_index {start_idx} out of range [0, {total_prompts-1}]")
    if not (start_idx < end_idx <= total_prompts):
        raise ValueError(f"❌ end_index {end_idx} out of range ({start_idx+1} .. {total_prompts})")

    selected = all_prompts[start_idx:end_idx]

    # =====================================================
    # RESUME LOGIC (filter fully completed prompts)
    # =====================================================
    filtered = []
    already_done = 0

    for stem, prompt in selected:
        done = True
        for k in range(args.images_per_prompt):
            if not os.path.exists(os.path.join(args.output_dir, f"{stem}_{k}.png")):
                done = False
                break
        if done:
            already_done += 1
        else:
            filtered.append((stem, prompt))

    print(f"📋 Selected prompts [{start_idx}:{end_idx}] → {len(selected)}")
    print(f"✅ Already completed: {already_done}")
    print(f"🚀 Remaining to generate: {len(filtered)}")
    print(f"📐 Image size: {args.width}x{args.height}")
    print(f"🔧 Inference steps: {args.num_steps}")
    print(f"🔁 Images per prompt: {args.images_per_prompt}")

    num_gpus = torch.cuda.device_count()
    print(f"🖥️ Found {num_gpus} GPUs")
    if num_gpus == 0:
        raise RuntimeError("❌ No GPUs found!")

    if len(filtered) == 0:
        print("🎉 Nothing left to generate.")
        exit(0)

    if num_gpus == 1:
        generate_on_gpu(0, filtered, args)
    else:
        mp.set_start_method("spawn", force=True)
        chunks = [filtered[i::num_gpus] for i in range(num_gpus)]

        procs = []
        for gpu_id in range(num_gpus):
            p = mp.Process(target=generate_on_gpu, args=(gpu_id, chunks[gpu_id], args))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

    print("\n🎯 All prompts processed successfully!")
