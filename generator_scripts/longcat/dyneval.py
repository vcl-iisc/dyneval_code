# coding=utf-8
import os
import argparse
import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm
import multiprocessing as mp

# =====================================================
# ARGPARSE
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/home/anirban/yashwanthm/dheeraj/LongCat/LongCat-Image-model")

    parser.add_argument("--start_index", type=int, default=14001)
    parser.add_argument("--end_index", type=int, default=16000)

    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--images_per_prompt", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    return parser.parse_args()


# =====================================================
# GPU WORKER
# =====================================================
def generate_on_gpu(gpu_id, prompts_subset, args):
    torch.cuda.set_device(gpu_id)

    pipe = DiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(f"cuda:{gpu_id}")

    pipe.set_progress_bar_config(disable=True)
    print(f"🖥️ GPU {gpu_id}: Model loaded")

    for idx, prompt in tqdm(prompts_subset, desc=f"GPU {gpu_id}", position=gpu_id):
        for k in range(args.images_per_prompt):
            out_path = os.path.join(args.output_dir, f"prompt{idx}_{k}.png")

            # -------- RESUME LOGIC --------
            if os.path.exists(out_path):
                continue

            try:
                image = pipe(
                    prompt=prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                ).images[0]

                image.save(out_path)

            except Exception as e:
                tqdm.write(f"❌ GPU {gpu_id} Error prompt{idx}_{k}: {e}")


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

    selected = []
    for i, prompt in enumerate(lines[start_idx:end_idx], start=start_idx):
        selected.append((i, prompt))

    # =====================================================
    # RESUME LOGIC (filter fully completed prompts)
    # =====================================================
    filtered = []
    already_done = 0

    for idx, prompt in selected:
        done = True
        for k in range(args.images_per_prompt):
            if not os.path.exists(os.path.join(args.output_dir, f"prompt{idx}_{k}.png")):
                done = False
                break
        if done:
            already_done += 1
        else:
            filtered.append((idx, prompt))

    print(f"📋 Selected prompts [{start_idx}:{end_idx}] → {len(selected)}")
    print(f"✅ Already completed: {already_done}")
    print(f"🚀 Remaining to generate: {len(filtered)}")

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

    print("🎯 Done!")
