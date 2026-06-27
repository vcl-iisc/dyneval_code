import sys
import os

# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp
from src.builder import BUILDER
from PIL import Image
from mmengine.config import Config
import argparse
from tqdm import tqdm

# Start index for prompts
START_INDEX = 0
IMAGES_PER_PROMPT = 4


def load_prompts(prompt_file, start_index=START_INDEX):
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts[start_index:], start_index


def get_completed_indices(output_dir):
    """Get indices of already completed prompts (all IMAGES_PER_PROMPT images exist)."""
    completed = set()
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        by_prompt = {}
        for f in files:
            if f.endswith(".png") and f.startswith("prompt"):
                try:
                    base = f.replace(".png", "")
                    pidx, imgidx = base.split("_")
                    pidx = int(pidx.replace("prompt", ""))
                    by_prompt.setdefault(pidx, set()).add(int(imgidx))
                except Exception:
                    pass

        for k, v in by_prompt.items():
            if len(v) >= IMAGES_PER_PROMPT:
                completed.add(k)
    return completed


def generate_images(model, prompt, args, num_images=IMAGES_PER_PROMPT):
    """Generate multiple 512x512 images for the given prompt."""
    formatted_prompt = f"Generate an image: {prompt}"
    class_info = model.prepare_text_conditions(formatted_prompt, args.cfg_prompt)

    input_ids = class_info['input_ids']
    attention_mask = class_info['attention_mask']

    assert len(input_ids) == 2  # conditional + unconditional

    bsz = num_images

    if args.cfg != 1.0:
        input_ids = torch.cat([
            input_ids[:1].expand(bsz, -1),
            input_ids[1:].expand(bsz, -1),
        ])
        attention_mask = torch.cat([
            attention_mask[:1].expand(bsz, -1),
            attention_mask[1:].expand(bsz, -1),
        ])
    else:
        input_ids = input_ids[:1].expand(bsz, -1)
        attention_mask = attention_mask[:1].expand(bsz, -1)

    m = n = 512 // 16

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        samples = model.sample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_iter=args.num_iter,
            cfg=args.cfg,
            cfg_schedule=args.cfg_schedule,
            temperature=args.temperature,
            progress=False,
            image_shape=(m, n)
        )

    images = []
    for i in range(num_images):
        img = samples[i].permute(1, 2, 0)
        img = torch.clamp(127.5 * img + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
        images.append(Image.fromarray(img))

    return images


def worker_process(gpu_id, prompts_with_indices, args, output_dir, total_prompts):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"[GPU {gpu_id}] Loading model...", flush=True)
    config = Config.fromfile(args.config)
    model = BUILDER.build(config.model).eval().to(device)
    model = model.to(model.dtype)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    info = model.load_state_dict(checkpoint, strict=False)
    print(f"[GPU {gpu_id}] Model loaded. Missing keys: {len(info.missing_keys)}, Unexpected keys: {len(info.unexpected_keys)}", flush=True)

    pbar = tqdm(prompts_with_indices, desc=f"GPU {gpu_id}", position=gpu_id, leave=True)

    completed_count = 0
    skipped_count = 0
    error_count = 0

    for item in pbar:
        # 🔒 Robust unpacking (prevents "too many values to unpack")
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            error_count += 1
            print(f"[GPU {gpu_id}] Malformed prompt item: {item}", flush=True)
            continue

        idx, prompt = item[0], item[1]

        out_paths = [os.path.join(output_dir, f"prompt{idx}_{k}.png") for k in range(IMAGES_PER_PROMPT)]

        if all(os.path.exists(p) for p in out_paths):
            skipped_count += 1
            pbar.set_postfix({'done': completed_count, 'skip': skipped_count, 'err': error_count})
            continue

        try:
            images = generate_images(model, prompt, args, IMAGES_PER_PROMPT)
            for k, img in enumerate(images):
                img.save(os.path.join(output_dir, f"prompt{idx}_{k}.png"))

            completed_count += 1
            pbar.set_postfix({
                'done': completed_count,
                'skip': skipped_count,
                'err': error_count,
                'img': f"prompt{idx}_*"
            })

        except Exception as e:
            error_count += 1
            print(f"[GPU {gpu_id}] Error on prompt {idx}: {e}", flush=True)

    print(f"[GPU {gpu_id}] Finished. Done={completed_count}, Skipped={skipped_count}, Errors={error_count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config file path.')
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompts_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cfg_prompt", type=str, default='Generate an image.')
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--cfg_schedule', type=str, default='constant')
    parser.add_argument('--num_iter', type=int, default=8)
    parser.add_argument('--num_gpus', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    prompts, start_idx = load_prompts(args.prompts_file, START_INDEX)
    completed = get_completed_indices(args.output_dir)

    prompts_with_indices = [(start_idx + i, p) for i, p in enumerate(prompts) if (start_idx + i) not in completed]

    if len(prompts_with_indices) == 0:
        print("✅ All prompts already processed.")
        return

    num_available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus or num_available_gpus
    num_gpus = min(num_gpus, num_available_gpus, len(prompts_with_indices))

    mp.set_start_method('spawn', force=True)

    processes = []
    buckets = [[] for _ in range(num_gpus)]
    for i, item in enumerate(prompts_with_indices):
        buckets[i % num_gpus].append(item)

    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process, args=(gpu_id, buckets[gpu_id], args, args.output_dir, len(prompts)))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
