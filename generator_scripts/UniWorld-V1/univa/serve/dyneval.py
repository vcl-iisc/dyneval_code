# coding=utf-8
import sys
import os

# Add the project root to path (handles running from different directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(script_dir))  # Also add univa directory

from transformers import AutoProcessor
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from univa.utils.flux_pipeline import FluxPipeline
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from qwen_vl_utils import process_vision_info

import torch
from PIL import Image
from transformers import set_seed
from torch import nn
import argparse
import torch.multiprocessing as mp
from pathlib import Path

# =====================================================
# HELPERS
# =====================================================
def get_completed_indices(output_dir, images_per_prompt):
    completed = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith(".png") and filename.startswith("prompt"):
                try:
                    base = filename.replace(".png", "")
                    p, i = base.replace("prompt", "").split("_")
                    completed.add((int(p), int(i)))
                except Exception:
                    pass
    return completed


def load_prompts_file(prompts_file):
    with open(prompts_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines


# =====================================================
# FLASH-ATTENTION SAFE MODEL LOADING
# =====================================================
def load_main_model_and_processor(model_path, device, min_pixels=448*448, max_pixels=448*448):
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(device).eval()

    task_head = nn.Sequential(
        nn.Linear(3584, 10240),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(10240, 2)
    ).to(device)
    task_head.load_state_dict(torch.load(os.path.join(model_path, "task_head_final.pt"), map_location=device))
    task_head.eval()

    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    return model, task_head, processor


def load_pipe(denoiser, flux_path, device):
    denoiser = denoiser.to(device).eval()
    pipe = FluxPipeline.from_pretrained(flux_path, transformer=denoiser, torch_dtype=torch.bfloat16).to(device)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    return pipe, tokenizers, text_encoders


def generate_single_image(prompt, model, processor, pipe, tokenizers, text_encoders,
                          device, height, width, num_inference_steps, guidance_scale,
                          no_joint_with_t5, seed):

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    chat_text = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])

    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        lvlm_embeds = model(
            inputs.input_ids,
            pixel_values=getattr(inputs, "pixel_values", None),
            attention_mask=inputs.attention_mask,
            image_grid_thw=getattr(inputs, "image_grid_thw", None),
            siglip_hidden_states=None,
            output_type="denoise_embeds",
        )

    t5_prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders, tokenizers,
        prompt if not no_joint_with_t5 else "",
        256, device, 1
    )

    input_embeds = lvlm_embeds
    if not no_joint_with_t5:
        input_embeds = torch.cat([t5_prompt_embeds, input_embeds], dim=1)

    image = pipe(
        prompt_embeds=input_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=torch.Generator(device=device).manual_seed(seed),
    ).images[0]

    return image


def worker_process(gpu_id, tasks, args):
    device = torch.device(f"cuda:{gpu_id}")
    set_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    model, task_head, processor = load_main_model_and_processor(args.model_path, device)
    pipe, tokenizers, text_encoders = load_pipe(model.denoise_tower.denoiser, args.flux_path, device)

    for prompt_idx, image_idx, prompt in tasks:
        output_path = os.path.join(args.output_dir, f"prompt{prompt_idx}_{image_idx}.png")

        # -------- RESUME LOGIC --------
        if os.path.exists(output_path):
            continue

        image_seed = args.seed + prompt_idx * 1000 + image_idx

        img = generate_single_image(
            prompt, model, processor, pipe, tokenizers, text_encoders,
            device, args.image_size, args.image_size,
            args.num_inference_steps, args.guidance_scale,
            args.no_joint_with_t5, image_seed
        )
        img.save(output_path)
        print(f"[GPU {gpu_id}] Saved {output_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    lines = load_prompts_file(args.prompts_file)
    total_prompts = len(lines)
    print(f"📋 Total prompts loaded: {total_prompts}")

    start_idx = args.start_index
    end_idx = args.end_index if args.end_index != -1 else total_prompts

    if not (0 <= start_idx < total_prompts):
        raise ValueError(f"❌ start_index {start_idx} out of range [0, {total_prompts-1}]")
    if not (start_idx < end_idx <= total_prompts):
        raise ValueError(f"❌ end_index {end_idx} out of range ({start_idx+1} .. {total_prompts})")

    selected = lines[start_idx:end_idx]

    completed = get_completed_indices(args.output_dir, args.images_per_prompt)

    all_tasks = []
    for i, prompt in enumerate(selected, start=start_idx):
        for image_idx in range(args.images_per_prompt):
            if (i, image_idx) not in completed:
                all_tasks.append((i, image_idx, prompt))

    print(f"📋 Selected prompts [{start_idx}:{end_idx}] → {len(selected)}")
    print(f"🚀 Remaining images to generate: {len(all_tasks)}")

    if len(all_tasks) == 0:
        print("🎉 Nothing left to generate.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"🖥️ Found {num_gpus} GPUs")
    if num_gpus == 0:
        raise RuntimeError("❌ No GPUs found!")

    mp.set_start_method("spawn", force=True)

    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        tasks_per_gpu[i % num_gpus].append(task)

    procs = []
    for gpu_id in range(num_gpus):
        if tasks_per_gpu[gpu_id]:
            p = mp.Process(target=worker_process, args=(gpu_id, tasks_per_gpu[gpu_id], args))
            p.start()
            procs.append(p)

    for p in procs:
        p.join()

    print("✅ Generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str, required=True, help="Single txt file, one prompt per line")
    parser.add_argument("--output_dir", type=str, required=True)

    # ---- Keep current defaults ----
    parser.add_argument("--model_path", type=str, default="/storage/users/anirban/arunabhsingh25/UniWorld-V1-model")
    parser.add_argument("--flux_path", type=str, default="/storage/users/anirban/dheerajbaiju/FLUX.1-dev-model")
    parser.add_argument("--siglip_path", type=str, default="/storage/users/anirban/arunabhsingh25/siglip-path")

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--no_joint_with_t5", action="store_true")

    # ---- NEW CLI ----
    parser.add_argument("--start_index", type=int, default=12001)
    parser.add_argument("--end_index", type=int, default=14000)
    parser.add_argument("--images_per_prompt", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Optional perf boost
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    main(args)
