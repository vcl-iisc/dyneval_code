import sys
import os

# Add the project root to path (handles running from different directories)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(script_dir))  # Also add univa directory

from transformers import AutoTokenizer, AutoProcessor
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import UnivaQwen2p5VLForConditionalGeneration
from transformers import SiglipImageProcessor, SiglipVisionModel
from univa.utils.flux_pipeline import FluxPipeline
from univa.utils.get_ocr import get_ocr_result
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from qwen_vl_utils import process_vision_info
from univa.utils.anyres_util import dynamic_resize

import torch
from PIL import Image
from transformers import set_seed
from torch import nn
import argparse
import torch.multiprocessing as mp
from pathlib import Path

# =====================================================
# CONSTANTS
# =====================================================
PROMPT_FILE = "/home/anirban/dheerajbaiju/dheeraj/dyneval-models/geneval-final-prompt-order.txt"
OUTPUT_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/uniworld-geneval-results"
START_INDEX = 0
IMAGES_PER_PROMPT = 4
IMAGE_SIZE = 1024
SEED = 42


def get_completed_indices(output_dir):
    completed = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith(".png") and filename.startswith("prompt"):
                try:
                    base = filename.replace(".png", "")
                    p, i = base.replace("prompt", "").split("_")
                    completed.add((int(p), int(i)))
                except:
                    pass
    return completed


def load_prompts(prompt_file, start_index=START_INDEX):
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts[start_index:], start_index


# =====================================================
# FLASH-ATTENTION SAFE MODEL LOADING
# =====================================================
def load_main_model_and_processor(model_path, device, min_pixels=448*448, max_pixels=448*448):
    # Load on CPU first
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,   # do NOT auto place
    )

    # Move to GPU before use
    model = model.to(device).eval()

    task_head = nn.Sequential(
        nn.Linear(3584, 10240),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(10240, 2)
    ).to(device)

    task_head.load_state_dict(
        torch.load(os.path.join(model_path, "task_head_final.pt"), map_location=device)
    )
    task_head.eval()

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )

    return model, task_head, processor


def load_pipe(denoiser, flux_path, device):
    # Make sure denoiser is already on GPU
    denoiser = denoiser.to(device).eval()

    pipe = FluxPipeline.from_pretrained(
        flux_path,
        transformer=denoiser,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)

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


def worker_process(gpu_id, tasks, args, output_dir):
    device = torch.device(f"cuda:{gpu_id}")
    set_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True

    model, task_head, processor = load_main_model_and_processor(args.model_path, device)
    pipe, tokenizers, text_encoders = load_pipe(model.denoise_tower.denoiser, args.flux_path, device)

    for prompt_idx, image_idx, prompt in tasks:
        output_path = os.path.join(output_dir, f"prompt{prompt_idx}_{image_idx}.png")

        if os.path.exists(output_path):
            continue

        image_seed = SEED + image_idx

        img = generate_single_image(
            prompt, model, processor, pipe, tokenizers, text_encoders,
            device, IMAGE_SIZE, IMAGE_SIZE,
            args.num_inference_steps, args.guidance_scale,
            args.no_joint_with_t5, image_seed
        )

        img.save(output_path)
        print(f"[GPU {gpu_id}] Saved {output_path}")


def main(args):
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    prompts, start_idx = load_prompts(PROMPT_FILE)
    completed = get_completed_indices(output_dir)

    all_tasks = []
    for i, prompt in enumerate(prompts):
        prompt_idx = start_idx + i
        for image_idx in range(IMAGES_PER_PROMPT):
            if (prompt_idx, image_idx) not in completed:
                all_tasks.append((prompt_idx, image_idx, prompt))

    num_gpus = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)

    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        tasks_per_gpu[i % num_gpus].append(task)

    procs = []
    for gpu_id in range(num_gpus):
        if tasks_per_gpu[gpu_id]:
            p = mp.Process(target=worker_process, args=(gpu_id, tasks_per_gpu[gpu_id], args, output_dir))
            p.start()
            procs.append(p)

    for p in procs:
        p.join()

    print("✅ Generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/storage/users/anirban/arunabhsingh25/UniWorld-V1-model")
    parser.add_argument("--flux_path", type=str, default="/storage/users/anirban/dheerajbaiju/FLUX.1-dev-model")
    parser.add_argument("--siglip_path", type=str, default="/storage/users/anirban/arunabhsingh25/siglip-path")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--no_joint_with_t5", action="store_true")
    args = parser.parse_args()

    # Optional perf boost
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    main(args)
