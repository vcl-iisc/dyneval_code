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
import os
import argparse
import torch.multiprocessing as mp
from pathlib import Path


# Constants
PROMPT_FILE = "/home/anirban/dheerajbaiju/dheeraj/instance-seg/evalmuse-test-prompts.txt"
OUTPUT_DIR = "UNIWORLD-TEST-IMAGES"
START_INDEX = 0  # Start from the first prompt
IMAGES_PER_PROMPT = 1  # Generate 4 images per prompt
IMAGE_SIZE = 512
SEED = 42


def get_completed_indices(output_dir):
    """Get set of completed (prompt_idx, image_idx) tuples by checking existing images."""
    completed = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.png'):
                try:
                    # Extract indices from filename (format: 0000_0.png)
                    base = filename.replace('.png', '')
                    parts = base.split('_')
                    if len(parts) == 2:
                        prompt_idx = int(parts[0])
                        image_idx = int(parts[1])
                        completed.add((prompt_idx, image_idx))
                except ValueError:
                    continue
    return completed


def load_prompts(prompt_file, start_index=START_INDEX):
    """Load prompts from file, starting from start_index till end."""
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts[start_index:], start_index


def load_main_model_and_processor(
    model_path, 
    device, 
    min_pixels=448*448,
    max_pixels=448*448
):
    """Load model and processor."""
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)
    
    task_head = nn.Sequential(
        nn.Linear(3584, 10240),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(10240, 2)
    ).to(device)
    task_head.load_state_dict(torch.load(os.path.join(model_path, 'task_head_final.pt')))
    task_head.eval()

    processor = AutoProcessor.from_pretrained(
        model_path, 
        min_pixels=min_pixels, max_pixels=max_pixels
    )
    return model, task_head, processor


def load_pipe(denoiser, flux_path, device):
    """Load the Flux pipeline."""
    pipe = FluxPipeline.from_pretrained(
        flux_path,
        transformer=denoiser,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [
        pipe.text_encoder,
        pipe.text_encoder_2,
    ]
    return pipe, tokenizers, text_encoders


def load_siglip_and_processor(siglip_path, device):
    """Load SigLIP model and processor."""
    siglip_processor, siglip_model = None, None
    if siglip_path:
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
        siglip_model = SiglipVisionModel.from_pretrained(
            siglip_path, 
            torch_dtype=torch.bfloat16, 
        ).to(device)
    return siglip_processor, siglip_model


def generate_single_image(
    prompt,
    model,
    processor,
    pipe,
    tokenizers,
    text_encoders,
    device,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    no_joint_with_t5,
    seed
):
    """Generate a single image from a prompt."""
    # Build conversation with just text prompt
    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    
    # Prepare inputs for model
    chat_text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    chat_text = '<|im_end|>\n'.join(chat_text.split('<|im_end|>\n')[1:])  # drop system
    
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    
    # Generate embeddings with speed optimizations
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        lvlm_embeds = model(
            inputs.input_ids,
            pixel_values=getattr(inputs, 'pixel_values', None),
            attention_mask=inputs.attention_mask, 
            image_grid_thw=getattr(inputs, 'image_grid_thw', None),
            siglip_hidden_states=None, 
            output_type="denoise_embeds",
        )
    
    assert lvlm_embeds.shape[0] == 1
    input_embeds = lvlm_embeds
    
    t5_prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        prompt if not no_joint_with_t5 else '',
        256,
        device,
        1,
    )
    
    if not no_joint_with_t5:
        input_embeds = torch.concat([t5_prompt_embeds, input_embeds], dim=1)
    
    # Generate image
    output_image = pipe(
        prompt_embeds=input_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale, 
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]
    
    return output_image


def worker_process(gpu_id, tasks, args, output_dir):
    """Worker process for each GPU.
    
    Args:
        tasks: List of (prompt_idx, image_idx, prompt) tuples
    """
    device = torch.device(f"cuda:{gpu_id}")
    
    # Set seed for reproducibility
    set_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
    torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner for faster kernels
    
    print(f"[GPU {gpu_id}] Loading models...")
    
    # Load models
    model, task_head, processor = load_main_model_and_processor(
        args.model_path, 
        device,  
    )
    
    pipe, tokenizers, text_encoders = load_pipe(
        model.denoise_tower.denoiser, args.flux_path, device
    )
    
    print(f"[GPU {gpu_id}] Models loaded. Processing {len(tasks)} image tasks...")
    
    for prompt_idx, image_idx, prompt in tasks:
        output_path = os.path.join(output_dir, f"{prompt_idx:04d}_{image_idx}.png")
        
        # Skip if already generated (resume capability)
        if os.path.exists(output_path):
            print(f"[GPU {gpu_id}] Skipping {prompt_idx:04d}_{image_idx} (already exists)")
            continue
        
        try:
            print(f"[GPU {gpu_id}] Generating image {prompt_idx:04d}_{image_idx}...")
            
            # Use different seed for each image variation
            image_seed = SEED + image_idx
            
            output_image = generate_single_image(
                prompt=prompt,
                model=model,
                processor=processor,
                pipe=pipe,
                tokenizers=tokenizers,
                text_encoders=text_encoders,
                device=device,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                no_joint_with_t5=args.no_joint_with_t5,
                seed=image_seed
            )
            
            output_image.save(output_path)
            print(f"[GPU {gpu_id}] Saved {output_path}")
            
        except Exception as e:
            print(f"[GPU {gpu_id}] Error generating image {prompt_idx:04d}_{image_idx}: {e}")
            continue


def main(args):
    # Create output directory
    output_dir = os.path.join(os.path.dirname(PROMPT_FILE), OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load prompts from START_INDEX to end
    prompts, start_idx = load_prompts(PROMPT_FILE)
    total_prompts = len(prompts)
    total_images = total_prompts * IMAGES_PER_PROMPT
    print(f"Loaded {total_prompts} prompts (from index {start_idx} to end)")
    print(f"Will generate {IMAGES_PER_PROMPT} images per prompt = {total_images} total images")
    
    # Get already completed (prompt_idx, image_idx) tuples for resume capability
    completed = get_completed_indices(output_dir)
    print(f"Found {len(completed)} already completed images")
    
    # Create list of all tasks: (prompt_idx, image_idx, prompt)
    all_tasks = []
    for i in range(len(prompts)):
        prompt_idx = start_idx + i
        for image_idx in range(IMAGES_PER_PROMPT):
            if (prompt_idx, image_idx) not in completed:
                all_tasks.append((prompt_idx, image_idx, prompts[i]))
    
    print(f"Remaining images to generate: {len(all_tasks)}")
    
    if len(all_tasks) == 0:
        print("All images already generated. Exiting.")
        return
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        return
    
    if num_gpus == 1:
        # Single GPU mode
        print("Running in single GPU mode...")
        worker_process(0, all_tasks, args, output_dir)
    else:
        # Multi-GPU mode using multiprocessing
        print(f"Running in multi-GPU mode with {num_gpus} GPUs...")
        
        # Distribute tasks across GPUs
        tasks_per_gpu = [[] for _ in range(num_gpus)]
        for i, task in enumerate(all_tasks):
            gpu_id = i % num_gpus
            tasks_per_gpu[gpu_id].append(task)
        
        # Print distribution
        for gpu_id in range(num_gpus):
            print(f"GPU {gpu_id}: {len(tasks_per_gpu[gpu_id])} images")
        
        # Spawn processes
        mp.set_start_method('spawn', force=True)
        processes = []
        
        for gpu_id in range(num_gpus):
            if len(tasks_per_gpu[gpu_id]) > 0:
                p = mp.Process(
                    target=worker_process,
                    args=(gpu_id, tasks_per_gpu[gpu_id], args, output_dir)
                )
                p.start()
                processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
    
    print("Generation complete!")
    
    # Print final statistics
    final_completed = get_completed_indices(output_dir)
    print(f"Total images generated: {len(final_completed)}/{total_images}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="X-Omni Evalmuse Image Generation")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the main model")
    parser.add_argument("--flux_path", type=str, required=True,
                        help="Path to the Flux model")
    parser.add_argument("--siglip_path", type=str, default=None,
                        help="Path to SigLIP model (optional)")
    parser.add_argument("--num_inference_steps", type=int, default=17,
                        help="Number of inference steps (lower = faster)")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Guidance scale for generation")
    parser.add_argument("--no_joint_with_t5", action="store_true",
                        help="Disable joint T5 encoding")
    
    args = parser.parse_args()
    main(args)
