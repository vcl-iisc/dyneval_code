"""
Emu3 DPG-BENCH Inference Script
- Optimized for fast generation with multi-GPU support
- Resume capability (skips already generated images)
- Reads prompts from ELLA/dpg_bench/prompts directory
- Generates 4 images per prompt
"""
import os
import time
import glob
import torch
import torch.multiprocessing as mp
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor

# Speed optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Configuration
PROMPTS_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark/prompts/dpg_bench"
OUTPUT_DIR = "DPG-BENCH-EMU3-RESULTS"
EMU_HUB = "Emu3-Gen-model"
VQ_HUB = "Emu3-VisionTokenizer-model"
IMAGES_PER_PROMPT = 4

# Generation settings
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
CLASSIFIER_FREE_GUIDANCE = 3.0
IMAGE_RATIO = "1:1"
IMAGE_AREA = 256 * 256  # 65536 for 256x256, faster than default


def load_prompts(prompts_dir):
    """Load prompts from directory containing .txt files."""
    prompt_files = sorted(glob.glob(os.path.join(prompts_dir, "*.txt")))
    prompts = []
    for filepath in prompt_files:
        filename = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, 'r') as f:
            prompt_text = f.read().strip()
        prompts.append((filename, prompt_text))
    return prompts


def get_pending_work(output_dir, prompts):
    """Get list of (prompt_idx, image_idx) tuples that haven't been generated yet."""
    os.makedirs(output_dir, exist_ok=True)
    pending = []
    for prompt_idx, (filename, _) in enumerate(prompts):
        for img_idx in range(IMAGES_PER_PROMPT):
            img_path = os.path.join(output_dir, f"{filename}_{img_idx}.png")
            if not os.path.exists(img_path):
                pending.append((prompt_idx, img_idx))
    return pending


def worker(rank, world_size, all_prompts, pending_work):
    """Worker function for each GPU."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    
    # Speed optimizations
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    
    # Split pending work for this GPU
    gpu_work = [work for i, work in enumerate(pending_work) if i % world_size == rank]
    
    if not gpu_work:
        print(f"[GPU {rank}] No work assigned. Exiting.")
        return
    
    print(f"[GPU {rank}] Assigned {len(gpu_work)} images to generate")

    # Load model
    print(f"[GPU {rank}] Loading Emu3 model...")
    model = AutoModelForCausalLM.from_pretrained(
        EMU_HUB,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    
    # Load tokenizer and processors
    tokenizer = AutoTokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(VQ_HUB, device_map=device, trust_remote_code=True).eval()
    
    # Import Emu3Processor
    from emu3.mllm.processing_emu3 import Emu3Processor
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    
    print(f"[GPU {rank}] Model loaded successfully")

    # Prepare generation config - reduced max_new_tokens for 256x256
    GENERATION_CONFIG = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=8192,  # Reduced from 40960 - sufficient for 256x256
        do_sample=True,
        top_k=2048,
    )

    # Pre-compute negative prompt inputs (same for all prompts)
    kwargs = dict(
        mode='G',
        ratio=IMAGE_RATIO,
        image_area=IMAGE_AREA,
        return_tensors="pt",
        padding="longest",
    )
    neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

    # Process each work item
    total_generated = 0
    total_time = 0

    for prompt_idx, img_idx in gpu_work:
        filename, prompt_text = all_prompts[prompt_idx]
        full_prompt = prompt_text + POSITIVE_PROMPT
        
        print(f"[GPU {rank}] Processing {filename}_{img_idx}: {prompt_text[:50]}...")
        
        t1 = time.time()
        
        # Prepare inputs
        pos_inputs = processor(text=full_prompt, **kwargs)
        
        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = processor.build_prefix_constrained_fn(h, w)
        
        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                CLASSIFIER_FREE_GUIDANCE,
                model,
                unconditional_ids=neg_inputs.input_ids.to(device),
            ),
            PrefixConstrainedLogitsProcessor(
                constrained_fn,
                num_beams=1,
            ),
        ])

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                pos_inputs.input_ids.to(device),
                GENERATION_CONFIG,
                logits_processor=logits_processor,
                attention_mask=pos_inputs.attention_mask.to(device),
            )

        # Decode and save
        mm_list = processor.decode(outputs[0])
        for im in mm_list:
            if isinstance(im, Image.Image):
                img_path = os.path.join(OUTPUT_DIR, f"{filename}_{img_idx}.png")
                im.save(img_path)
                break  # Only save first image
        
        gen_time = time.time() - t1
        total_time += gen_time
        total_generated += 1
        
        print(f"[GPU {rank}] Saved {filename}_{img_idx}.png | Time: {gen_time:.2f}s | Progress: {total_generated}/{len(gpu_work)}")

    print(f"[GPU {rank}] COMPLETE: Generated {total_generated} images in {total_time:.2f}s")


def main():
    # Detect number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")

    # Load prompts
    all_prompts = load_prompts(PROMPTS_DIR)
    total_prompts = len(all_prompts)
    print(f"Loaded {total_prompts} prompts from {PROMPTS_DIR}")

    # Get pending work
    pending_work = get_pending_work(OUTPUT_DIR, all_prompts)
    total_images = total_prompts * IMAGES_PER_PROMPT
    print(f"Pending images to generate: {len(pending_work)} / {total_images}")

    if not pending_work:
        print("All images already generated. Exiting.")
        return

    # Launch multi-GPU workers
    if num_gpus > 1:
        print(f"Starting multi-GPU generation with {num_gpus} GPUs...")
        mp.spawn(
            worker,
            args=(num_gpus, all_prompts, pending_work),
            nprocs=num_gpus,
            join=True
        )
    else:
        print("Single GPU mode")
        worker(0, 1, all_prompts, pending_work)

    print(f"\n=== ALL COMPLETE ===")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
