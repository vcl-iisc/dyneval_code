import os
import json
import time
import torch
import random
import numpy as np
import argparse
import multiprocessing as mp
from datetime import datetime
from PIL import Image
from tqdm import tqdm

from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
    init_empty_weights,
)
from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer


# Number of images to generate per prompt
IMAGES_PER_PROMPT = 4


def seed_everything(seed: int):
    """Sets the seed for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # OPTIMIZATION: Set to False for speed, True for reproducibility
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # OPTIMIZATION: Enable cudnn auto-tuner


def reset_taylorseer_state(model):
    """Reset TaylorSeer state to prevent corruption between generations."""
    try:
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
            lm_model = model.language_model.model
            # Reset TaylorSeer flags
            if hasattr(lm_model, 'enable_taylorseer'):
                lm_model.enable_taylorseer = False
            if hasattr(lm_model, 'cache_dic'):
                lm_model.cache_dic = None
            if hasattr(lm_model, 'current'):
                lm_model.current = None
            # Reset layer-level TaylorSeer state
            if hasattr(lm_model, 'layers'):
                for layer in lm_model.layers:
                    if hasattr(layer, 'enable_taylorseer'):
                        layer.enable_taylorseer = False
                    if hasattr(layer, 'current'):
                        layer.current = None
    except Exception:
        pass  # Ignore cleanup errors


def generate_image(inferencer, prompt, seed, inference_hyper):
    """Generate a single image with optimized settings."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Reset TaylorSeer state before generation to prevent state corruption
    reset_taylorseer_state(inferencer.model)
    
    try:
        # Use autocast with bfloat16 to match model dtype and prevent type mismatches
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = inferencer(text=prompt, **inference_hyper)
            if isinstance(result, dict) and "image" in result:
                return result["image"]
            elif isinstance(result, Image.Image):
                return result
            elif (
                isinstance(result, list)
                and isinstance(result[0], dict)
                and "image" in result[0]
            ):
                return result[0]["image"]
            raise RuntimeError("Unexpected inferencer output format.")
    finally:
        # Always reset state after generation attempt (success or failure)
        reset_taylorseer_state(inferencer.model)
        # Clear any residual tensors
        torch.cuda.empty_cache()


def load_model_components(model_path):
    """Load model configuration components."""
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(model_path, "vit_config.json")
    )
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(
        local_path=os.path.join(model_path, "ae.safetensors")
    )

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    return llm_config, vit_config, vae_model, vae_config, config


def load_prompts_from_directory(prompts_dir):
    """Load prompts from text files in the specified directory."""
    prompts = []
    
    txt_files = [f for f in os.listdir(prompts_dir) if f.endswith('.txt')]
    txt_files.sort()
    
    for txt_file in txt_files:
        file_path = os.path.join(prompts_dir, txt_file)
        prompt_name = os.path.splitext(txt_file)[0]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        if prompt_text:
            prompts.append({
                'name': prompt_name,
                'prompt': prompt_text
            })
    
    return prompts


def gpu_worker(gpu_id, task_queue, args, shared_results, total_tasks):
    """Worker function for each GPU process with optimizations."""
    print(f"🚀 GPU {gpu_id}: Loading BAGEL model...", flush=True)

    seed_everything(args.seed + gpu_id)

    try:
        # OPTIMIZATION: Set CUDA device first
        torch.cuda.set_device(gpu_id)
        
        # Load model components
        llm_config, vit_config, vae_model, vae_config, config = load_model_components(
            args.model_path
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
                vit_config, meta=True
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(args.model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        max_memory = {gpu_id: args.max_mem}
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        first_device = f"cuda:{gpu_id}"
        for k in same_device_modules:
            device_map[k] = first_device

        print(f"📦 GPU {gpu_id}: Loading weights...", flush=True)
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(args.model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=False,  # Disable for H200 with lots of VRAM
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()

        # OPTIMIZATION: Try torch.compile if available (PyTorch 2.0+)
        if args.compile and hasattr(torch, 'compile'):
            print(f"⚡ GPU {gpu_id}: Compiling model with torch.compile()...", flush=True)
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as e:
                print(f"⚠️ GPU {gpu_id}: torch.compile failed, continuing without: {e}", flush=True)

        # OPTIMIZATION: Move VAE to GPU and set to eval
        vae_model = vae_model.to(f"cuda:{gpu_id}").eval()

        inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        # OPTIMIZATION: Reduced timesteps + TaylorSeer for faster generation
        inference_hyper = dict(
            cfg_text_scale=args.cfg_text_scale,
            cfg_img_scale=1.0,
            cfg_interval=[0.4, 1.0],
            timestep_shift=3.0,
            num_timesteps=args.num_timesteps,
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
            enable_taylorseer=args.taylorseer,  # Enable TaylorSeer acceleration
            image_shapes=(args.image_size, args.image_size),  # Output image size
        )

        print(f"✅ GPU {gpu_id}: Model loaded successfully.", flush=True)

        # OPTIMIZATION: Warm-up run to compile CUDA kernels
        if args.warmup:
            print(f"🔥 GPU {gpu_id}: Warming up...", flush=True)
            try:
                _ = generate_image(inferencer, "warmup test", 42, inference_hyper)
                torch.cuda.synchronize(gpu_id)
            except:
                pass

        # Process tasks with progress bar
        local_generated = 0
        local_skipped = 0
        local_errors = 0
        start_time = time.time()

        # Create progress bar
        pbar = tqdm(
            total=total_tasks,
            desc=f"GPU {gpu_id}",
            position=gpu_id,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        while not task_queue.empty():
            try:
                item = task_queue.get_nowait()
            except:
                break

            prompt_name = item.get("name")
            prompt_text = item.get("prompt")
            image_index = item.get("image_index")

            if not prompt_name or not prompt_text:
                local_skipped += 1
                pbar.update(1)
                continue

            output_filename = f"{prompt_name}_{image_index}.png"
            save_path = os.path.join(args.outdir, output_filename)

            if os.path.exists(save_path):
                local_skipped += 1
                pbar.update(1)
                pbar.set_postfix({'done': local_generated, 'skip': local_skipped, 'err': local_errors})
                continue

            try:
                seed = args.seed + hash(f"{prompt_name}_{image_index}") % 10000
                img = generate_image(inferencer, prompt_text, seed, inference_hyper)
                
                # Debug: Check if image is valid before saving
                if img is None:
                    print(f"\n⚠️ GPU {gpu_id}: Generated image is None for {prompt_name}_{image_index}", flush=True)
                    local_errors += 1
                    pbar.update(1)
                    continue
                
                # Ensure output directory exists (in worker process)
                os.makedirs(args.outdir, exist_ok=True)
                
                img.save(save_path)
                local_generated += 1
                
                # Verify save worked
                if local_generated == 1:
                    print(f"\n✅ GPU {gpu_id}: First image saved to {save_path}", flush=True)
                
                # Calculate speed
                elapsed = time.time() - start_time
                speed = local_generated / elapsed if elapsed > 0 else 0
                
                pbar.set_postfix({
                    'done': local_generated, 
                    'skip': local_skipped, 
                    'err': local_errors,
                    'img/s': f'{speed:.2f}'
                })

            except Exception as e:
                local_errors += 1
                # Print detailed error for debugging
                print(f"\n❌ GPU {gpu_id}: Error saving {prompt_name}_{image_index}: {type(e).__name__}: {str(e)}", flush=True)
                pbar.set_postfix({'done': local_generated, 'skip': local_skipped, 'err': local_errors})

            pbar.update(1)
            
            # OPTIMIZATION: Clear cache periodically to prevent OOM
            if local_generated % 50 == 0:
                torch.cuda.empty_cache()

        pbar.close()
        
        shared_results[gpu_id] = {
            "generated": local_generated,
            "skipped": local_skipped,
            "errors": local_errors,
        }
        print(
            f"\n🏁 GPU {gpu_id}: Finished! Generated: {local_generated}, Skipped: {local_skipped}, Errors: {local_errors}",
            flush=True
        )

    except Exception as e:
        print(f"❌ GPU {gpu_id}: Failed to initialize: {str(e)}", flush=True)
        shared_results[gpu_id] = {"generated": 0, "skipped": 0, "errors": 0, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Generate images from prompt files using BAGEL-7B-MoT with multi-GPU scaling"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to BAGEL model weights"
    )
    parser.add_argument(
        "--prompts_dir",
        type=str,
        default="/home/anirban/dheerajbaiju/dheeraj/instance-seg/Bagel/ELLA/dpg_bench/prompts",
        help="Directory containing prompt text files",
    )
    parser.add_argument(
        "--outdir", type=str, default="outputs", help="Output directory for images"
    )
    parser.add_argument(
        "--max_mem",
        type=str,
        default="120GiB",  # H200 has 141GB, use most of it
        help="Max memory per device for device_map",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_timesteps", type=int, default=30, 
        help="Number of diffusion timesteps (lower=faster, 30 is good balance)"
    )
    parser.add_argument(
        "--taylorseer", action="store_true", default=True,
        help="Enable TaylorSeer acceleration for faster generation"
    )
    parser.add_argument(
        "--no_taylorseer", action="store_false", dest="taylorseer",
        help="Disable TaylorSeer acceleration"
    )
    parser.add_argument(
        "--image_size", type=int, default=512,
        help="Output image size (default: 512 for 512x512)"
    )
    parser.add_argument(
        "--cfg_text_scale", type=float, default=4.0,
        help="CFG text scale (lower can be faster)"
    )
    parser.add_argument(
        "--offload_dir",
        type=str,
        default="./offload_cache",
        help="Directory to offload model weights",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Use torch.compile() for faster inference (PyTorch 2.0+)"
    )
    parser.add_argument(
        "--warmup", action="store_true", default=True,
        help="Run warmup inference to compile CUDA kernels"
    )
    parser.add_argument(
        "--no_warmup", action="store_false", dest="warmup",
        help="Skip warmup inference"
    )
    args = parser.parse_args()

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        print("❌ No GPU detected!")
        return
    print(f"✅ Using {num_gpus} GPU(s) for generation")

    mp.set_start_method("spawn", force=True)

    # Load prompts
    print(f"Loading prompts from directory: {args.prompts_dir}")
    prompts = load_prompts_from_directory(args.prompts_dir)
    print(f"✅ Loaded {len(prompts)} prompts from directory.")

    os.makedirs(args.outdir, exist_ok=True)

    # Resume functionality
    existing_files = set()
    if os.path.exists(args.outdir):
        existing_files = set(os.listdir(args.outdir))

    # Create task queue
    task_queue = mp.Queue()
    total_tasks = 0
    skipped_existing = 0

    for prompt_data in prompts:
        for image_idx in range(IMAGES_PER_PROMPT):
            output_filename = f"{prompt_data['name']}_{image_idx}.png"
            if output_filename in existing_files:
                skipped_existing += 1
                continue
            task_queue.put({
                "name": prompt_data["name"],
                "prompt": prompt_data["prompt"],
                "image_index": image_idx
            })
            total_tasks += 1

    total_possible = len(prompts) * IMAGES_PER_PROMPT
    
    print(f"\n{'='*60}")
    print(f"BAGEL Image Generation - Optimized")
    print(f"{'='*60}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Images per prompt: {IMAGES_PER_PROMPT}")
    print(f"Already completed: {skipped_existing}")
    print(f"Remaining tasks: {total_tasks}")
    print(f"Progress: {skipped_existing}/{total_possible} ({(skipped_existing/total_possible)*100:.1f}%)")
    print(f"{'='*60}")
    print(f"Timesteps: {args.num_timesteps} (lower=faster)")
    print(f"CFG text scale: {args.cfg_text_scale}")
    print(f"torch.compile: {'enabled' if args.compile else 'disabled'}")
    print(f"{'='*60}\n")

    if total_tasks == 0:
        print("✅ All images already generated! Nothing to do.")
        return

    # Shared results
    manager = mp.Manager()
    shared_results = manager.dict()

    # Calculate tasks per GPU for progress bars
    tasks_per_gpu = total_tasks // num_gpus + 1

    # Start workers
    start_time = time.time()
    procs = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker, 
            args=(gpu_id, task_queue, args, shared_results, tasks_per_gpu)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Results
    elapsed = time.time() - start_time
    total_generated = sum(shared_results.get(i, {}).get("generated", 0) for i in range(num_gpus))
    total_skipped = sum(shared_results.get(i, {}).get("skipped", 0) for i in range(num_gpus))
    total_errors = sum(shared_results.get(i, {}).get("errors", 0) for i in range(num_gpus))

    # Save metadata
    meta_info = {
        "model_path": args.model_path,
        "prompts_dir": os.path.abspath(args.prompts_dir),
        "total_prompts": len(prompts),
        "images_per_prompt": IMAGES_PER_PROMPT,
        "total_possible_tasks": total_possible,
        "previously_completed": skipped_existing,
        "generated_this_run": total_generated,
        "skipped_this_run": total_skipped,
        "errors_this_run": total_errors,
        "num_timesteps": args.num_timesteps,
        "cfg_text_scale": args.cfg_text_scale,
        "seed": args.seed,
        "num_gpus_used": num_gpus,
        "elapsed_seconds": elapsed,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(args.outdir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta_info, f, indent=4)

    final_completed = skipped_existing + total_generated
    
    print(f"\n{'='*60}")
    print(f"✅ Generation Complete!")
    print(f"{'='*60}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Generated this run: {total_generated}")
    print(f"Total completed: {final_completed}/{total_possible} ({(final_completed/total_possible)*100:.1f}%)")
    if total_generated > 0:
        print(f"Speed: {total_generated/elapsed:.2f} images/second")
        print(f"Time per image: {elapsed/total_generated:.2f} seconds")
    print(f"Errors: {total_errors}")
    print(f"Output: {args.outdir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
