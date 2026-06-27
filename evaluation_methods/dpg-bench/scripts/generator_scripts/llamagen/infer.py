from pathlib import Path
import glob
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from torchvision.utils import save_image

import os
import time
import argparse
import torch.multiprocessing as mp
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configuration
PROMPTS_DIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark/prompts/dpg_bench"
OUTPUT_DIR = "evalmuse-llamagen-train-images"
BATCH_SIZE = 8  # Process multiple prompts at once per GPU
# NOTE: For faster generation, using image_size=256 (256 tokens) instead of 512 (1024 tokens)
#       This is ~4x faster but lower resolution. Change --image-size 512 for higher quality.


def load_prompts(prompts_file):
    """Load prompts from file."""
    with open(prompts_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def get_pending_indices(output_dir, total_prompts):
    """Get indices of prompts that haven't been processed yet."""
    os.makedirs(output_dir, exist_ok=True)
    pending = []
    for i in range(total_prompts):
        img_path = os.path.join(output_dir, f"{i:04d}.png")
        if not os.path.exists(img_path):
            pending.append(i)
    return pending


def worker(rank, world_size, args, all_prompts, pending_indices):
    """Worker function for each GPU."""
    # Set device for this process
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    
    # Setup PyTorch for speed
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)
    
    # Split pending indices for this GPU
    gpu_indices = [idx for i, idx in enumerate(pending_indices) if i % world_size == rank]
    
    if not gpu_indices:
        print(f"[GPU {rank}] No work assigned. Exiting.")
        return
    
    print(f"[GPU {rank}] Assigned {len(gpu_indices)} images to generate")

    # Load VQ model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"[GPU {rank}] Image tokenizer loaded")

    # Load GPT model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if "model" in checkpoint:
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight")
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"[GPU {rank}] GPT model loaded")

    # Compile model for faster inference
    if args.compile:
        print(f"[GPU {rank}] Compiling model...")
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)

    # Load T5 text encoder (downloads from HuggingFace if not cached)
    t5_model = T5Embedder(
        device=device,
        local_cache=True,  # Use local files
        cache_dir=args.t5_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    print(f"[GPU {rank}] T5 model loaded")

    # Process in batches
    batch_size = min(BATCH_SIZE, len(gpu_indices))
    total_generated = 0
    total_time = 0

    for batch_start in range(0, len(gpu_indices), batch_size):
        batch_indices = gpu_indices[batch_start:batch_start + batch_size]
        batch_prompts = [all_prompts[idx] for idx in batch_indices]
        current_batch_size = len(batch_prompts)

        # Get text embeddings
        caption_embs, emb_masks = t5_model.get_text_embeddings(batch_prompts)

        if not args.no_left_padding:
            new_emb_masks = torch.flip(emb_masks, dims=[-1])
            new_caption_embs = []
            for caption_emb, emb_mask in zip(caption_embs, emb_masks):
                valid_num = int(emb_mask.sum().item())
                new_caption_emb = torch.cat([caption_emb[valid_num:], caption_emb[:valid_num]])
                new_caption_embs.append(new_caption_emb)
            new_caption_embs = torch.stack(new_caption_embs)
        else:
            new_caption_embs, new_emb_masks = caption_embs, emb_masks

        c_indices = new_caption_embs * new_emb_masks[:, :, None]
        c_emb_masks = new_emb_masks
        qzshape = [current_batch_size, args.codebook_embed_dim, latent_size, latent_size]

        # Generate
        t1 = time.time()
        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2,
            c_emb_masks,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, sample_logits=True,
        )
        sampling_time = time.time() - t1

        # Decode
        t2 = time.time()
        samples = vq_model.decode_code(index_sample, qzshape)
        decoder_time = time.time() - t2

        batch_time = sampling_time + decoder_time
        total_time += batch_time

        # Save images
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)

        for i, idx in enumerate(batch_indices):
            img_path = os.path.join(OUTPUT_DIR, f"{idx:04d}.png")
            save_image(samples[i], img_path)
            total_generated += 1

        print(f"[GPU {rank}] Batch {batch_start // batch_size + 1}: {total_generated}/{len(gpu_indices)} | {batch_time:.2f}s")

    print(f"[GPU {rank}] COMPLETE: Generated {total_generated} images in {total_time:.2f}s")


def main(args):
    # Detect number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")
    
    if num_gpus == 0:
        raise RuntimeError("No GPUs available!")

    # Load prompts
    all_prompts = load_prompts(PROMPTS_FILE)
    total_prompts = len(all_prompts)
    print(f"Loaded {total_prompts} prompts from {PROMPTS_FILE}")

    # Get pending indices
    pending_indices = get_pending_indices(OUTPUT_DIR, total_prompts)
    print(f"Pending images to generate: {len(pending_indices)} / {total_prompts}")

    if not pending_indices:
        print("All images already generated. Exiting.")
        return

    # Launch multi-GPU workers
    if num_gpus > 1:
        print(f"Starting multi-GPU generation with {num_gpus} GPUs...")
        mp.spawn(
            worker,
            args=(num_gpus, args, all_prompts, pending_indices),
            nprocs=num_gpus,
            join=True
        )
    else:
        print("Single GPU mode")
        worker(0, 1, args, all_prompts, pending_indices)

    print(f"\n=== ALL COMPLETE ===")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t5-path", type=str, default='/home/anirban/yashwanthm/dheeraj/LlamaGen')  # Parent dir, not the model folder
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--no-left-padding", action='store_true', default=False)
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default='/home/anirban/yashwanthm/dheeraj/LlamaGen/t2i_XL_stage1_256.pt')
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i")
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true', default=False)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='/home/anirban/yashwanthm/dheeraj/LlamaGen/vq_ds16_t2i.pt')
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    main(args)
