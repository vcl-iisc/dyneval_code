import os
os.environ["HF_HOME"] = "/storage/users/anirban/akhilsakthieswaran/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/storage/users/anirban/akhilsakthieswaran/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/storage/users/anirban/akhilsakthieswaran/hf_cache"

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

# =====================================================
# CONFIGURATION (now via CLI)
# =====================================================
BATCH_SIZE = 8  # Process multiple prompts at once per GPU


def load_prompts(prompts_file):
    """Load prompts from a single TXT file."""
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
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # Speed settings
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    # Split pending indices across GPUs
    gpu_indices = [idx for i, idx in enumerate(pending_indices) if i % world_size == rank]

    if not gpu_indices:
        print(f"[GPU {rank}] No work assigned. Exiting.")
        return

    print(f"[GPU {rank}] Assigned {len(gpu_indices)} images")

    # -------------------------------
    # Load VQ Model
    # -------------------------------
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_model.eval()

    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print(f"[GPU {rank}] VQ model loaded")

    # -------------------------------
    # Load GPT Model
    # -------------------------------
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size

    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    model_weight = checkpoint.get("model", checkpoint.get("module", checkpoint.get("state_dict")))
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"[GPU {rank}] GPT model loaded")

    if args.compile:
        print(f"[GPU {rank}] Compiling GPT model...")
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)

    # -------------------------------
    # Load T5
    # -------------------------------
    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    print(f"[GPU {rank}] T5 loaded")

    # -------------------------------
    # Inference Loop
    # -------------------------------
    batch_size = min(BATCH_SIZE, len(gpu_indices))
    total_generated = 0
    total_time = 0

    for batch_start in range(0, len(gpu_indices), batch_size):
        batch_indices = gpu_indices[batch_start:batch_start + batch_size]
        batch_prompts = [all_prompts[idx] for idx in batch_indices]

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
        qzshape = [len(batch_prompts), args.codebook_embed_dim, latent_size, latent_size]

        t1 = time.time()
        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2, new_emb_masks,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )
        t2 = time.time()

        samples = vq_model.decode_code(index_sample, qzshape)
        t3 = time.time()

        samples = ((samples + 1) / 2).clamp(0, 1)

        for i, idx in enumerate(batch_indices):
            save_image(samples[i], os.path.join(args.output_dir, f"{idx:04d}.png"))
            total_generated += 1

        print(f"[GPU {rank}] {total_generated}/{len(gpu_indices)} | gen={t2-t1:.2f}s decode={t3-t2:.2f}s")

    print(f"[GPU {rank}] DONE: {total_generated} images")


def main(args):
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    if not os.path.isfile(args.prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")

    os.makedirs(args.output_dir, exist_ok=True)

    all_prompts = load_prompts(args.prompts_file)
    total_prompts = len(all_prompts)
    print(f"Loaded {total_prompts} prompts")

    pending_indices = get_pending_indices(args.output_dir, total_prompts)
    print(f"Pending: {len(pending_indices)} / {total_prompts}")

    if not pending_indices:
        print("Nothing to generate. Exiting.")
        return

    if num_gpus > 1:
        mp.spawn(worker, args=(num_gpus, args, all_prompts, pending_indices), nprocs=num_gpus, join=True)
    else:
        worker(0, 1, args, all_prompts, pending_indices)

    print("=== ALL DONE ===")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompts-file", type=str, default="/home/anirban/dheerajbaiju/dheeraj/dyneval-models/dpgbench-final-prompt-order.txt")
    parser.add_argument("--output-dir", type=str, default= "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/DPG-BENCH/januspro-dpg-results")

    parser.add_argument("--t5-path", type=str, default='google/flan-t5-xl')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--no-left-padding", action='store_true')

    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="gpt2-xl")
    parser.add_argument("--gpt-ckpt", type=str, default='/storage/users/anirban/akhilsakthieswaran/llamagen-checkpoints/t2i_XL_stage1_256.pt')
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i")
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--compile", action='store_true')

    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default='/storage/users/anirban/akhilsakthieswaran/llamagen-checkpoints/vq_ds16_t2i.pt')
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)

    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=384)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)

    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    args = parser.parse_args()
    main(args)
