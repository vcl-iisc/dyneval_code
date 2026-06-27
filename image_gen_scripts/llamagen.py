#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

# Set up HF cache
HF_CACHE = os.environ.get("HF_HOME", "/tmp/hf_cache")
os.environ["HF_HOME"] = HF_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Optimize PyTorch settings
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# Import LlamaGen components (assuming they're available in the environment)
try:
    from autoregressive.models.generate import generate
    from autoregressive.models.gpt import GPT_models
    from language.t5 import T5Embedder
    from tokenizer.tokenizer_image.vq_model import VQ_models
except ImportError:
    print(
        "❌ LlamaGen components not found. Please ensure LlamaGen package is installed."
    )
    exit(1)


def load_prompts_range(prompts_file, start_line, finish_line):
    """Load prompts from specified line range (1-based)"""
    with open(prompts_file, "r") as f:
        all_prompts = [line.strip() for line in f if line.strip()]

    # Convert to 0-based indexing
    start_idx = start_line - 1
    finish_idx = finish_line

    if start_idx < 0:
        start_idx = 0
    if finish_idx > len(all_prompts):
        finish_idx = len(all_prompts)

    prompts = all_prompts[start_idx:finish_idx]
    return prompts, start_idx


def get_existing_images(output_dir, start_idx, num_prompts):
    """Get set of already generated image indices"""
    existing = set()
    if not os.path.exists(output_dir):
        return existing

    for i in range(num_prompts):
        line_no = start_idx + i + 1  # Convert back to 1-based
        img_path = os.path.join(output_dir, f"{line_no}.png")
        if os.path.exists(img_path):
            existing.add(i)

    return existing


def main():
    parser = argparse.ArgumentParser(description="Generate images using LlamaGen model")
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to text file containing prompts (one per line)",
    )
    parser.add_argument(
        "--start_line", type=int, required=True, help="Starting line number (1-based)"
    )
    parser.add_argument(
        "--finish_line", type=int, required=True, help="Ending line number (1-based)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated images",
    )

    # Model-specific arguments
    parser.add_argument("--gpt-model", type=str, default="gpt2-xl")
    parser.add_argument(
        "--gpt-ckpt", type=str, required=True, help="Path to GPT checkpoint"
    )
    parser.add_argument("--gpt-type", type=str, default="t2i")
    parser.add_argument("--cls-token-num", type=int, default=120)
    parser.add_argument("--t5-feature-max-len", type=int, default=120)

    parser.add_argument("--vq-model", type=str, default="VQ-16")
    parser.add_argument(
        "--vq-ckpt", type=str, required=True, help="Path to VQ checkpoint"
    )
    parser.add_argument("--codebook-size", type=int, default=16384)
    parser.add_argument("--codebook-embed-dim", type=int, default=8)

    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--downsample-size", type=int, default=16)

    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["none", "bf16", "fp16"]
    )
    parser.add_argument("--cfg-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts for specified range
    prompts, start_idx = load_prompts_range(
        args.prompts_file, args.start_line, args.finish_line
    )

    if not prompts:
        print("❌ No prompts found in specified range")
        return

    # Check for existing images (resume logic)
    existing = get_existing_images(args.output_dir, start_idx, len(prompts))
    remaining_indices = [i for i in range(len(prompts)) if i not in existing]
    remaining_prompts = [prompts[i] for i in remaining_indices]

    print(f"📋 Total prompts in range: {len(prompts)}")
    print(f"✅ Already generated: {len(existing)}")
    print(f"🕗 Remaining to generate: {len(remaining_prompts)}")
    print(f"📐 Image size: {args.image_size}x{args.image_size}")

    if not remaining_prompts:
        print("🎉 All images already generated!")
        return

    # Set device and precision
    device = "cuda"
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    latent_size = args.image_size // args.downsample_size

    # Load VQ model
    print("🔄 Loading VQ model...")
    vq_model = (
        VQ_models[args.vq_model](
            codebook_size=args.codebook_size, codebook_embed_dim=args.codebook_embed_dim
        )
        .to(device)
        .eval()
    )

    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    print("✅ VQ model loaded")

    # Load GPT model
    print("🔄 Loading GPT model...")
    gpt_model = (
        GPT_models[args.gpt_model](
            block_size=latent_size**2,
            cls_token_num=args.cls_token_num,
            model_type=args.gpt_type,
        )
        .to(device=device, dtype=precision)
        .eval()
    )

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    gpt_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
    del checkpoint
    print("✅ GPT model loaded")

    # Load T5 model
    print("🔄 Loading T5 model...")
    t5_repo_id = "google/flan-t5-xl"
    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=HF_CACHE,
        dir_or_name=t5_repo_id,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    print("✅ T5 model loaded")

    # Generate images
    batch_size = 4  # Adjust based on memory

    for i in range(0, len(remaining_indices), batch_size):
        batch_indices = remaining_indices[i : i + batch_size]
        batch_prompts = [prompts[idx] for idx in batch_indices]

        # Get text embeddings
        caption_embs, emb_masks = t5_model.get_text_embeddings(batch_prompts)
        c_indices = caption_embs * emb_masks[:, :, None]
        qzshape = [
            len(batch_prompts),
            args.codebook_embed_dim,
            latent_size,
            latent_size,
        ]

        # Generate indices
        index_sample = generate(
            gpt_model,
            c_indices,
            latent_size**2,
            emb_masks,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )

        # Decode to images
        samples = vq_model.decode_code(index_sample, qzshape)
        samples = ((samples + 1) / 2).clamp(0, 1)

        # Save images
        for j, prompt_idx in enumerate(batch_indices):
            line_no = start_idx + prompt_idx + 1  # Convert to 1-based line number
            output_path = os.path.join(args.output_dir, f"{line_no}.png")

            # Skip if already exists (double-check for resume safety)
            if os.path.exists(output_path):
                continue

            try:
                save_image(samples[j], output_path)
            except Exception as e:
                print(f"❌ Error saving line {line_no}: {e}")

        print(
            f"🔄 Generated batch {i // batch_size + 1}/{(len(remaining_indices) + batch_size - 1) // batch_size}"
        )

    print("🎯 Generation completed!")


if __name__ == "__main__":
    main()
