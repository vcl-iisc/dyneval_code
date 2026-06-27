# coding=utf-8
# DPG-Bench / JanusPro Inference with start_index + end_index + resume + multi-GPU

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
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
from torchvision.utils import save_image

import time
import argparse
import torch.multiprocessing as mp
from tokenizer.tokenizer_image.vq_model import VQ_models
from language.t5 import T5Embedder
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BATCH_SIZE = 8


def load_prompts(prompts_file):
    with open(prompts_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def get_pending_indices(output_dir, indices):
    os.makedirs(output_dir, exist_ok=True)
    pending = []
    for i in indices:
        img_path = os.path.join(output_dir, f"{i:04d}.png")
        if not os.path.exists(img_path):
            pending.append(i)
    return pending


def worker(rank, world_size, args, all_prompts, pending_indices):
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    gpu_indices = [idx for i, idx in enumerate(pending_indices) if i % world_size == rank]

    if not gpu_indices:
        print(f"[GPU {rank}] No work assigned.")
        return

    print(f"[GPU {rank}] Assigned {len(gpu_indices)} prompts")

    # ---------------- VQ ----------------
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim
    ).to(device)
    vq_model.eval()

    ckpt = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(ckpt["model"])
    del ckpt
    print(f"[GPU {rank}] VQ model loaded")

    # ---------------- GPT ----------------
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size

    gpt_model = GPT_models[args.gpt_model](
        block_size=latent_size ** 2,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
    model_weight = ckpt.get("model", ckpt.get("module", ckpt.get("state_dict")))
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del ckpt
    print(f"[GPU {rank}] GPT model loaded")

    if args.compile:
        gpt_model = torch.compile(gpt_model, mode="reduce-overhead", fullgraph=True)

    # ---------------- T5 ----------------
    t5_model = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
        model_max_length=args.t5_feature_max_len,
    )
    print(f"[GPU {rank}] T5 loaded")

    # ---------------- Inference ----------------
    batch_size = min(BATCH_SIZE, len(gpu_indices))
    total_generated = 0

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

        index_sample = generate(
            gpt_model, c_indices, latent_size ** 2, new_emb_masks,
            cfg_scale=args.cfg_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )

        samples = vq_model.decode_code(index_sample, qzshape)
        samples = ((samples + 1) / 2).clamp(0, 1)

        for i, idx in enumerate(batch_indices):
            save_image(samples[i], os.path.join(args.output_dir, f"{idx:04d}.png"))
            total_generated += 1

        print(f"[GPU {rank}] {total_generated}/{len(gpu_indices)} done")

    print(f"[GPU {rank}] DONE")


def main(args):
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    all_prompts = load_prompts(args.prompts_file)
    total = len(all_prompts)

    start = args.start_index
    end = args.end_index if args.end_index is not None else total - 1

    assert 0 <= start <= end < total, f"Invalid range [{start}, {end}] for total={total}"

    selected_indices = list(range(start, end + 1))
    pending_indices = get_pending_indices(args.output_dir, selected_indices)

    print(f"🚀 Running prompts [{start} → {end}] | Pending: {len(pending_indices)}")

    if not pending_indices:
        print("Nothing left to generate.")
        return

    if num_gpus > 1:
        mp.spawn(worker, args=(num_gpus, args, all_prompts, pending_indices), nprocs=num_gpus, join=True)
    else:
        worker(0, 1, args, all_prompts, pending_indices)

    print("=== ALL DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompts-file", type=str, default="/home/anirban/dheerajbaiju/dheeraj/dyneval-models/dpgbench-final-prompt-order.txt")
    parser.add_argument("--output-dir", type=str, default="/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/DPG-BENCH/januspro-dpg-results")

    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)

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
