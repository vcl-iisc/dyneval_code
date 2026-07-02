#!/usr/bin/env python3
"""Generate missing Kolors images for the DYNEVAL-1K set.

Uses the Kolors repo at ./Kolors (same pipeline as scripts/sample.py).
Loads the model once, then generates every prompt not yet present in:

    DYNEVAL-1K-IMAGES/Kolors/<gid>.png

Usage (from DYNEVAL project root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/Kolors/scripts/image-gen.py
    python dyneval_code/generator_scripts/Kolors/scripts/image-gen.py --limit 5
    python dyneval_code/generator_scripts/Kolors/scripts/image-gen.py --dry-run
"""
import argparse
import csv
import os
import sys
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KOLORS_ROOT = os.path.dirname(SCRIPT_DIR)
MAP = "DYNEVAL-1K-PROMPTS-MAP.tsv"
OUT_DIR = os.path.join("DYNEVAL-1K-IMAGES", "Kolors")
EXT = {".png", ".jpg", ".jpeg", ".webp"}


def load_pipe():
    sys.path.insert(0, KOLORS_ROOT)
    os.chdir(KOLORS_ROOT)

    import torch
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, UNet2DConditionModel
    from kolors.models.modeling_chatglm import ChatGLMModel
    from kolors.models.tokenization_chatglm import ChatGLMTokenizer
    from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import (
        StableDiffusionXLPipeline,
    )

    ckpt_dir = os.path.join(KOLORS_ROOT, "weights", "Kolors")
    text_encoder = ChatGLMModel.from_pretrained(
        f"{ckpt_dir}/text_encoder", torch_dtype=torch.float16
    ).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f"{ckpt_dir}/text_encoder")
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(
        f"{ckpt_dir}/unet", revision=None
    ).half()
    pipe = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=False,
    )
    pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    return pipe


def generate(pipe, prompt, seed):
    import torch

    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        num_images_per_prompt=1,
        generator=torch.Generator(pipe.device).manual_seed(seed),
    ).images[0]
    return image


def pending_rows():
    have = set()
    if os.path.isdir(OUT_DIR):
        for fn in os.listdir(OUT_DIR):
            if os.path.splitext(fn.lower())[1] in EXT:
                have.add(os.path.splitext(fn)[0].zfill(4))

    rows = list(csv.DictReader(open(MAP, encoding="utf-8"), delimiter="\t"))
    missing = []
    for r in rows:
        gid = r["id"].zfill(4)
        if gid not in have:
            missing.append((gid, r["benchmark"], r["prompt"]))
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=66,
                    help="same default as Kolors scripts/sample.py")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    todo = pending_rows()
    print(f"[Kolors] missing={len(todo)}  -> {OUT_DIR}")
    if args.limit:
        todo = todo[: args.limit]
        print(f"  limited to {len(todo)} this run")
    if not todo:
        print("Nothing to do.")
        return
    if args.dry_run:
        for gid, bench, prompt in todo[:20]:
            print(f"  {gid}  [{bench}]  {prompt[:90]}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return

    print("Loading Kolors pipeline (once) ...")
    pipe = load_pipe()

    fail_log = "generate_fail_Kolors.log"
    ok = bad = 0
    for i, (gid, bench, prompt) in enumerate(todo, 1):
        dst = os.path.join(OUT_DIR, f"{gid}.png")
        try:
            generate(pipe, prompt, args.seed).save(dst)
            ok += 1
            print(f"[{i}/{len(todo)}] {gid} OK  [{bench}]")
        except Exception as e:
            bad += 1
            with open(fail_log, "a", encoding="utf-8") as lf:
                lf.write(f"{gid}\t{bench}\t{prompt}\t{e}\n")
                lf.write(traceback.format_exc() + "\n")
            print(f"[{i}/{len(todo)}] {gid} FAILED: {e}")

    print(f"\nDone Kolors: generated={ok}  failed={bad}  -> {OUT_DIR}")
    if bad:
        print(f"Failures logged to {fail_log}")


if __name__ == "__main__":
    main()
