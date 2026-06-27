#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.generation import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from transformers.generation.configuration_utils import GenerationConfig

# Import Emu3 processor (assuming it's available in the environment)
try:
    from emu3.mllm.processing_emu3 import Emu3Processor
except ImportError:
    print("❌ Emu3 processor not found. Please ensure emu3 package is installed.")
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
    parser = argparse.ArgumentParser(description="Generate images using Emu3 model")
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
    parser.add_argument(
        "--emu_hub",
        type=str,
        default="BAAI/Emu3-Gen",
        help="Emu3 model path",
    )
    parser.add_argument(
        "--vq_hub",
        type=str,
        default="BAAI/Emu3-VisionTokenizer",
        help="Emu3 Vision Tokenizer path",
    )

    args = parser.parse_args()

    # Optimize CUDA settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    print(f"📐 Image size: 1024x1024")

    if not remaining_prompts:
        print("🎉 All images already generated!")
        return

    # Load Emu3 model
    print("🔄 Loading Emu3 model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.emu_hub,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        args.emu_hub, trust_remote_code=True, padding_side="left"
    )
    image_processor = AutoImageProcessor.from_pretrained(
        args.vq_hub, trust_remote_code=True
    )
    image_tokenizer = AutoModel.from_pretrained(
        args.vq_hub, device_map="cuda:0", trust_remote_code=True
    ).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    print("✅ Model loaded")

    # Generation config
    POSITIVE_PROMPT = " masterpiece, film grained, best quality."
    NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
    CFG = 3.0

    generation_config = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=40960,
        do_sample=True,
        top_k=2048,
    )

    # Generate images
    for i, prompt_idx in enumerate(
        tqdm(remaining_indices, desc="Generating images", unit="image")
    ):
        line_no = start_idx + prompt_idx + 1  # Convert to 1-based line number
        base_prompt = prompts[prompt_idx]
        output_path = os.path.join(args.output_dir, f"{line_no}.png")

        # Skip if already exists (double-check for resume safety)
        if os.path.exists(output_path):
            continue

        try:
            prompt = base_prompt + POSITIVE_PROMPT

            kwargs = dict(
                mode="G",
                ratio="1:1",
                image_area=model.config.image_area,
                return_tensors="pt",
                padding="longest",
            )

            pos_inputs = processor(text=prompt, **kwargs)
            neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

            h = pos_inputs.image_size[:, 0]
            w = pos_inputs.image_size[:, 1]
            constrained_fn = processor.build_prefix_constrained_fn(h, w)

            input_ids = pos_inputs.input_ids.to("cuda:0")
            attn_mask = pos_inputs.attention_mask.to("cuda:0")
            neg_ids = neg_inputs.input_ids.to("cuda:0")

            logits_processor = LogitsProcessorList(
                [
                    UnbatchedClassifierFreeGuidanceLogitsProcessor(
                        CFG,
                        model,
                        unconditional_ids=neg_ids,
                    ),
                    PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
                ]
            )

            logits_processor = LogitsProcessorList(
                [
                    UnbatchedClassifierFreeGuidanceLogitsProcessor(
                        CFG,
                        model,
                        unconditional_ids=neg_ids,
                    ),
                    PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
                ]
            )

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    generation_config,
                    logits_processor=logits_processor,
                    attention_mask=attn_mask,
                )

            mm_list = processor.decode(outputs[0])
            for im in mm_list:
                if isinstance(im, Image.Image):
                    # Resize to 1024x1024 if needed
                    if im.size != (1024, 1024):
                        im = im.resize((1024, 1024), Image.LANCZOS)
                    im.save(output_path)
                    break

        except Exception as e:
            print(f"❌ Error generating line {line_no}: {e}")

    print("🎯 Generation completed!")


if __name__ == "__main__":
    main()
