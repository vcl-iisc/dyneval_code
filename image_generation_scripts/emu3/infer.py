
import argparse
import os
import time

DEFAULT_OUT = "outputs"
DEFAULT_EMU_HUB = "BAAI/Emu3-Gen"
DEFAULT_VQ_HUB = "BAAI/Emu3-VisionTokenizer"
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg "
    "artifacts, signature, watermark, username, blurry."
)


def resolve_attn_implementation(preferred):
    if preferred != "auto":
        return preferred
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_stack(emu_hub, vq_hub, attn_implementation):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import (
        LogitsProcessorList,
        PrefixConstrainedLogitsProcessor,
        UnbatchedClassifierFreeGuidanceLogitsProcessor,
    )
    from transformers.generation.configuration_utils import GenerationConfig

    from emu3.mllm.processing_emu3 import Emu3Processor

    device = "cuda:0"
    attn = resolve_attn_implementation(attn_implementation)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            emu_hub,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn,
            trust_remote_code=True,
        )
    except Exception:
        if attn == "flash_attention_2":
            attn = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                emu_hub,
                device_map=device,
                torch_dtype=torch.bfloat16,
                attn_implementation=attn,
                trust_remote_code=True,
            )
        else:
            raise

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        emu_hub, trust_remote_code=True, padding_side="left"
    )
    image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(
        vq_hub, device_map=device, trust_remote_code=True
    ).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    generation_config = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=40960,
        do_sample=True,
        top_k=2048,
    )

    return {
        "model": model,
        "processor": processor,
        "generation_config": generation_config,
        "device": device,
        "attn_implementation": attn,
        "Image": Image,
        "torch": torch,
        "LogitsProcessorList": LogitsProcessorList,
        "PrefixConstrainedLogitsProcessor": PrefixConstrainedLogitsProcessor,
        "UnbatchedClassifierFreeGuidanceLogitsProcessor": UnbatchedClassifierFreeGuidanceLogitsProcessor,
    }


def generate_one(stack, prompt, guidance_scale, ratio, target_size):
    model = stack["model"]
    processor = stack["processor"]
    generation_config = stack["generation_config"]
    device = stack["device"]
    torch = stack["torch"]
    Image = stack["Image"]
    LogitsProcessorList = stack["LogitsProcessorList"]
    PrefixConstrainedLogitsProcessor = stack["PrefixConstrainedLogitsProcessor"]
    UnbatchedClassifierFreeGuidanceLogitsProcessor = stack["UnbatchedClassifierFreeGuidanceLogitsProcessor"]

    full_prompt = prompt + POSITIVE_PROMPT
    proc_kwargs = dict(
        mode="G",
        ratio=ratio,
        image_area=model.config.image_area,
        return_tensors="pt",
        padding="longest",
    )

    pos_inputs = processor(text=full_prompt, **proc_kwargs)
    neg_inputs = processor(text=NEGATIVE_PROMPT, **proc_kwargs)

    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    constrained_fn = processor.build_prefix_constrained_fn(h, w)
    logits_processor = LogitsProcessorList([
        UnbatchedClassifierFreeGuidanceLogitsProcessor(
            guidance_scale,
            model,
            unconditional_ids=neg_inputs.input_ids.to(device),
        ),
        PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
    ])

    with torch.inference_mode():
        outputs = model.generate(
            pos_inputs.input_ids.to(device),
            generation_config,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(device),
        )

    for item in processor.decode(outputs[0]):
        if isinstance(item, Image.Image):
            if target_size and item.size != target_size:
                item = item.resize(target_size, Image.LANCZOS)
            return item

    raise RuntimeError("Emu3 decode returned no image")


def parse_gpus(gpu_str):
    if not gpu_str:
        return None
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]


def main():
    ap = argparse.ArgumentParser(description="Generate one Emu3 image from a single prompt.")
    ap.add_argument("prompt", help="text prompt to generate")
    ap.add_argument("--output", default=None, help="full output image path")
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--filename", default="output.png")
    ap.add_argument("--emu_hub", default=DEFAULT_EMU_HUB)
    ap.add_argument("--vq_hub", default=DEFAULT_VQ_HUB)
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--ratio", default="1:1")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument(
        "--resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resize decoded image to width x height (default: 1024x1024)",
    )
    ap.add_argument(
        "--attn_implementation",
        default="auto",
        choices=["auto", "flash_attention_2", "sdpa", "eager"],
    )
    ap.add_argument("--gpu", type=int, default=None, help="GPU id to expose as cuda:0")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"  emu_hub={args.emu_hub}")
    print(f"  vq_hub={args.vq_hub}")
    print(f"  ratio={args.ratio}  guidance={args.guidance_scale}  attn={args.attn_implementation}")
    print(f"  output -> {output}")

    t0 = time.time()
    print(f"Loading Emu3 ({args.emu_hub}) attn={args.attn_implementation} ...")
    stack = load_stack(args.emu_hub, args.vq_hub, args.attn_implementation)
    print(
        f"Model ready in {time.time() - t0:.1f}s "
        f"(attn={stack['attn_implementation']})"
    )

    target_size = (args.width, args.height) if args.resize else None
    start = time.time()
    image = generate_one(
        stack,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        ratio=args.ratio,
        target_size=target_size,
    )
    image.save(output)
    print(f"Saved {output} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
