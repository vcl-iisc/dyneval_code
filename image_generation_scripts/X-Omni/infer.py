import argparse
import os

DEFAULT_OUT = "outputs"
DEFAULT_MODEL = "X-Omni/X-Omni-En"
DEFAULT_FLUX = "black-forest-labs/FLUX.1-dev"


def parse_image_size(image_size):
    if isinstance(image_size, int):
        return image_size, image_size
    if len(image_size) == 1:
        return image_size[0], image_size[0]
    return image_size[0], image_size[1]


def resolve_output_path(args):
    if args.output:
        return args.output
    return os.path.join(args.output_dir, args.filename)


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    ).cuda()
    model.init_vision(args.flux_model_name_or_path)
    model.set_generation_mode("image")
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, prompt, args):
    import torch
    from transformers.generation import GenerationConfig

    image_size = parse_image_size(args.image_size)
    token_h = image_size[0] // args.downsample_size
    token_w = image_size[1] // args.downsample_size
    image_prefix = f"<SOM>{token_h} {token_w}<IMAGE>"

    generation_config = GenerationConfig(
        max_new_tokens=token_h * token_w,
        do_sample=True,
        temperature=args.temperature,
        min_p=args.min_p,
        top_p=args.top_p,
        guidance_scale=args.cfg_scale,
        suppress_tokens=tokenizer.convert_tokens_to_ids(model.config.mm_special_tokens),
    )

    tokens = tokenizer(
        [prompt + image_prefix],
        return_tensors="pt",
        padding="longest",
        padding_side="left",
    )
    input_ids = tokens.input_ids.cuda()
    attention_mask = tokens.attention_mask.cuda()
    negative_ids = tokenizer.encode(
        image_prefix,
        add_special_tokens=False,
        return_tensors="pt",
    ).cuda().expand(1, -1)

    torch.manual_seed(args.seed)
    out_tokens = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        negative_prompt_ids=negative_ids,
    )

    out_tokens = torch.nn.functional.pad(
        out_tokens,
        (0, 1),
        value=tokenizer.convert_tokens_to_ids("<EOM>"),
    )
    torch.manual_seed(args.seed)
    _, images = model.mmdecode(tokenizer, out_tokens[0], skip_special_tokens=False)
    return images[0]


def main():
    parser = argparse.ArgumentParser(description="Generate one X-Omni image from a single prompt.")
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--output", default=None, help="full output image path")
    parser.add_argument("--output_dir", default=DEFAULT_OUT, help="directory used with --filename")
    parser.add_argument("--filename", default="output.png", help="output filename when --output is not set")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--flux_model_name_or_path", default=DEFAULT_FLUX)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs="+",
        default=[1152],
        dest="image_size",
        help="image size as one value for square output, or height width",
    )
    parser.add_argument("--downsample-size", type=int, default=16, dest="downsample_size")
    parser.add_argument("--cfg-scale", type=float, default=1.0, dest="cfg_scale")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.03, dest="min_p")
    parser.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["sdpa", "eager", "flash_attention_2"],
        dest="attn_implementation",
        help="attention backend for model loading; use flash_attention_2 only when flash-attn is installed",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU id to expose as cuda:0")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_path = resolve_output_path(args)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Loading X-Omni model: {args.model_name_or_path}")
    print(f"Loading FLUX decoder: {args.flux_model_name_or_path}")
    model, tokenizer = load_model(args)

    image = generate_one(model, tokenizer, args.prompt, args)
    image.save(output_path)
    print(f"Saved image to {output_path}")


if __name__ == "__main__":
    main()
