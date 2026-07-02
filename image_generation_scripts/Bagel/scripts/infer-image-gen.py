
import argparse
import os
import random
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Default to PyTorch SDPA (no flash-attn) for inference scripts.
os.environ.setdefault("BAGEL_USE_FLASH_ATTN", "0")

import numpy as np
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image

DEFAULT_OUT = "outputs"
DEFAULT_OFFLOAD = os.path.join(".tmp", "bagel_offload")

SAME_DEVICE_MODULES = [
    "language_model.model.embed_tokens",
    "time_embedder",
    "latent_pos_embed",
    "vae2llm",
    "llm2vae",
    "connector",
    "vit_pos_embed",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_device_map(model, max_mem_per_gpu):
    device_map = infer_auto_device_map(
        model,
        max_memory={i: max_mem_per_gpu for i in range(torch.cuda.device_count())},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(SAME_DEVICE_MODULES[0], "cuda:0")
        for key in SAME_DEVICE_MODULES:
            device_map[key] = first_device if key in device_map else "cuda:0"
    else:
        first_device = device_map.get(SAME_DEVICE_MODULES[0])
        for key in SAME_DEVICE_MODULES:
            if key in device_map:
                device_map[key] = first_device
    return device_map


def load_inferencer(args):
    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer

    attn_backend = "flash-attn" if os.environ.get("BAGEL_USE_FLASH_ATTN") == "1" else "PyTorch SDPA"
    print(f"Attention backend: {attn_backend}", flush=True)

    model_path = args.model_path
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    if getattr(llm_config, "pad_token_id", None) is None:
        llm_config.pad_token_id = getattr(llm_config, "eos_token_id", 151643)

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=args.max_latent_size,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(args.image_size, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = build_device_map(model, args.max_mem_per_gpu)
    print(f"device_map: {device_map}", flush=True)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=args.offload_folder,
    ).eval()

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def inference_hyper(args):
    return {
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "cfg_interval": args.cfg_interval,
        "timestep_shift": args.timestep_shift,
        "num_timesteps": args.num_timesteps,
        "cfg_renorm_min": args.cfg_renorm_min,
        "cfg_renorm_type": args.cfg_renorm_type,
        "image_shapes": (args.image_size, args.image_size),
        "enable_taylorseer": args.enable_taylorseer,
        "max_think_token_n": args.max_think_token_n,
        "do_sample": args.do_sample,
    }


def generate_one(inferencer, prompt, args):
    hyper = inference_hyper(args)
    if args.think:
        output = inferencer(text=prompt, think=True, **hyper)
    else:
        output = inferencer(text=prompt, **hyper)
    return output


def parse_cfg_interval(value):
    parts = [float(x.strip()) for x in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("cfg_interval must be two floats, e.g. 0.4,1.0")
    return parts


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--use-flash-attn", action="store_true")
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.use_flash_attn:
        os.environ["BAGEL_USE_FLASH_ATTN"] = "1"

    parser = argparse.ArgumentParser(description="Generate one BAGEL image from a single prompt.")
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to BAGEL-7B-MoT weights directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUT,
                        help="Directory used with --filename")
    parser.add_argument("--filename", type=str, default="output.png",
                        help="Output filename when --output is not set")
    parser.add_argument("--output", type=str, default=None,
                        help="Full output image path")
    parser.add_argument("--think", action="store_true",
                        help="Enable think-before-generate mode")
    parser.add_argument("--max-think-token-n", type=int, default=1000)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-latent-size", type=int, default=64)
    parser.add_argument("--max-mem-per-gpu", type=str, default="40GiB")
    parser.add_argument("--offload-folder", type=str, default=DEFAULT_OFFLOAD)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=1.0)
    parser.add_argument("--cfg-interval", type=parse_cfg_interval, default=[0.4, 1.0])
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", type=str, default="global",
                        choices=["global", "channel", "text_channel"])
    parser.add_argument("--enable-taylorseer", action="store_true")
    parser.add_argument(
        "--use-flash-attn",
        action="store_true",
        help="Use flash-attn if installed (default: PyTorch SDPA only).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Loading model from {args.model_path} ...", flush=True)
    t0 = time.time()
    inferencer = load_inferencer(args)
    print(f"Model ready in {time.time() - t0:.1f}s", flush=True)

    set_seed(args.seed)
    start = time.time()
    output = generate_one(inferencer, args.prompt.strip(), args)
    image = output["image"]
    if image is None:
        raise RuntimeError("inferencer returned no image")
    image.save(output_path)
    if args.think and output.get("text"):
        print(output["text"], flush=True)
    print(f"Saved {output_path} in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
