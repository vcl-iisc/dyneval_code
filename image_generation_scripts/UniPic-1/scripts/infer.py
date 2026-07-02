
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from einops import rearrange
from mmengine.config import Config
from PIL import Image

from src.builder import BUILDER

DEFAULT_OUT = "outputs"


def configure_torch(disable_cudnn=False):
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.init()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.enabled = not disable_cudnn
        # Warm up CUDA/cuDNN before the first real forward pass.
        torch.zeros(1, device="cuda").item()


def prepare_model_for_inference(model):
    """Disable training-only paths that break inference (checkpointing/cuDNN)."""
    if hasattr(model, "mar"):
        model.mar.grad_checkpointing = False
        if hasattr(model.mar, "gradient_checkpointing_disable"):
            model.mar.gradient_checkpointing_disable()
        net = getattr(getattr(model.mar, "diffloss", None), "net", None)
        if net is not None:
            net.grad_checkpointing = False
    if hasattr(model, "llm") and hasattr(model.llm, "gradient_checkpointing_disable"):
        model.llm.gradient_checkpointing_disable()
    model.eval()
    return model


def resolve_model_paths(config, checkpoint_path, model_dir=None):
    """Point VAE/LLM/SigLIP paths at the directory that contains pytorch_model.bin."""
    if model_dir is None:
        model_dir = os.path.dirname(os.path.abspath(checkpoint_path))

    paths = {
        "kl16.ckpt": os.path.join(model_dir, "kl16.ckpt"),
        "Qwen2.5-1.5B-Instruct": os.path.join(model_dir, "Qwen2.5-1.5B-Instruct"),
        "siglip2-so400m-patch16-512": os.path.join(model_dir, "siglip2-so400m-patch16-512"),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {name} under model dir {model_dir}. "
                f"Expected: {path}"
            )

    config.model.vae.ckpt_path = paths["kl16.ckpt"]
    config.model.llm.pretrained_model_name_or_path = paths["Qwen2.5-1.5B-Instruct"]
    config.model.siglip2.pretrained_model_name_or_path = paths["siglip2-so400m-patch16-512"]
    config.model.tokenizer.pretrained_model_name_or_path = paths["Qwen2.5-1.5B-Instruct"]
    return model_dir


def warn_cuda_env():
    if "CUDA_VISIBLE_DEVICE" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        print(
            "WARNING: CUDA_VISIBLE_DEVICE is ignored by CUDA/PyTorch. "
            "Use CUDA_VISIBLE_DEVICES instead.",
            flush=True,
        )


def load_model(config_path, checkpoint_path, attn_implementation="eager", model_dir=None,
               disable_cudnn=False):
    configure_torch(disable_cudnn=disable_cudnn)
    config = Config.fromfile(config_path)
    model_dir = resolve_model_paths(config, checkpoint_path, model_dir)
    print(f"Model components from: {model_dir}", flush=True)
    if attn_implementation:
        config.model.llm.attn_implementation = attn_implementation
    config.model.mar.grad_checkpointing = False
    model = BUILDER.build(config.model).eval().cuda()
    model = model.to(model.dtype)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return prepare_model_for_inference(model)


def generate_image(model, prompt, args, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    formatted = f"Generate an image: {prompt.strip()}"
    class_info = model.prepare_text_conditions(formatted, args.cfg_prompt)

    input_ids = class_info["input_ids"]
    attention_mask = class_info["attention_mask"]

    assert len(input_ids) == 2
    if args.cfg == 1.0:
        input_ids = input_ids[:1]
        attention_mask = attention_mask[:1]

    bsz = args.grid_size ** 2
    if args.cfg != 1.0:
        input_ids = torch.cat([
            input_ids[:1].expand(bsz, -1),
            input_ids[1:].expand(bsz, -1),
        ])
        attention_mask = torch.cat([
            attention_mask[:1].expand(bsz, -1),
            attention_mask[1:].expand(bsz, -1),
        ])
    else:
        input_ids = input_ids.expand(bsz, -1)
        attention_mask = attention_mask.expand(bsz, -1)

    m = n = args.image_size // 16
    samples = model.sample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_iter=args.num_iter,
        cfg=args.cfg,
        cfg_schedule=args.cfg_schedule,
        temperature=args.temperature,
        progress=False,
        image_shape=(m, n),
    )
    samples = rearrange(
        samples, "(m n) c h w -> (m h) (n w) c", m=args.grid_size, n=args.grid_size)
    samples = torch.clamp(
        127.5 * samples + 128.0, 0, 255).to("cpu", dtype=torch.uint8).numpy()
    return Image.fromarray(samples)


def is_cudnn_init_error(exc):
    return "CUDNN_STATUS_NOT_INITIALIZED" in str(exc)


def disable_cudnn_after_failure():
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Generate one UniPic image from a single prompt.")
    parser.add_argument("config", help="config file path")
    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory with kl16.ckpt, Qwen2.5-1.5B-Instruct/, siglip2-so400m-patch16-512/ "
        "(default: parent directory of --checkpoint).",
    )
    parser.add_argument("--output_dir", default=DEFAULT_OUT,
                        help="directory used with --filename")
    parser.add_argument("--filename", type=str, default="output.jpg")
    parser.add_argument("--output", type=str, default=None,
                        help="full output image path")
    parser.add_argument("--cfg_prompt", type=str, default="Generate an image.")
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_schedule", type=str, default="constant")
    parser.add_argument("--num_iter", type=int, default=32)
    parser.add_argument("--grid_size", type=int, default=1,
                        help="grid side length (1 = single 1024x1024 image)")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="disable cuDNN; useful if the driver reports CUDNN_STATUS_NOT_INITIALIZED",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="LLM attention backend (default: eager, no flash-attn).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=None, help="GPU id to expose as cuda:0")
    args = parser.parse_args()
    warn_cuda_env()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"  checkpoint={args.checkpoint}")
    print(f"  cfg={args.cfg}  num_iter={args.num_iter}"
          f"  size={args.image_size}  grid_size={args.grid_size}  seed={args.seed}"
          f"  attn={args.attn_implementation}")
    print(f"  output={output}")

    model = load_model(
        args.config, args.checkpoint, args.attn_implementation, args.model_dir,
        args.disable_cudnn)
    try:
        image = generate_image(model, args.prompt, args, seed=args.seed)
    except RuntimeError as exc:
        if args.disable_cudnn or not is_cudnn_init_error(exc):
            raise
        print("cuDNN init failed; disabling cuDNN and retrying once.", flush=True)
        args.disable_cudnn = True
        disable_cudnn_after_failure()
        image = generate_image(model, args.prompt, args, seed=args.seed)
    image.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
