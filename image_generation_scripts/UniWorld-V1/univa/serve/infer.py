import argparse
import os
import sys

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch import nn
from transformers import AutoProcessor, set_seed
from transformers import SiglipImageProcessor, SiglipVisionModel

# Repo root (UniWorld-V1/) must be on sys.path for `import univa`.
# `sys.path.append("..")` only works when cwd is univa/serve/; use the file
# location instead so both `python -m univa.serve.cli` and
# `python univa/serve/cli.py` work from the repo root.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl import (
    UnivaQwen2p5VLForConditionalGeneration,
)
from univa.utils.anyres_util import dynamic_resize
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from univa.utils.flux_pipeline import FluxPipeline


DEFAULT_OUT = "outputs"
DEFAULT_SEED = 42
DEFAULT_MODEL_PATH = "PKU-YuanGroup/UniWorld-V1"
DEFAULT_FLUX_PATH = "black-forest-labs/FLUX.1-dev"
DEFAULT_SIGLIP_PATH = "google/siglip-so400m-patch14-384"

set_seed(DEFAULT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(DEFAULT_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def resolve_pretrained_path(model_path):
    if os.path.isdir(model_path):
        expected_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
            "model.safetensors.index.json",
        ]
        if any(os.path.exists(os.path.join(model_path, f)) for f in expected_files):
            return model_path
        snapshots_dir = os.path.join(model_path, "snapshots")
        if os.path.isdir(snapshots_dir):
            snapshots = sorted(
                d for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))
            )
            if snapshots:
                resolved = os.path.join(snapshots_dir, snapshots[-1])
                print(f"Resolved model_path to snapshot: {resolved}")
                return resolved
    if "/" in model_path:
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=model_path)
    return model_path


def resolve_task_head_path(model_path, task_head_path=None):
    if task_head_path:
        if os.path.exists(task_head_path):
            return task_head_path
        raise FileNotFoundError(
            f"Explicit task_head_path provided but not found: {task_head_path}"
        )

    candidates = [model_path]
    if os.path.isdir(model_path):
        parent = os.path.dirname(model_path)
        if parent and parent != model_path:
            candidates.append(parent)
        grandparent = os.path.dirname(parent)
        if grandparent and grandparent != parent:
            candidates.append(grandparent)

    for path in candidates:
        candidate = os.path.join(path, "task_head_final.pt")
        if os.path.exists(candidate):
            print(f"Resolved task_head_final.pt to: {candidate}")
            return candidate

    raise FileNotFoundError(
        "task_head_final.pt was not found in the resolved model path or its parent directories. "
        "Please provide the correct model root or use --task_head_path."
    )


def load_main_model_and_processor(
    model_path,
    device,
    task_head_path=None,
    attn_implementation="sdpa",
    min_pixels=448 * 448,
    max_pixels=448 * 448,
):
    model_path = resolve_pretrained_path(model_path)
    model = UnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).to(device)
    task_head = nn.Sequential(
        nn.Linear(3584, 10240),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(10240, 2),
    ).to(device)
    task_head_file = resolve_task_head_path(model_path, task_head_path)
    task_head.load_state_dict(torch.load(task_head_file))
    task_head.eval()

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return model, task_head, processor


def load_pipe(denoiser, flux_path, device):
    pipe = FluxPipeline.from_pretrained(
        flux_path,
        transformer=denoiser,
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(device)
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    return pipe, tokenizers, text_encoders


def load_siglip_and_processor(siglip_path, device):
    siglip_processor, siglip_model = None, None
    if siglip_path:
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
        siglip_model = SiglipVisionModel.from_pretrained(
            siglip_path,
            torch_dtype=torch.bfloat16,
        ).to(device)
    return siglip_processor, siglip_model


def preprocess_siglip_pixel_values(siglip_model, siglip_processor, image_paths):
    siglip_pixel_values = []
    for image_path in image_paths:
        siglip_pixel_value = siglip_processor.preprocess(
            images=Image.open(image_path).convert("RGB"),
            do_resize=True,
            return_tensors="pt",
            do_convert_rgb=True,
        ).pixel_values
        siglip_pixel_values.append(siglip_pixel_value)
    siglip_pixel_values = torch.concat(siglip_pixel_values)
    siglip_pixel_values = siglip_pixel_values.to(siglip_model.device)
    return siglip_model(siglip_pixel_values).last_hidden_state


def update_size(i1, i2, anyres="any_11ratio", anchor_pixels=1024 * 1024):
    shapes = []
    for p in (i1, i2):
        if p:
            im = Image.open(p)
            w, h = im.size
            shapes.append((w, h))
    if not shapes:
        return int(anchor_pixels**0.5), int(anchor_pixels**0.5)
    if len(shapes) == 1:
        w, h = shapes[0]
    else:
        w = sum(s[0] for s in shapes) / len(shapes)
        h = sum(s[1] for s in shapes) / len(shapes)
    new_h, new_w = dynamic_resize(int(h), int(w), anyres, anchor_pixels=anchor_pixels)
    return new_h, new_w


def generate_t2i_image(
    args,
    state,
    prompt,
    image_paths=None,
    height=None,
    width=None,
    force_generate=True,
):
    """Text-to-image for a single prompt. Returns PIL.Image or None if routed to text."""
    device = state["device"]
    model = state["model"]
    task_head = state["task_head"]
    processor = state["processor"]
    pipe = state["pipe"]
    tokenizers = state["tokenizers"]
    text_encoders = state["text_encoders"]
    siglip_processor = state["siglip_processor"]
    siglip_model = state["siglip_model"]

    image_paths = image_paths or []
    height = height or args.height
    width = width or args.width

    conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    chat_text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    chat_text = "<|im_end|>\n".join(
        chat_text.split("<|im_end|>\n")[1:]
    )
    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    assistant_mask = inputs.input_ids == 77091
    assistant_vectors = hidden_states[assistant_mask][-1:]
    task_result = task_head(assistant_vectors.float())[0]

    do_generate = force_generate or (task_result[0] < task_result[1])
    if not do_generate:
        return None, "text_route"

    siglip_hidden_states = None
    if siglip_processor is not None and image_paths:
        siglip_hidden_states = preprocess_siglip_pixel_values(
            siglip_model, siglip_processor, image_paths
        )

    with torch.no_grad():
        lvlm_embeds = model(
            inputs.input_ids,
            pixel_values=getattr(inputs, "pixel_values", None),
            attention_mask=inputs.attention_mask,
            image_grid_thw=getattr(inputs, "image_grid_thw", None),
            siglip_hidden_states=siglip_hidden_states,
            output_type="denoise_embeds",
        )

    input_embeds = lvlm_embeds
    t5_prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        prompt if not args.no_joint_with_t5 else "",
        256,
        device,
        1,
    )
    if not args.no_joint_with_t5:
        input_embeds = torch.concat([t5_prompt_embeds, input_embeds], dim=1)

    output_image = pipe(
        prompt_embeds=input_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        height=height,
        width=width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    ).images[0]
    return output_image, "image"


def main(args):
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, task_head, processor = load_main_model_and_processor(
        args.model_path,
        device,
        task_head_path=args.task_head_path,
        attn_implementation=args.attn_implementation,
    )
    pipe, tokenizers, text_encoders = load_pipe(
        model.denoise_tower.denoiser, args.flux_path, device
    )
    siglip_processor, siglip_model = load_siglip_and_processor(
        args.siglip_path, device
    )

    state = {
        "device": device,
        "model": model,
        "task_head": task_head,
        "processor": processor,
        "pipe": pipe,
        "tokenizers": tokenizers,
        "text_encoders": text_encoders,
        "siglip_processor": siglip_processor,
        "siglip_model": siglip_model,
    }

    image, route = generate_t2i_image(
        args, state, args.prompt, force_generate=args.force_generate
    )
    if image is None:
        raise RuntimeError(f"UniWorld routed this prompt to {route}; no image was generated.")
    image.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate one UniWorld-V1 image from a single prompt."
    )

    parser.add_argument("prompt", help="text prompt to generate")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="UniWorld-V1 Hugging Face repo id or local model directory")
    parser.add_argument(
        "--task_head_path",
        type=str,
        default=None,
        help="Optional explicit path to task_head_final.pt",
    )
    parser.add_argument("--flux_path", type=str, default=DEFAULT_FLUX_PATH)
    parser.add_argument("--siglip_path", type=str, default=DEFAULT_SIGLIP_PATH)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="sdpa or flash_attention_2 if installed",
    )
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--no_joint_with_t5", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=str, default=None, help="full output image path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUT,
        help="directory used with --filename",
    )
    parser.add_argument("--filename", type=str, default="output.png")
    parser.add_argument(
        "--force_generate",
        action="store_true",
        default=True,
        help="Always run image branch for the prompt (default: True)",
    )
    parser.add_argument(
        "--no_force_generate",
        action="store_false",
        dest="force_generate",
        help="Use task_head routing instead of always generating",
    )

    args = parser.parse_args()
    main(args)
