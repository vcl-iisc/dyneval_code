import argparse
import json
import os
import sys
import traceback

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
from univa.utils.get_ocr import get_ocr_result


seed = 42
set_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

generate_image_temp = "./generate_image_{}.png"

DEFAULT_BATCH_JSON = "DYNEVAL-1K-rem-36-prompts.json"
DEFAULT_OUTPUT_DIR = os.path.join("DYNEVAL-1K-IMAGES", "UniWorld-V1")
LOCAL_BATCH_JSON_NAMES = (
    "DYNEVAL-1K-rem-36-prompts.json",
    "missing_prompts.json",
    "dyneval-prompts-remaining-univa.json",
)


def resolve_batch_json(explicit_path=None):
    """Find the DYNEVAL missing-prompts JSON on this machine."""
    if explicit_path:
        if os.path.isfile(explicit_path):
            return os.path.abspath(explicit_path)
        raise FileNotFoundError(f"--batch_json not found: {explicit_path}")

    env_path = os.environ.get("DYNEVAL_BATCH_JSON")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    candidates = [DEFAULT_BATCH_JSON]
    candidates.extend(os.path.join(os.getcwd(), name) for name in LOCAL_BATCH_JSON_NAMES)
    candidates.extend(
        os.path.join(_REPO_ROOT, name) for name in LOCAL_BATCH_JSON_NAMES
    )

    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def resolve_output_dir(explicit_path, batch_json_path):
    """Use explicit path, main-machine DYNEVAL folder, or ./uniworld_outputs."""
    if explicit_path:
        return os.path.abspath(explicit_path)
    if os.path.isdir(os.path.dirname(DEFAULT_OUTPUT_DIR)):
        return DEFAULT_OUTPUT_DIR
    return os.path.abspath(os.path.join(os.path.dirname(batch_json_path), "uniworld_outputs"))


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
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]
    return output_image, "image"


def run_batch_dyneval(args):
    """Generate DYNEVAL-1K missing UniWorld-V1 images from JSON prompt list."""
    with open(args.batch_json, encoding="utf-8") as f:
        doc = json.load(f)

    prompts = sorted(doc.get("prompts", doc), key=lambda p: int(p["gid"]))
    os.makedirs(args.output_dir, exist_ok=True)

    pending = []
    for item in prompts:
        out_path = os.path.join(args.output_dir, item["filename"])
        if os.path.exists(out_path) and not args.overwrite:
            continue
        pending.append(item)

    print(f"[DYNEVAL batch] source={args.batch_json}")
    print(f"  total_in_json={len(prompts)}")
    print(f"  output_dir={args.output_dir}")
    print(f"  already_done={len(prompts) - len(pending)}  to_generate={len(pending)}")
    if args.limit:
        pending = pending[: args.limit]
        print(f"  limited to {len(pending)} this run")

    if not pending:
        print("Nothing to do.")
        return

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

    fail_log = os.path.join(args.output_dir, "generate_fail.log")
    manifest_path = os.path.join(args.output_dir, "generated_manifest.jsonl")
    ok = bad = skipped_text = 0

    for i, item in enumerate(pending, 1):
        gid = item["gid"]
        filename = item["filename"]
        prompt = item["prompt"]
        out_path = os.path.join(args.output_dir, filename)

        try:
            image, route = generate_t2i_image(
                args, state, prompt, force_generate=args.force_generate
            )
            if image is None:
                skipped_text += 1
                with open(fail_log, "a", encoding="utf-8") as lf:
                    lf.write(f"{filename}\tTEXT_ROUTE\t{prompt}\n")
                print(f"[{i}/{len(pending)}] {filename} SKIPPED (text route)")
                continue

            image.save(out_path)
            ok += 1
            record = {
                "gid": gid,
                "filename": filename,
                "benchmark": item.get("benchmark"),
                "prompt": prompt,
            }
            with open(manifest_path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(pending)}] {filename} OK  gid={gid}  [{item.get('benchmark', '')}]")

        except Exception as e:
            bad += 1
            with open(fail_log, "a", encoding="utf-8") as lf:
                lf.write(f"{filename}\t{e}\t{prompt}\n")
                lf.write(traceback.format_exc() + "\n")
            print(f"[{i}/{len(pending)}] {filename} FAILED: {e}")

    dest_note = (
        "Saved directly to DYNEVAL-1K-IMAGES/UniWorld-V1/."
        if os.path.abspath(args.output_dir) == os.path.abspath(DEFAULT_OUTPUT_DIR)
        else f"Images -> {args.output_dir}\nCopy them to DYNEVAL-1K-IMAGES/UniWorld-V1/ on the main machine."
    )
    print(
        f"\nDone. generated={ok}  text_route={skipped_text}  failed={bad}\n"
        f"{dest_note}"
    )


def run_interactive(args):
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

    cur_ocr_i = 0
    cur_genimg_i = 0
    history_image_paths = []
    conversation = []

    print("Interactive UniWorld-V1 Chat (Exit if input is empty)")
    while True:
        txt = input("Text prompt (or press Enter to skip): ").strip()
        img_input = input(
            "Image URLs (comma-separated, or press Enter to skip): "
        ).strip()

        if not img_input and not txt:
            print("Exit.")
            break

        content = []
        urls = []
        if img_input:
            urls = [u.strip() for u in img_input.split(",") if u.strip()]

        if txt:
            if args.ocr_enhancer and urls:
                ocr_parts = []
                for url in urls:
                    ocr_parts.append(get_ocr_result(url, cur_ocr_i))
                    cur_ocr_i += 1
                txt = txt + "\n".join(ocr_parts)
            content.append({"type": "text", "text": txt})

        new_h, new_w = args.height, args.width
        if urls:
            for url in urls:
                content.append(
                    {
                        "type": "image",
                        "image": url,
                        "min_pixels": 448 * 448,
                        "max_pixels": 448 * 448,
                    }
                )
                history_image_paths.append(url)
            new_h, new_w = update_size(
                urls[0] if len(urls) > 0 else None,
                urls[1] if len(urls) > 1 else None,
                "any_11ratio",
                anchor_pixels=args.height * args.width,
            )

        conversation.append({"role": "user", "content": content})
        print("conversation:\n", conversation)

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

        if task_result[0] < task_result[1]:
            siglip_hidden_states = None
            if siglip_processor is not None and history_image_paths:
                siglip_hidden_states = preprocess_siglip_pixel_values(
                    siglip_model, siglip_processor, history_image_paths
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
                txt if not args.no_joint_with_t5 else "",
                256,
                device,
                1,
            )
            if not args.no_joint_with_t5:
                input_embeds = torch.concat([t5_prompt_embeds, input_embeds], dim=1)

            output_image = pipe(
                prompt_embeds=input_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                height=new_h,
                width=new_w,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            ).images[0]
            img_url = generate_image_temp.format(cur_genimg_i)
            cur_genimg_i += 1
            output_image.save(img_url)
            conversation.append(
                {"role": "assistant", "content": [{"type": "image", "image": img_url}]}
            )
            history_image_paths.append(img_url)
            print(f"Assistant: generate image at {img_url}\n")
        else:
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            reply = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            print(f"Assistant: {reply}\n")
            conversation.append(
                {"role": "assistant", "content": [{"type": "text", "text": reply}]}
            )


def main(args):
    if args.interactive:
        run_interactive(args)
        return

    batch_json = resolve_batch_json(args.batch_json)
    if not batch_json:
        sys.exit(
            "DYNEVAL batch JSON not found.\n"
            "Place UniWorld-V1-MISSING-PROMPTS.json in the current directory, or pass:\n"
            "  --batch_json /path/to/UniWorld-V1-MISSING-PROMPTS.json\n"
            "For chat mode instead, pass --interactive"
        )

    args.batch_json = batch_json
    args.output_dir = resolve_output_dir(args.output_dir, batch_json)
    run_batch_dyneval(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="UniWorld-V1 DYNEVAL batch generation (default) or interactive chat"
    )

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--task_head_path",
        type=str,
        default=None,
        help="Optional explicit path to task_head_final.pt",
    )
    parser.add_argument("--flux_path", type=str, required=True)
    parser.add_argument("--siglip_path", type=str, required=True)
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
    parser.add_argument("--ocr_enhancer", action="store_true")
    parser.add_argument("--no_joint_with_t5", action="store_true")

    # Default: DYNEVAL batch (367 missing prompts). Use --interactive for chat.
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive chat instead of DYNEVAL batch generation",
    )
    parser.add_argument(
        "--batch_json",
        type=str,
        default=None,
        help="Prompt JSON (auto-detected: ./UniWorld-V1-MISSING-PROMPTS.json or DYNEVAL_BATCH_JSON env)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Save gid-named png files here (default: ./uniworld_outputs next to JSON, or DYNEVAL-1K-IMAGES/UniWorld-V1/)",
    )
    parser.add_argument(
        "--dyneval",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Generate at most N images (test)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if output file already exists",
    )
    parser.add_argument(
        "--force_generate",
        action="store_true",
        default=True,
        help="Always run image branch for batch prompts (default: True)",
    )
    parser.add_argument(
        "--no_force_generate",
        action="store_false",
        dest="force_generate",
        help="Use task_head routing instead of always generating",
    )

    args = parser.parse_args()
    main(args)
