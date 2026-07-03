
import argparse
import os
import sys
import time

import numpy as np
import PIL.Image
import torch
from transformers import AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from janus.models import MultiModalityCausalLM, VLChatProcessor

DEFAULT_OUT = "outputs"
DEFAULT_MODEL = "deepseek-ai/Janus-Pro-7B"


_TRIANGULAR_BFLOAT16_PATCHED = False


def patch_bfloat16_triangular_cuda():
    """Work around older PyTorch CUDA builds missing bf16 triu/tril kernels."""
    global _TRIANGULAR_BFLOAT16_PATCHED
    if _TRIANGULAR_BFLOAT16_PATCHED:
        return

    original_triu = torch.triu
    original_tril = torch.tril

    def _supports_bfloat16_cuda_triangular(op):
        if not torch.cuda.is_available():
            return True
        try:
            sample = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
            op(sample)
            return True
        except RuntimeError as exc:
            if "triu_tril_cuda_template" in str(exc) and "BFloat16" in str(exc):
                return False
            raise

    triu_supported = _supports_bfloat16_cuda_triangular(original_triu)
    tril_supported = _supports_bfloat16_cuda_triangular(original_tril)
    if triu_supported and tril_supported:
        _TRIANGULAR_BFLOAT16_PATCHED = True
        return

    def triu(input, diagonal=0, *, out=None):
        if input.is_cuda and input.dtype == torch.bfloat16:
            result = original_triu(input.float(), diagonal=diagonal).to(torch.bfloat16)
            if out is not None:
                return out.copy_(result)
            return result
        return original_triu(input, diagonal=diagonal, out=out)

    def tril(input, diagonal=0, *, out=None):
        if input.is_cuda and input.dtype == torch.bfloat16:
            result = original_tril(input.float(), diagonal=diagonal).to(torch.bfloat16)
            if out is not None:
                return out.copy_(result)
            return result
        return original_tril(input, diagonal=diagonal, out=out)

    torch.triu = triu
    torch.tril = tril
    _TRIANGULAR_BFLOAT16_PATCHED = True


def configure_torch():
    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        patch_bfloat16_triangular_cuda()


def build_prompt(vl_chat_processor, text):
    conversation = [
        {"role": "<|User|>", "content": text},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    return sft_format + vl_chat_processor.image_start_tag


def load_model(model_path):
    configure_torch()
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()
    return vl_gpt, vl_chat_processor


@torch.inference_mode()
def generate_image(
    mmgpt,
    vl_chat_processor,
    prompt_text,
    temperature=1.0,
    cfg_weight=5.0,
    image_token_num_per_image=576,
    img_size=384,
    patch_size=16,
    seed=0,
):
    if seed >= 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    prompt = build_prompt(vl_chat_processor, prompt_text)
    parallel_size = 1

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros(
        (parallel_size, image_token_num_per_image), dtype=torch.int,
    ).cuda()
    outputs = None

    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if i != 0 else None,
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat(
            [next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1,
        ).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    return PIL.Image.fromarray(dec[0])


def main():
    ap = argparse.ArgumentParser(description="Generate one Janus image from a single prompt.")
    ap.add_argument("prompt", help="text prompt to generate")
    ap.add_argument("--output", default=None, help="full output image path")
    ap.add_argument("--output_dir", default=DEFAULT_OUT,
                    help="directory used with --filename")
    ap.add_argument("--filename", default="output.png")
    ap.add_argument("--model_path", default=DEFAULT_MODEL,
                    help="Janus-Pro-7B model path or HF id")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0,
                    help="fixed seed; use -1 for random")
    args = ap.parse_args()

    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"  model={args.model_path}")
    print(f"  cfg_weight={args.cfg_weight}  temperature={args.temperature}"
          f"  img_size={args.img_size}  seed={args.seed}")

    print(f"Loading {args.model_path} ...")
    vl_gpt, vl_chat_processor = load_model(args.model_path)

    start = time.time()
    image = generate_image(
        vl_gpt,
        vl_chat_processor,
        args.prompt,
        temperature=args.temperature,
        cfg_weight=args.cfg_weight,
        img_size=args.img_size,
        seed=args.seed,
    )
    image.save(output)
    print(f"Saved {output} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
