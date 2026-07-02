
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

from PIL import Image
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import get_config, denorm, get_hyper_params, path_to_llm_name, load_state_dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHOWO_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = "/mnt/18_TB/shyam/dyneval_project"
DEFAULT_VAE_PATH = os.path.join(SCRIPT_DIR, "Wan2.1_VAE.pth")
DEFAULT_OUT = "outputs"


def _find_existing_file(path):
    if not path:
        return None
    if os.path.isfile(path):
        return os.path.abspath(path)
    for root in (SCRIPT_DIR, SHOWO_DIR, PROJECT_ROOT, os.getcwd()):
        resolved = os.path.join(root, path)
        if os.path.isfile(resolved):
            return os.path.abspath(resolved)
    return None


def resolve_vae_model_path(config):
    configured = config.get("vae_model_path")
    if configured is None:
        configured = OmegaConf.select(config, "model.vae_model.pretrained_model_path")

    if configured and os.path.isabs(configured) and os.path.isfile(configured):
        return configured

    for path in (DEFAULT_VAE_PATH, os.path.join(SHOWO_DIR, "Wan2.1_VAE.pth")):
        if os.path.isfile(path):
            return os.path.abspath(path)

    resolved = _find_existing_file(configured)
    if resolved:
        return resolved

    raise FileNotFoundError(
        "Wan VAE weights were not found. Download with:\n"
        "  cd /mnt/18_TB/shyam/dyneval_project/show-o/show-o2\n"
        "  wget https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.pth\n"
        f"Expected file at {DEFAULT_VAE_PATH}"
    )


def sync_config_with_model(config, model):
    """Keep YAML resolution settings for HQ checkpoints."""
    ckpt_lh = int(model.config.image_latent_height)
    ckpt_lw = int(model.config.image_latent_width)
    yaml_lh = int(OmegaConf.select(config, "model.showo.image_latent_height"))
    yaml_lw = int(OmegaConf.select(config, "model.showo.image_latent_width"))
    if (yaml_lh, yaml_lw) != (ckpt_lh, ckpt_lw):
        print(
            f"[Show-O] HQ/runtime resolution {yaml_lh}x{yaml_lw} "
            f"(checkpoint native {ckpt_lh}x{ckpt_lw}; using interpolated pos embeds)"
        )
        config.dataset.preprocessing.latent_height = yaml_lh
        config.dataset.preprocessing.latent_width = yaml_lw
        config.dataset.preprocessing.num_t2i_image_tokens = yaml_lh * yaml_lw
        return

    config.model.showo.image_latent_height = ckpt_lh
    config.model.showo.image_latent_width = ckpt_lw
    config.dataset.preprocessing.latent_height = ckpt_lh
    config.dataset.preprocessing.latent_width = ckpt_lw
    config.dataset.preprocessing.num_t2i_image_tokens = ckpt_lh * ckpt_lw
    patch_size = int(config.model.showo.patch_size)
    config.dataset.preprocessing.resolution = ckpt_lh * patch_size * 8


def prepare_config(config):
    if config.get("generation_timesteps", None) is not None:
        config.num_inference_steps = config.generation_timesteps
    if config.get("batch_size", None) is None:
        config.batch_size = 1
    if config.get("guidance_scale", None) is None:
        config.guidance_scale = config.transport.get("guidance_scale", 7.5)
    if config.get("num_inference_steps", None) is None:
        config.num_inference_steps = config.transport.get("num_inference_steps", 50)

    output_dir = config.get("output_dir", DEFAULT_OUT)
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)
    config.output_dir = output_dir

    vae_model_path = resolve_vae_model_path(config)
    config.model.vae_model.pretrained_model_path = vae_model_path
    return config


def load_models(config, device):
    from models import Showo2Qwen2_5, WanVAE
    from models.misc import get_text_tokenizer, prepare_gen_input
    from models import omni_attn_mask_naive
    from transport import Sampler, create_transport

    if config.model.weight_type == "bfloat16":
        weight_type = torch.bfloat16
    elif config.model.weight_type == "float32":
        weight_type = torch.float32
    else:
        raise NotImplementedError

    vae_model = WanVAE(
        vae_pth=config.model.vae_model.pretrained_model_path,
        dtype=weight_type,
        device=device,
    )

    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path,
            use_safetensors=False,
            low_cpu_mem_usage=False,
        ).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)

    model.to(weight_type)
    model.eval()
    sync_config_with_model(config, model)

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, _ = get_hyper_params(
        config, text_tokenizer, showo_token_ids
    )

    guidance_scale = config.guidance_scale
    config.transport.num_inference_steps = config.num_inference_steps

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=num_t2i_image_tokens,
    )
    sampler = Sampler(transport)

    return {
        "model": model,
        "vae_model": vae_model,
        "text_tokenizer": text_tokenizer,
        "showo_token_ids": showo_token_ids,
        "weight_type": weight_type,
        "num_t2i_image_tokens": num_t2i_image_tokens,
        "max_seq_len": max_seq_len,
        "max_text_len": max_text_len,
        "image_latent_dim": image_latent_dim,
        "patch_size": patch_size,
        "latent_width": latent_width,
        "latent_height": latent_height,
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "boi_id": boi_id,
        "eoi_id": eoi_id,
        "img_pad_id": img_pad_id,
        "guidance_scale": guidance_scale,
        "sampler": sampler,
        "prepare_gen_input": prepare_gen_input,
        "omni_attn_mask_naive": omni_attn_mask_naive,
    }


def generate_single_image(config, runtime, prompt, device):
    from models.misc import prepare_gen_input

    model = runtime["model"]
    vae_model = runtime["vae_model"]
    text_tokenizer = runtime["text_tokenizer"]
    weight_type = runtime["weight_type"]

    batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
        prepare_gen_input(
            [prompt],
            text_tokenizer,
            runtime["num_t2i_image_tokens"],
            runtime["bos_id"],
            runtime["eos_id"],
            runtime["boi_id"],
            runtime["eoi_id"],
            runtime["pad_id"],
            runtime["img_pad_id"],
            runtime["max_text_len"],
            device,
        )

    z = torch.randn(
        (
            1,
            runtime["image_latent_dim"],
            runtime["latent_height"] * runtime["patch_size"],
            runtime["latent_width"] * runtime["patch_size"],
        ),
        device=device,
        dtype=torch.bfloat16,
    )

    guidance_scale = runtime["guidance_scale"]
    if guidance_scale > 0:
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat(
            [batch_modality_positions, batch_modality_positions_null], dim=0
        )
    else:
        text_tokens = batch_text_tokens
        modality_positions = batch_modality_positions

    block_mask = runtime["omni_attn_mask_naive"](
        text_tokens.size(0),
        runtime["max_seq_len"],
        modality_positions,
        device,
    ).to(weight_type)

    model_kwargs = dict(
        text_tokens=text_tokens,
        attention_mask=block_mask,
        modality_positions=modality_positions,
        output_hidden_states=True,
        max_seq_len=runtime["max_seq_len"],
        guidance_scale=guidance_scale,
    )

    sample_fn = runtime["sampler"].sample_ode(
        sampling_method=config.transport.sampling_method,
        num_steps=config.transport.num_inference_steps,
        atol=config.transport.atol,
        rtol=config.transport.rtol,
        reverse=config.transport.reverse,
        time_shifting_factor=config.transport.time_shifting_factor,
    )
    samples = sample_fn(z, model.t2i_generate, **model_kwargs)[-1]
    if guidance_scale > 0:
        samples = torch.chunk(samples, 2)[0]

    samples = samples.unsqueeze(2)
    images = vae_model.batch_decode(samples).squeeze(2)
    images = denorm(images)
    return Image.fromarray(images[0])


def normalize_cli_args(argv):
    """Support one positional prompt while preserving OmegaConf key=value overrides."""
    prompt = None
    normalized = []
    value_flags = {
        "--config": "config",
        "--prompt": "prompt",
        "--output": "output",
        "--output-dir": "output_dir",
        "--output_dir": "output_dir",
        "--filename": "filename",
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in value_flags:
            if i + 1 >= len(argv):
                raise SystemExit(f"{arg} requires a value")
            key = value_flags[arg]
            value = argv[i + 1]
            if key == "prompt":
                prompt = value
            else:
                normalized.append(f"{key}={value}")
            i += 2
            continue

        matched_flag = False
        for flag, key in value_flags.items():
            prefix = f"{flag}="
            if arg.startswith(prefix):
                value = arg[len(prefix):]
                if key == "prompt":
                    prompt = value
                else:
                    normalized.append(f"{key}={value}")
                matched_flag = True
                break
        if matched_flag:
            i += 1
            continue

        if arg.startswith("--"):
            normalized.append(arg[2:])
        elif "=" in arg:
            normalized.append(arg)
        elif prompt is None:
            prompt = arg
        else:
            raise SystemExit(f"Unexpected extra positional argument: {arg}")
        i += 1

    return prompt, normalized


if __name__ == "__main__":
    prompt, omega_args = normalize_cli_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + omega_args
    config = prepare_config(get_config())
    if prompt is None:
        prompt = config.get("prompt")
    if not prompt:
        raise SystemExit(
            'Missing prompt. Example: python inference_t2i.py "a red chair" config=configs/example.yaml'
        )

    if config.get("seed") is not None:
        torch.manual_seed(int(config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.seed))

    output = config.get("output")
    if not output:
        output = os.path.join(config.output_dir, config.get("filename", "output.png"))
    if not os.path.isabs(output):
        output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Show-O] output: {output}")
    print(f"[Show-O] loading models on {device} ...")
    runtime = load_models(config, device)
    image = generate_single_image(config, runtime, prompt, device)
    image.save(output)
    print(f"Saved {output}")
