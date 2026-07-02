# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import multiprocessing as mp
import os
import time
import traceback

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

from PIL import Image
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import get_config, denorm, get_hyper_params, path_to_llm_name, load_state_dict

DEFAULT_DYNEVAL_PROMPTS_FILE = (
    "/mnt/18_TB/shyam/dyneval_project/"
    "DYNEVAL-UPDATED-1K-EXPERIMENT/prompts/DYNEVAL-1K-FIXED-PROMPTS-SHYAM.json"
)
DEFAULT_DYNEVAL_OUTPUT_DIR = (
    "/mnt/18_TB/shyam/dyneval_project/DYNEVAL-1K-IMAGES/SHOW-O"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SHOWO_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = "/mnt/18_TB/shyam/dyneval_project"
DEFAULT_VAE_PATH = os.path.join(SCRIPT_DIR, "Wan2.1_VAE.pth")


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


def resolve_prompts_file(config):
    explicit = config.get("validation_prompts_file")
    if explicit:
        resolved = _find_existing_file(explicit)
        if resolved:
            return resolved
        print(f"[Show-O] WARNING: prompts file not found: {explicit}")

    resolved = _find_existing_file(DEFAULT_DYNEVAL_PROMPTS_FILE)
    if resolved:
        if explicit:
            print(f"[Show-O] Falling back to {resolved}")
        return resolved

    dataset_file = OmegaConf.select(config, "dataset.params.validation_prompts_file")
    resolved = _find_existing_file(dataset_file)
    if resolved:
        return resolved

    raise FileNotFoundError(
        "Could not find a prompts file. "
        f"Expected default at {DEFAULT_DYNEVAL_PROMPTS_FILE}"
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


def load_t2i_prompt_items(prompts_file):
    with open(prompts_file, "r", encoding="utf-8") as f:
        if prompts_file.endswith(".json"):
            data = json.load(f)
            prompt_items = data["prompts"] if isinstance(data, dict) and "prompts" in data else data
            items = [
                {
                    "gid": item.get("gid"),
                    "prompt": item["prompt"],
                    "filename": item.get("filename", f"{idx + 1:04d}.png"),
                }
                for idx, item in enumerate(prompt_items)
            ]
            if items and items[0].get("gid") is not None:
                items.sort(key=lambda item: int(item["gid"]))
            return items

        prompts = [line.strip() for line in f if line.strip()]
        return [
            {"gid": idx + 1, "prompt": prompt, "filename": f"{idx + 1:04d}.png"}
            for idx, prompt in enumerate(prompts)
        ]


def shard(items, n_shards, shard_id):
    return items[shard_id::n_shards]


def parse_gpus(gpu_str):
    if gpu_str:
        return [int(x.strip()) for x in str(gpu_str).split(",") if x.strip() != ""]
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs available.")
    return list(range(torch.cuda.device_count()))


def prepare_config(config):
    if config.get("generation_timesteps", None) is not None:
        config.num_inference_steps = config.generation_timesteps
    if config.get("batch_size", None) is None:
        config.batch_size = 1
    if config.get("guidance_scale", None) is None:
        config.guidance_scale = config.transport.get("guidance_scale", 7.5)
    if config.get("num_inference_steps", None) is None:
        config.num_inference_steps = config.transport.get("num_inference_steps", 50)

    prompts_file = resolve_prompts_file(config)
    config.dataset.params.validation_prompts_file = prompts_file

    output_dir = config.get("output_dir", DEFAULT_DYNEVAL_OUTPUT_DIR)
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


def generate_images(config, runtime, prompt_items, device, tag=""):
    from models.misc import prepare_gen_input

    model = runtime["model"]
    vae_model = runtime["vae_model"]
    text_tokenizer = runtime["text_tokenizer"]
    weight_type = runtime["weight_type"]
    output_dir = config.output_dir
    manifest_path = os.path.join(
        output_dir,
        "generated_manifest.jsonl" if not tag else f"generated_manifest.{tag}.jsonl",
    )
    fail_log = os.path.join(
        output_dir,
        "generate_fail.log" if not tag else f"generate_fail.{tag}.log",
    )

    ok = bad = 0
    prefix = f"[{tag}] " if tag else ""

    for step, start in enumerate(range(0, len(prompt_items), config.batch_size)):
        batch_items = prompt_items[start:start + config.batch_size]
        prompts = [item["prompt"] for item in batch_items]
        t1 = time.time()

        try:
            batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = \
                prepare_gen_input(
                    prompts,
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
                    len(prompts),
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
                block_mask = runtime["omni_attn_mask_naive"](
                    text_tokens.size(0),
                    runtime["max_seq_len"],
                    modality_positions,
                    device,
                ).to(weight_type)
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
            pil_images = [Image.fromarray(image) for image in images]

            for item, image in zip(batch_items, pil_images):
                image.save(os.path.join(output_dir, item["filename"]))

            ok += len(batch_items)
            elapsed = time.time() - t1
            for item in batch_items:
                with open(manifest_path, "a", encoding="utf-8") as mf:
                    mf.write(
                        json.dumps(
                            {
                                "gid": item["gid"],
                                "filename": item["filename"],
                                "prompt": item["prompt"],
                                "seconds": round(elapsed / len(batch_items), 2),
                                "gpu": tag,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                print(
                    f"{prefix}{item['filename']} OK  "
                    f"{elapsed / len(batch_items):.1f}s"
                )
        except Exception as exc:
            bad += len(batch_items)
            for item in batch_items:
                with open(fail_log, "a", encoding="utf-8") as lf:
                    lf.write(f"{item['gid']}\t{item['filename']}\t{item['prompt']}\t{exc}\n")
                    lf.write(traceback.format_exc() + "\n")
                print(f"{prefix}{item['filename']} FAILED: {exc}")

    return ok, bad


def worker_main(gpu_id, pending, config_dict):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.chdir(SCRIPT_DIR)

    tag = f"gpu{gpu_id}"
    config = OmegaConf.create(config_dict)
    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{tag}] worker: {len(pending)} prompts")
    if not pending:
        return 0, 0

    device = torch.device("cuda")
    t0 = time.time()
    print(f"[{tag}] loading models ...")
    runtime = load_models(config, device)
    print(f"[{tag}] models ready in {time.time() - t0:.1f}s")

    ok, bad = generate_images(config, runtime, pending, device, tag=tag)
    print(f"[{tag}] done: generated={ok} failed={bad}")
    return ok, bad


def run_single_gpu(config, prompt_items, gpu_id=None):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Loading models on {device} ...")
    runtime = load_models(config, device)
    return generate_images(config, runtime, prompt_items, device)


if __name__ == "__main__":
    config = prepare_config(get_config())

    prompt_items = load_t2i_prompt_items(config.dataset.params.validation_prompts_file)
    skip_existing = bool(config.get("skip_existing", True))
    if config.get("overwrite", False):
        skip_existing = False

    if config.get("limit") is not None:
        prompt_items = prompt_items[: int(config.limit)]

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"[Show-O] prompts file: {config.dataset.params.validation_prompts_file}")
    print(f"[Show-O] output dir:    {config.output_dir}")
    print(f"[Show-O] total prompts: {len(prompt_items)}")

    if skip_existing:
        before = len(prompt_items)
        prompt_items = [
            item
            for item in prompt_items
            if not os.path.exists(os.path.join(config.output_dir, item["filename"]))
        ]
        print(f"[Show-O] already done: {before - len(prompt_items)}  pending: {len(prompt_items)}")

    if not prompt_items:
        print("Nothing to do.")
        raise SystemExit(0)

    gpus = parse_gpus(config.get("gpus"))
    print(f"[Show-O] gpus: {gpus}")

    if config.get("dry_run", False):
        for rank, gpu in enumerate(gpus):
            part = shard(prompt_items, len(gpus), rank)
            print(f"  GPU {gpu}: {len(part)} prompts")
        for item in prompt_items[:5]:
            print(f"  gid={item['gid']}  {item['filename']}  {item['prompt'][:80]}")
        if len(prompt_items) > 5:
            print(f"  ... and {len(prompt_items) - 5} more")
        raise SystemExit(0)

    config_dict = OmegaConf.to_container(config, resolve=True)

    if len(gpus) > 1:
        jobs = [
            (gpu, shard(prompt_items, len(gpus), rank))
            for rank, gpu in enumerate(gpus)
            if shard(prompt_items, len(gpus), rank)
        ]
        print(f"Launching {len(jobs)} GPU workers ...")
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(jobs)) as pool:
            results = pool.starmap(worker_main, [(gpu, part, config_dict) for gpu, part in jobs])
        ok = sum(r[0] for r in results)
        bad = sum(r[1] for r in results)
    else:
        ok, bad = run_single_gpu(config, prompt_items, gpu_id=gpus[0])

    print(f"\nDone Show-O: generated={ok}  failed={bad}  -> {config.output_dir}")
    if bad:
        print("Check generate_fail*.log in output_dir")
