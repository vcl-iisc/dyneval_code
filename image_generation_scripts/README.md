# Image generation scripts

This directory contains one folder per image generation backend. Most folders expose a single-prompt command: pass one text prompt directly on the command line, and the script writes one image to `outputs/output.png` by default.

The old dataset-style JSON prompt flow has been removed from the converted scripts. If a folder still needs extra inputs, its local README explains that explicitly.

## Standard workflow

1. Pick the model folder you want to run.
2. Read that folder's `README.md` or `README.single-prompt.md`.
3. Create or activate an environment with PyTorch/CUDA and that model's dependencies.
4. Install `requirements.txt` when the folder provides one:

```bash
pip install -r requirements.txt
```

5. Log in to Hugging Face before using gated or private models:

```bash
huggingface-cli login
```

6. Run the example command from the folder README with your own prompt.

Most converted scripts follow this shape:

```bash
python infer.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

Some folders use a different script name, for example `dyneval.py`, `image-gen.py`, or `inference.py`. Use the command shown in the folder README.

## Outputs

By default, converted scripts save to:

```text
outputs/output.png
```

Use `--output ./outputs/name.png` when you want an exact file path. Many scripts also support `--output_dir` or `--output-dir` plus `--filename`.

## Model sources

Folder READMEs list the default model source using the actual Hugging Face repo id or API model name. Prefer those repo ids over local placeholder text.

If a model is gated, first accept the model license on Hugging Face in a browser, then run `huggingface-cli login` in the environment.

API-backed folders do not use Hugging Face login:

- `GPT-Image` uses `OPENAI_API_KEY` or `--api_key`.
- `nanobanana` uses `GOOGLE_API_KEY`, `GEMINI_API_KEY`, or `--api_key`.

## Folder index

| Folder | README | Entry point | Default model/source |
| --- | --- | --- | --- |
| `Bagel` | `README.md` | `scripts/infer-image-gen.py` | `ByteDance-Seed/BAGEL-7B-MoT` |
| `FIBO` | `README.md` | `infer.py` | `briaai/FIBO` |
| `GLM-Image` | `README.md` | `infer.py` | `zai-org/GLM-Image` |
| `GPT-Image` | `README.md` | `infer.py` | `gpt-image-1.5` |
| `HunyuanDiT` | `README.md` | `infer.py` | `Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers` |
| `IF-XL` | `README.md` | `infer.py` | `DeepFloyd/IF-I-XL-v1.0` |
| `Kolors` | `README.md` | `scripts/image-gen.py` | `Kwai-Kolors/Kolors` |
| `OmniGen2` | `README.single-prompt.md` | `inference.py` | `VectorSpaceLab/OmniGen2` |
| `UniPic-1` | `README.single-prompt.md` | `scripts/infer.py` | `Skywork/Skywork-UniPic-1.5B` |
| `UniWorld-V1` | `README.single-prompt.md` | `univa/serve/infer.py` | `PKU-YuanGroup/UniWorld-V1` |
| `X-Omni` | `README.md` | `infer.py` | `X-Omni/X-Omni-En` plus `black-forest-labs/FLUX.1-dev` |
| `emu3` | `README.md` | `image-gen.py` | `BAAI/Emu3-Gen` |
| `flux1.dev` | `README.md` | `infer.py` | `black-forest-labs/FLUX.1-dev` |
| `flux2.dev` | `README.md` | `dyneval.py` | `black-forest-labs/FLUX.2-dev` |
| `flux2.klein` | `README.md` | `dyneval.py` | `black-forest-labs/FLUX.2-klein-9B` |
| `hi-image` | `README.md` | `dyneval.py` | `HiDream-ai/HiDream-I1-Full` |
| `incontext-llora` | `README.md` | `infer.py` | `black-forest-labs/FLUX.1-dev` plus `ali-vilab/In-Context-LoRA` |
| `janus` | `README.md` | `infer.py` | `deepseek-ai/Janus-Pro-7B` |
| `kandinsky3` | `README.md` | `infer.py` | see folder README |
| `llamagen` | `README.single-prompt.md` | `autoregressive/sample/image-gen.py` | `FoundationVision/LlamaGen` |
| `longcat` | `README.md` | `infer.py` | `meituan-longcat/LongCat-Image` |
| `nanobanana` | `README.md` | `infer.py` | `gemini-3.1-flash-image-preview` |
| `omnigen` | `README.md` | `image-infer.py` | `Shitao/OmniGen-v1` |
| `pixart-alpha` | `README.md` | `infer.py` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` |
| `pixart-sigma` | `README.md` | `infer.py` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` |
| `playground` | `README.md` | `infer.py` | `playgroundai/playground-v2.5-1024px-aesthetic` |
| `qwen-image` | `README.md` | `infer.py` | `Qwen/Qwen-Image` |
| `sana` | `README.md` | `infer.py` | `Efficient-Large-Model/Sana_1600M_1024px` |
| `sd3.5` | `README.md` | `infer.py` | `stabilityai/stable-diffusion-3.5-large` |
| `sdv1.5` | `README.md` | `infer.py` | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| `sdv2.1` | `README.md` | `infer.py` | `sd2-community/stable-diffusion-2-1` |
| `sdxl` | `README.md` | `infer.py` | `stabilityai/stable-diffusion-xl-base-1.0` |
| `sdxl-turbo` | `README.md` | `infer.py` | `stabilityai/sdxl-turbo` |
| `show-o` | `README.single-prompt.md` | `inference_t2i.py` | official Show-O config/checkpoints |
| `ssd1b` | `README.md` | `infer.py` | `segmind/SSD-1B` |
| `z-image` | `README.md` | `infer.py` | `Tongyi-MAI/Z-Image` |

## Troubleshooting

- If a script cannot download a model, check that the Hugging Face repo exists, the license is accepted, and the environment is logged in.
- If CUDA runs out of memory, lower image size, use the script's CPU/offload option if available, or run on a larger GPU.
- If a script cannot import local modules, run it from the folder shown in that folder's README.
- If an API-backed script fails authentication, confirm the correct provider API key is exported in the same shell.
