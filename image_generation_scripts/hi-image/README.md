# HiDream Image single-prompt generation

This folder contains a one-prompt image generation entry point for `dyneval.py`. Run commands from this folder unless noted otherwise.

## Setup

1. Create or activate a Python environment with PyTorch, CUDA, and this model's dependencies.
2. Install the dependencies required by this script, usually `torch`, `diffusers`, `transformers`, `accelerate`, and `Pillow` for local diffusion models.
3. If the model is gated or private, authenticate before the first run:

```bash
huggingface-cli login
```

API-backed folders use their provider API key instead of Hugging Face login; see the notes below for those cases.

## Generate One Image

```bash
python dyneval.py "a cinematic photo of a red chair beside a window"
```

The default output is `outputs/output.png` unless this README shows a different file extension. To choose the exact output path:

```bash
python dyneval.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

You can also keep the default output directory and change only the filename with `--filename` when the script supports it.

## Model Source

Default Hugging Face model:

```text
HiDream-ai/HiDream-I1-Full
```

Hugging Face model page: https://huggingface.co/HiDream-ai/HiDream-I1-Full

## Useful Options

```bash
--model_path HiDream-ai/HiDream-I1-Full --model_type full --llama_path meta-llama/Meta-Llama-3.1-8B-Instruct --height 1024 --width 1024 --seed 123
```

## Notes
- The script accepts one prompt as the positional argument and does not read a JSON prompt file.
- `--model_type` can be `full`, `dev`, or `fast`.
- The Llama dependency defaults to `meta-llama/Meta-Llama-3.1-8B-Instruct`; use a local path only if you already downloaded it.

## Quick Troubleshooting

- If model download fails, confirm that the Hugging Face repo id above is accessible and that you accepted any required license.
- If CUDA memory is not enough, try the listed CPU/offload option if the script has one, lower image size, or run on a larger GPU.
- If imports fail, install the requirements for this folder inside the active environment.
