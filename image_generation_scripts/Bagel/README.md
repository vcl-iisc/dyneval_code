# BAGEL single-prompt generation

This folder contains a one-prompt image generation entry point for `infer.py`. Run commands from this folder unless noted otherwise.

## Setup

1. Create or activate a Python environment with PyTorch, CUDA, and this model's dependencies.
2. Install the local dependency file when starting from a fresh environment:

```bash
pip install -r requirements.txt
```

3. If the model is gated or private, authenticate before the first run:

```bash
huggingface-cli login
```

API-backed folders use their provider API key instead of Hugging Face login; see the notes below for those cases.

## Generate One Image

```bash
python infer.py "a cinematic photo of a red chair beside a window"
```

The default output is `outputs/output.png` unless this README shows a different file extension. To choose the exact output path:

```bash
python infer.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

You can also keep the default output directory and change only the filename with `--filename` when the script supports it.

## Model Source

Default Hugging Face model:

```text
ByteDance-Seed/BAGEL-7B-MoT
```

Hugging Face model page: https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT

Official project/code: https://github.com/ByteDance-Seed/Bagel/

## Useful Options

```bash
--model-path ByteDance-Seed/BAGEL-7B-MoT --image-size 1024 --num-timesteps 50 --seed 42
```

## Notes
- The script accepts one prompt as the positional argument and does not read a JSON prompt file.
- If the Hugging Face repo is used, the script downloads the snapshot before loading local BAGEL files.
- Use `--use-flash-attn` only when flash-attn is installed in the environment.

## Quick Troubleshooting

- If model download fails, confirm that the Hugging Face repo id above is accessible and that you accepted any required license.
- If CUDA memory is not enough, try the listed CPU/offload option if the script has one, lower image size, or run on a larger GPU.
- If imports fail, install the requirements for this folder inside the active environment.
