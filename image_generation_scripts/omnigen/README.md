# OmniGen single-prompt generation

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
Shitao/OmniGen-v1
```

Hugging Face model page: https://huggingface.co/Shitao/OmniGen-v1

Official project/code: https://github.com/VectorSpaceLab/OmniGen

## Useful Options

```bash
--model_name_or_path Shitao/OmniGen-v1 --height 1024 --width 1024 --guidance_scale 2.5 --seed 123
```

## Notes
- The script accepts one prompt as the positional argument and does not read a JSON prompt file.
- Use `--gpu 0` when you want the script to expose a specific GPU as `cuda:0`.

## Quick Troubleshooting

- If model download fails, confirm that the Hugging Face repo id above is accessible and that you accepted any required license.
- If CUDA memory is not enough, try the listed CPU/offload option if the script has one, lower image size, or run on a larger GPU.
- If imports fail, install the requirements for this folder inside the active environment.
