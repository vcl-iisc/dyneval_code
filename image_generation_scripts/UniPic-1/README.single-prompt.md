# UniPic-1 single-prompt generation

This folder contains a one-prompt image generation entry point for `scripts/infer.py`. Run commands from this folder unless noted otherwise.

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
python scripts/infer.py configs/unipic_1.5b.yaml "a cinematic photo of a red chair beside a window"
```

The default output is `outputs/output.png` unless this README shows a different file extension. To choose the exact output path:

```bash
python scripts/infer.py configs/unipic_1.5b.yaml "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.jpg
```

You can also keep the default output directory and change only the filename with `--filename` when the script supports it.

## Model Source

Default Hugging Face model:

```text
Skywork/Skywork-UniPic-1.5B
```

Hugging Face model page: https://huggingface.co/Skywork/Skywork-UniPic-1.5B

## Useful Options

```bash
--checkpoint Skywork/Skywork-UniPic-1.5B --image_size 1024 --num_iter 32 --cfg 3.0 --seed 42
```

## Notes
- The script still needs a UniPic config file first, then one prompt as the second positional argument.
- When `--checkpoint` is a Hugging Face repo id, the script downloads the snapshot and resolves the checkpoint file automatically.

## Quick Troubleshooting

- If model download fails, confirm that the Hugging Face repo id above is accessible and that you accepted any required license.
- If CUDA memory is not enough, try the listed CPU/offload option if the script has one, lower image size, or run on a larger GPU.
- If imports fail, install the requirements for this folder inside the active environment.
