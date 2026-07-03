# Show-O single-prompt generation

This folder contains a one-prompt image generation entry point for `infer.py`. Run commands from this folder unless noted otherwise.

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
python infer.py "a cinematic photo of a red chair beside a window" config=configs/showo2_1.5b_demo_512x512.yaml
```

The default output is `outputs/output.png` unless this README shows a different file extension. To choose the exact output path:

```bash
python infer.py "a cinematic photo of a red chair beside a window" config=configs/showo2_1.5b_demo_512x512.yaml --output ./outputs/red-chair.png
```

You can also keep the default output directory and change only the filename with `--filename` when the script supports it.

## Model Source

Official project:

```text
showlab/show-o
```

Official project/code: https://github.com/showlab/show-o

## Useful Options

```bash
config=configs/showo2_1.5b_demo_512x512.yaml seed=123 guidance_scale=7.5 num_inference_steps=50 filename=sample.png
```

## Notes
- The script accepts one prompt as the first positional argument and does not read a prompt list.
- Show-O still uses OmegaConf, so keep the `config=...` override in the command.
- The script expects `Wan2.1_VAE.pth` in this folder. It prints the exact download command if the file is missing.

## Quick Troubleshooting

- If model download fails, confirm that the Hugging Face repo id above is accessible and that you accepted any required license.
- If CUDA memory is not enough, try the listed CPU/offload option if the script has one, lower image size, or run on a larger GPU.
- If imports fail, install the requirements for this folder inside the active environment.
