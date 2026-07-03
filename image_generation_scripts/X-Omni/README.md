# X-Omni single-prompt generation

This folder contains a one-prompt image generation entry point for `infer.py`. Run commands from this folder unless noted otherwise.

## Setup

1. Create or activate a Python environment with PyTorch, CUDA, and X-Omni dependencies.
2. Install the local dependency file when starting from a fresh environment:

```bash
pip install -r requirements.txt
```

3. The official repo recommends flash-attn for best performance. The local script defaults to `--attn-implementation sdpa` so it can run without flash-attn. Use `--attn-implementation flash_attention_2` only after installing flash-attn.
4. X-Omni and FLUX are Hugging Face models. If access is gated, accept the license on Hugging Face and authenticate before the first run:

```bash
huggingface-cli login
```

## Generate One Image

```bash
python infer.py "a cinematic photo of a red chair beside a window"
```

The default output is `outputs/output.png`. To choose the exact output path:

```bash
python infer.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

You can also keep the default output directory and change only the filename:

```bash
python infer.py "a cinematic photo of a red chair beside a window" --output_dir outputs --filename red-chair.png
```

## Model Source

Default Hugging Face model:

```text
X-Omni/X-Omni-En
```

Default FLUX decoder:

```text
black-forest-labs/FLUX.1-dev
```

Hugging Face model pages:

- https://huggingface.co/X-Omni/X-Omni-En
- https://huggingface.co/X-Omni/X-Omni-Zh
- https://huggingface.co/black-forest-labs/FLUX.1-dev

Official project/code: https://github.com/X-Omni-Team/X-Omni

## Useful Options

```bash
--model_name_or_path X-Omni/X-Omni-En --flux_model_name_or_path black-forest-labs/FLUX.1-dev --image-size 1152 1152 --cfg-scale 1.0 --min-p 0.03 --seed 1234
```

Use the Chinese checkpoint when generating Chinese prompts:

```bash
python infer.py "生成一张雪中的紫禁城全景封面图" --model_name_or_path X-Omni/X-Omni-Zh --output ./outputs/chinese-example.png
```

## Notes

- The script accepts one prompt as the positional argument and does not read a JSON prompt file.
- `--image-size 1152` creates a square image. Use `--image-size 1152 768` for height and width.
- Use `--gpu 0` when you want the script to expose a specific GPU as `cuda:0`.
- The official repo examples use `generate.py`; this local `infer.py` keeps the same core generation logic but uses the simplified one-prompt interface used across this codebase.

## Quick Troubleshooting

- If model download fails, confirm that the Hugging Face repo ids above are accessible and that you accepted any required license.
- If CUDA memory is not enough, lower `--image-size` or run on a larger GPU.
- If flash-attn import or loading fails, keep the default `--attn-implementation sdpa`.
- If imports fail, install the requirements for this folder inside the active environment.
