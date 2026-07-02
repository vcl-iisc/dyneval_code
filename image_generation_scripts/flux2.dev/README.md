# FLUX.2-dev single-prompt generation

Generate one image by passing one prompt directly on the command line. The script no longer reads a prompts JSON file.

```bash
python dyneval.py "a cinematic photo of a red chair beside a window"
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python dyneval.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

Common options:

```bash
--seed 123 --height 1024 --width 1024 --num_inference_steps 50
```

Use `--device_map single` to force one GPU, or `--remote_encoder` when using the remote text encoder path.
