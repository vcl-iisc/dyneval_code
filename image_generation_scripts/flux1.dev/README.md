# FLUX.1-dev single-prompt generation

Generate one image by passing one prompt directly on the command line. The script no longer reads a prompts JSON file.

```bash
python infer.py "a cinematic photo of a red chair beside a window"
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python infer.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```
Use `--model_name_or_path` to override the default model path or Hugging Face id.
