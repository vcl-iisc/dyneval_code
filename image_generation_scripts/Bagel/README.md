# BAGEL single-prompt generation

Generate one image by passing one prompt directly on the command line. The script does not read a prompts JSON file.

```bash
python scripts/infer-image-gen.py "a cinematic photo of a red chair beside a window" --model-path /path/to/BAGEL-7B-MoT
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output-dir` plus `--filename`:

```bash
python scripts/infer-image-gen.py "a cinematic photo of a red chair beside a window" --model-path /path/to/BAGEL-7B-MoT --output ./outputs/red-chair.png
```
