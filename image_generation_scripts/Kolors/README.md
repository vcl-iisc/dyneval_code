# Kolors single-prompt generation

Generate one image by passing one prompt directly on the command line. The script does not read the prompt map TSV or a prompts JSON file.

```bash
python scripts/image-gen.py "a cinematic photo of a red chair beside a window"
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python scripts/image-gen.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

Common options:

```bash
--seed 66 --height 1024 --width 1024 --num_inference_steps 50 --guidance_scale 5.0
```
