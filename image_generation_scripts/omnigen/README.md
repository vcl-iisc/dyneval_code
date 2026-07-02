# OmniGen single-prompt generation

Generate one image by passing one prompt directly on the command line. The script does not read a prompts JSON file.

```bash
python image-infer.py "a cinematic photo of a red chair beside a window"
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python image-infer.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

Common options:

```bash
--model_name_or_path ./OMNIGEN-MODEL --seed 123 --height 1024 --width 1024 --guidance_scale 2.5
```

Use `--gpu 0` to expose one GPU as `cuda:0`.
