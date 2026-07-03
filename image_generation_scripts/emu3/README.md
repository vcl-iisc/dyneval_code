# Emu3 single-prompt generation

Generate one image by passing one prompt directly on the command line. The script does not read a prompts JSON file.

```bash
python image-gen.py "a cinematic photo of a red chair beside a window"
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python image-gen.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

For info regarding setting up conda env and other details refer to official repo https://github.com/baaivision/emu3
