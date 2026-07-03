# Janus single-prompt generation

Generate one image by passing one prompt directly on the command line. The script does not read a prompts JSON file.

```bash
python infer.py "a cinematic photo of a red chair beside a window"
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python infer.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

Common options:

```bash
--model_path ./JANUS-PRO-MODEL --seed 123 --img_size 384 --cfg_weight 5.0 --temperature 1.0
```
For info regarding setting up conda env and other details on image generation refer to offical repo https://github.com/deepseek-ai/janus