# UniWorld-V1 single-prompt generation

Generate one image by passing one prompt directly on the command line. The script does not read a prompts JSON file.

```bash
python univa/serve/infer.py "a cinematic photo of a red chair beside a window" --model_path /path/to/model --flux_path /path/to/flux --siglip_path /path/to/siglip
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python univa/serve/infer.py "a cinematic photo of a red chair beside a window" --model_path /path/to/model --flux_path /path/to/flux --siglip_path /path/to/siglip --output ./outputs/red-chair.png
```

Common options:

```bash
--seed 42 --height 1024 --width 1024 --num_inference_steps 28 --guidance_scale 3.5
```

Use `--task_head_path` when `task_head_final.pt` is not located under the model path.

For more info on setting up conda env or on image generation refer to offifial repo https://github.com/PKU-YuanGroup/UniWorld/tree/main/UniWorld-V1