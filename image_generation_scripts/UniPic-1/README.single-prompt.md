# UniPic single-prompt generation

Generate one image by passing one prompt directly on the command line. The script does not read a prompts JSON file.

```bash
python scripts/infer.py configs/example.py "a cinematic photo of a red chair beside a window" --checkpoint /path/to/pytorch_model.bin
```

By default the image is saved as `outputs/output.jpg`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python scripts/infer.py configs/example.py "a cinematic photo of a red chair beside a window" --checkpoint /path/to/pytorch_model.bin --output ./outputs/red-chair.jpg
```

Common options:

```bash
--seed 42 --image_size 1024 --num_iter 32 --cfg 3.0 --temperature 1.0
```


For more info on setting up conda env or on image generation refer to offifial repo https://github.com/SkyworkAI/UniPic/tree/main/UniPic-1