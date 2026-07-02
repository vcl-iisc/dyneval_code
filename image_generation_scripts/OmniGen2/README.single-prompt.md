# OmniGen2 single-prompt generation

Generate one image by passing one prompt directly on the command line.

```bash
python inference.py "a cinematic photo of a red chair beside a window" --model_path /path/to/OmniGen2
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python inference.py "a cinematic photo of a red chair beside a window" --model_path /path/to/OmniGen2 --output ./outputs/red-chair.png
```

Common options:

```bash
--seed 123 --height 1024 --width 1024 --num_inference_step 50 --text_guidance_scale 5.0
```

Use `--input_image_path image.png` for image-conditioned generation. Existing offload/cache options such as `--enable_model_cpu_offload`, `--enable_group_offload`, `--enable_teacache`, and `--enable_taylorseer` are still available.
