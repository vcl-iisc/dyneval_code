# LlamaGen single-prompt generation

Generate one image by passing one prompt directly on the command line. The script no longer uses the built-in demo prompt list.

```bash
python autoregressive/sample/image-gen.py "a cinematic photo of a red chair beside a window" --gpt-ckpt /path/to/gpt.ckpt --vq-ckpt /path/to/vq.ckpt
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output-dir` plus `--filename`:

```bash
python autoregressive/sample/image-gen.py "a cinematic photo of a red chair beside a window" --gpt-ckpt /path/to/gpt.ckpt --vq-ckpt /path/to/vq.ckpt --output ./outputs/red-chair.png
```

Common options:

```bash
--seed 123 --image-size 512 --cfg-scale 7.5 --temperature 1.0 --top-k 1000
```
