# Show-O single-prompt generation

Generate one image by passing one prompt directly on the command line. The script does not read a prompts JSON or text prompt list.

Show-O still needs its OmegaConf config, so pass it with the existing `config=...` syntax:

```bash
python inference_t2i.py "a cinematic photo of a red chair beside a window" config=configs/example.yaml
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `output=...`:

```bash
python inference_t2i.py "a cinematic photo of a red chair beside a window" config=configs/example.yaml --output ./outputs/red-chair.png
```

Common config overrides:

```bash
seed=123 guidance_scale=7.5 num_inference_steps=50 filename=sample.png
```
