# GPT-Image single-prompt generation

Generate one image by passing one prompt directly on the command line. The script does not read a prompts JSON or TSV file.

```bash
python infer.py "a cinematic photo of a red chair beside a window"
```

By default the image is saved as `outputs/output.png`. Override it with either `--output` or `--output_dir` plus `--filename`:

```bash
python infer.py "a cinematic photo of a red chair beside a window" --output ./outputs/red-chair.png
```

Common options:

```bash
--model gpt-image-1.5 --size 1024x1024 --max_retries 5 --retry_wait 10
```

Set `OPENAI_API_KEY`, or pass `--api_key`. Use `--base_url` for an OpenAI-compatible endpoint.
