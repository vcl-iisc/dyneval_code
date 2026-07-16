# Data Directory

Training expects prepared SFT data under this folder:

```text
data/sft/<dataset_name>/train.jsonl
data/sft/<dataset_name>/val.jsonl
data/sft/<dataset_name>/manifest.json
```

Use `data_building/build_dyneval_sft.py` to create these files from question/answer folders and an image root.

Example:

```bash
python data_building/build_dyneval_sft.py \
  --questions-dir /path/to/questions \
  --answers-dir /path/to/answers \
  --images-root /path/to/images \
  --output-dir data/sft/my_dyneval_sft_data \
  --val-ratio 0.05 \
  --seed 42
```

Then pass the generated folder to training:

```bash
--data-dir data/sft/my_dyneval_sft_data
```

Large JSONL files and image datasets should not be committed to git. Keep only this structure and small examples/placeholders in the repository.
