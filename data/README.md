# Data Directory

Training expects prepared SFT data under this folder:

```text
data/sft/<dataset_name>/train.jsonl
data/sft/<dataset_name>/val.jsonl
data/sft/<dataset_name>/manifest.json
```

Two builders create these files:

- `data/build_t2ia_sft.py` — `<T2IA>` and `<EVALUATION>` rows from question/answer folders and an image root.
- `data/build_iqa_sft.py` — `<IQA>` rows (scene graph, node-grounded questions, per-question scoring) from the teacher IQA outputs.

**T2IA and EVALUATION:**

```bash
python data/build_t2ia_sft.py \
  --questions-dir /path/to/questions \
  --answers-dir /path/to/answers \
  --images-root /path/to/images \
  --output-dir data/sft/my_dyneval_sft_data \
  --val-ratio 0.05 \
  --seed 42
```

**IQA:**

```bash
python data/build_iqa_sft.py \
  --scene-graph-dir /path/to/iqa_outputs \
  --answers-dir /path/to/iqa_answers \
  --images-root /path/to/images \
  --output-dir data/sft/my_iqa_sft_data \
  --val-ratio 0.05 \
  --seed 42
```

`--scene-graph-dir` is required; `--answers-dir` is optional and adds the per-question scoring rows. Then pass the generated folder to training:

```bash
--data-dir data/sft/my_dyneval_sft_data
```

Large JSONL files and image datasets should not be committed to git. Keep only this structure and small examples/placeholders in the repository.
