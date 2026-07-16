# DynEval Qwen3-VL Training

This training code does **not** use the older `Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py` dataset-registration flow.

Instead, training uses prebuilt SFT JSONL files:

```text
data/sft/<dataset_name>/train.jsonl
data/sft/<dataset_name>/val.jsonl
```

Each JSONL row already contains the full chat messages for one task. The trainer only needs:

```bash
--data-dir data/sft/<dataset_name>
```

## Task Tokens

The training script registers these three task tokens at startup:

```text
<T2IA>
<IQA>
<EVALUATION>
```

Training uses:

- `<T2IA>` for text-to-image alignment element extraction and visual question generation.
- `<IQA>` for image-quality assessment question generation, including scene/quality checks.
- `<EVALUATION>` for image-based answering and 1--5 scoring of generated questions.

## Training Data Format

The expected data directory must contain:

```text
train.jsonl
val.jsonl
```

Each row is a JSON object. Important fields:

```json
{
  "id": "sample_id",
  "task": "t2ia_element_extraction",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [{"type": "text", "text": "<T2IA> ..."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]
}
```

For IQA question-generation rows, include the image and use `<IQA>` in the user instruction:

```json
{
  "id": "sample_id",
  "task": "iqa_question_generation",
  "image_path": "/path/to/image.png",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "<IQA> ..."}
      ]
    },
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]
}
```

For image evaluation rows, include an image path:

```json
{
  "id": "sample_id",
  "task": "evaluation",
  "image_path": "/path/to/image.png",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "<EVALUATION> ..."}
      ]
    },
    {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
  ]
}
```

## Build SFT Data

Use `data/build_dyneval_sft.py` for DynEval-style folders that contain:

- a question folder
- an answer folder
- an image root

Example:

```bash
python data/build_dyneval_sft.py \
  --questions-dir /path/to/questions \
  --answers-dir /path/to/answers \
  --images-root /path/to/images \
  --output-dir data/sft/my_dyneval_sft_data \
  --val-ratio 0.05 \
  --seed 42
```

This creates mixed SFT data for:

- `<T2IA>` element extraction
- `<T2IA>` single-question generation
- `<IQA>` image-quality question generation
- `<EVALUATION>` image scoring


## Full Fine-Tuning 2B

Example for continuing from a local 2B DynEval checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 training/train_qwen3vl_dyneval.py \
  --train-only \
  --finetune-mode full \
  --model-path /path/to/qwen3vl-2b-checkpoint \
  --data-dir data/sft/my_dyneval_sft_data \
  --output-dir checkpoints/final/qwen3vl-2b-dyneval-new-v1 \
  --device-map none \
  --gradient-checkpointing \
  --ddp-find-unused-parameters \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --lr 1e-7 \
  --lr-scheduler-type cosine \
  --warmup-ratio 0.03 \
  --optim adafactor \
  --eval-steps 500 \
  --save-steps 1000000 \
  --logging-steps 20 \
  --dataloader-num-workers 4 \
  --max-length 4096 \
  --report-to none
```

## Full Fine-Tuning 4B

Example for continuing from a local 4B DynEval checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 training/train_qwen3vl_dyneval.py \
  --train-only \
  --finetune-mode full \
  --model-path /path/to/qwen3vl-4b-checkpoint \
  --data-dir data/sft/my_dyneval_sft_data \
  --output-dir checkpoints/final/qwen3vl-4b-dyneval-new-v1 \
  --device-map none \
  --gradient-checkpointing \
  --ddp-find-unused-parameters \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --lr 1e-7 \
  --lr-scheduler-type cosine \
  --warmup-ratio 0.03 \
  --optim adafactor \
  --eval-steps 500 \
  --save-steps 1000000 \
  --logging-steps 20 \
  --dataloader-num-workers 4 \
  --max-length 4096 \
  --report-to none
```


## Important Notes

- Always pass `--finetune-mode full` for full fine-tuning.
- Use a new `--output-dir` for each experiment. Do not overwrite your best checkpoint.
- If torchrun says port `29500` is already in use, pass another port with `--master_port=29511`.
- `checkpoint-*/` folders are Trainer resume snapshots. They are not needed for inference or Hugging Face upload if the parent output folder already contains final model files.
- Use `--save-steps 1000000` if you only want the final saved model and do not want many intermediate checkpoint folders.
