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
- `<IQA>` for the three-step image-quality assessment pipeline (scene graph, node-grounded questions, per-question scoring).
- `<EVALUATION>` for image-based answering and 1--5 scoring of generated questions.

Two training entry points share the same trainer machinery and task tokens:

- `train_t2ia_qwen3vl.py` trains the `<T2IA>` and `<EVALUATION>` tasks.
- `train_iqa_qwen3vl.py` trains the `<IQA>` task. Run it on top of a `<T2IA>`/`<EVALUATION>` checkpoint so the final model supports all three tasks.

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

IQA rows are built by `train_iqa_qwen3vl.py` and come in three image-conditioned types, all using `<IQA>` in the user instruction. There is no system message (matching inference). The `task` field is one of `IQA_scene_graph`, `IQA_question_generation`, or `IQA_evaluation`:

```json
{
  "id": "sample_id_iqa_scene_graph",
  "task": "IQA_scene_graph",
  "image_path": "/path/to/image.png",
  "messages": [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "<IQA>\nHere is an image generated for this prompt \"...\". ..."}]},
    {"role": "assistant", "content": [{"type": "text", "text": "{\"nodes\": [...], \"edges\": [...]}"}]}
  ]
}
```

The other two IQA row types follow the same shape: the user message adds the scene-graph JSON (and, for scoring, the question JSON), and the assistant target is `{"questions": [...]}`.

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

### T2IA and EVALUATION

Use `data/build_t2ia_sft.py` for DynEval-style folders that contain:

- a question folder
- an answer folder
- an image root

Example:

```bash
python data/build_t2ia_sft.py \
  --questions-dir /path/to/questions \
  --answers-dir /path/to/answers \
  --images-root /path/to/images \
  --output-dir data/sft/my_dyneval_sft_data \
  --val-ratio 0.05 \
  --seed 42
```

This creates SFT data for:

- `<T2IA>` element extraction
- `<T2IA>` single-question generation
- `<EVALUATION>` image scoring

### IQA

Build `<IQA>` SFT data with `data/build_iqa_sft.py`, pointing at the teacher IQA outputs from `Distill Annotations/IQA/`:

```bash
python data/build_iqa_sft.py \
  --scene-graph-dir /path/to/iqa_outputs \
  --answers-dir /path/to/iqa_answers \
  --images-root /path/to/images \
  --output-dir data/sft/my_iqa_sft_data \
  --val-ratio 0.05 \
  --seed 42
```

`--scene-graph-dir` is required (scene graph + question rows); `--answers-dir` is optional and adds the per-question scoring rows. Use `--score-mode {correct,overall}` to control how per-question 1--5 scores are derived. This creates SFT data for:

- `<IQA>` scene graph generation
- `<IQA>` node-grounded question generation
- `<IQA>` per-question scoring

The same data-building logic is also available inline via `training/train_iqa_qwen3vl.py --prepare-only` if you prefer a single command for prep + training.


## Full Fine-Tuning 2B

Example for continuing from a local 2B DynEval checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29511 training/train_t2ia_qwen3vl.py \
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
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29512 training/train_t2ia_qwen3vl.py \
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

## Full Fine-Tuning IQA

Train the `<IQA>` task with `train_iqa_qwen3vl.py`. Start from a checkpoint that already has `<T2IA>` and `<EVALUATION>` trained so the resulting model supports all three tasks:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29513 training/train_iqa_qwen3vl.py \
  --train-only \
  --finetune-mode full \
  --model-path checkpoints/final/qwen3vl-4b-dyneval-new-v1 \
  --data-dir data/sft/my_iqa_sft_data \
  --output-dir checkpoints/final/qwen3vl-4b-dyneval-iqa-v1 \
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

Use `--score-mode` to control how per-question 1--5 scores are derived from the teacher: `correct` (default) maps correct/incorrect to 5/1, and `overall` reuses the teacher's overall score for each question.

## W&B Logging

To log offline and sync later:

```bash
WANDB_MODE=offline \
WANDB_PROJECT=dyneval-qwen3vl \
WANDB_RUN_NAME=my-run-name \
WANDB_LOG_MODEL=false \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29511 training/train_t2ia_qwen3vl.py \
  ... \
  --report-to wandb
```

After training:

```bash
wandb sync wandb/offline-run-...
```

## Important Notes

- Always pass `--finetune-mode full` for full fine-tuning.
- Use a new `--output-dir` for each experiment. Do not overwrite your best checkpoint.
- If torchrun says port `29500` is already in use, pass another port with `--master_port=29511`.
- `checkpoint-*/` folders are Trainer resume snapshots. They are not needed for inference or Hugging Face upload if the parent output folder already contains final model files.
- Use `--save-steps 1000000` if you only want the final saved model and do not want many intermediate checkpoint folders.
