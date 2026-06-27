# Distill Annotations

This folder contains teacher-VLM scripts for generating T2IA and IQA annotations, plus a builder that converts those outputs into **DynEvalInstruct** JSON for Step 4 fine-tuning.

The teacher model is [`Qwen/Qwen3-VL-235B-A22B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct). The student DynEval model is trained with task tokens:

| Token | Token ID | Used in |
|-------|----------|---------|
| `<\|T2IA\|>` | 151669 | Stage 1 question generation (prompt only) |
| `<\|IQA\|>` | 151670 | Stage 2 scene-graph / quality-question generation |
| `<\|EVALUATION\|>` | 151671 | Stage 1 and Stage 2 answering / scoring |

These tokens are **not** sent to the teacher VLM. They are added by `build_dynevalinstruct.py` when creating the fine-tuning JSON consumed in [Step 4](../README.md#step-4--fine-tune-dyneval).

## End-to-End Workflow

```text
DYNEVAL-250K-PROMPTS.json
  -> T2IA/question.py            -> T2IA questions
  -> T2IA/answering.py           -> T2IA answers (1-5 scoring)
  -> IQA/question.py             -> IQA scene graph + questions
  -> IQA/answer.py               -> IQA answers (1-5 scoring)
  -> build_dynevalinstruct.py    -> dynevalinstruct_t2ia.json + dynevalinstruct_iqa.json
  -> Qwen3-VL fine-tuning        -> DynEval-2B / DynEval-4B
```

## Expected Input Format

The prompt mapping file must be a JSON object with a `prompts` list:

```json
{
  "prompts": [
    {
      "pair_id": "D250K-000001",
      "text_id": "000",
      "prompt": "statue of a man",
      "image_path": "DYNEVAL-250K-IMAGES/D250K-000001.png",
      "questions_file": "model-responses/questions/D250K-000001.json",
      "response_file": "model-responses/answers/D250K-000001.json",
      "generation_model": "unknown"
    }
  ]
}
```

Relative paths are resolved from:

1. The current working directory
2. The script folder
3. The repository root

## Dependencies

```bash
pip install transformers accelerate qwen-vl-utils
```

For FP8 / quantized Qwen inference through vLLM:

```bash
pip install vllm qwen-vl-utils
```

Default models:

```text
Qwen/Qwen3-VL-235B-A22B-Instruct
Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
```

With `--backend auto`, FP8 models use vLLM automatically. Non-FP8 models use Transformers.

## Common Arguments

Most scripts support:

```bash
--annotations-file DYNEVAL-250K-PROMPTS.json
--questions-dir path/to/questions
--images-root path/to/images
--output-dir path/to/outputs
--model Qwen/Qwen3-VL-235B-A22B-Instruct
--backend auto
--start-idx 0
--end-idx 10
--force
```

vLLM-related arguments:

```bash
--gpu-memory-utilization 0.70
--tensor-parallel-size 8
--temperature 0.0
```

## T2IA Workflow

T2IA generates prompt-alignment yes/no questions and then answers those questions for each image.

### 1. Generate T2IA Questions

```bash
python T2IA/question.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --questions-dir model-responses/questions \
  --start-idx 0 \
  --end-idx 10
```

This script makes two teacher calls per prompt:

1. Standard alignment yes/no questions with target answers
2. Distortion-based yes/no questions with target answers

If `--questions-dir` is not provided, it writes to each record's `questions_file` path from the prompt mapping JSON.

### 2. Answer T2IA Questions

```bash
python T2IA/answering.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --questions-dir model-responses/questions \
  --images-root DYNEVAL-250K-IMAGES \
  --output-dir model-responses/answers \
  --start-idx 0 \
  --end-idx 10
```

This script loads each image and its generated questions, answers yes/no based on the image, compares against target answers, and saves per-question scores on a **1–5** scale.

## IQA Workflow

IQA generates image-quality questions using a scene graph first, then answers those questions with a 1–5 quality score.

### 1. Generate IQA Questions

```bash
python IQA/question.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --images-root DYNEVAL-250K-IMAGES \
  --output-dir iqa_outputs \
  --questions-dir iqa_questions \
  --start-idx 0 \
  --end-idx 10
```

This script makes two teacher calls per image:

1. Generate a scene graph from the image and prompt
2. Generate yes/no quality questions with target answers from the scene graph

The combined output is saved to `--output-dir`. If `--questions-dir` is provided, the generated yes/no questions are also saved separately there.

### 2. Answer IQA Questions

```bash
python IQA/answer.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --questions-dir iqa_questions \
  --images-root DYNEVAL-250K-IMAGES \
  --output-dir iqa_answers \
  --start-idx 0 \
  --end-idx 10
```

This script sends the prompt, image, and generated IQA yes/no questions to the teacher VLM. It answers each question and produces an overall quality score from 1 to 5.

## Build DynEvalInstruct for Fine-tuning

After the teacher workflows finish, convert the outputs into curriculum JSON files for Step 4:

```bash
python build_dynevalinstruct.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --t2ia-questions-dir model-responses/questions \
  --t2ia-answers-dir model-responses/answers \
  --iqa-outputs-dir iqa_outputs \
  --iqa-answers-dir iqa_answers \
  --output-t2ia dynevalinstruct_t2ia.json \
  --output-iqa dynevalinstruct_iqa.json
```

This script wraps teacher outputs into Qwen-VL conversation samples:

| Output file | Human turn starts with | Notes |
|-------------|------------------------|-------|
| `dynevalinstruct_t2ia.json` | `<\|T2IA\|>` or `<\|EVALUATION\|>` | Stage 1 curriculum |
| `dynevalinstruct_iqa.json` | `<\|IQA\|>` or `<\|EVALUATION\|>` | Stage 2 curriculum |

Human-turn templates are defined in `task_tokens.py` and match the root [README Step 4](../README.md#4b--annotation-format):

- `<\|T2IA\|>` samples are **prompt-only** and do not include an `image` field
- `<\|IQA\|>` and `<\|EVALUATION\|>` samples include `"image": "..."` and start with `<image>` in the human turn

Then point fine-tuning to the generated files:

```bash
export DYNEVALINSTRUCT_T2IA_ANNOTATION=/path/to/dynevalinstruct_t2ia.json
export DYNEVALINSTRUCT_T2IA_DATA=/path/to/DYNEVAL-250K-IMAGES
export DYNEVALINSTRUCT_IQA_ANNOTATION=/path/to/dynevalinstruct_iqa.json
export DYNEVALINSTRUCT_IQA_DATA=/path/to/DYNEVAL-250K-IMAGES
```

## Running FP8 with vLLM

```bash
python T2IA/answering.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --questions-dir model-responses/questions \
  --images-root DYNEVAL-250K-IMAGES \
  --output-dir model-responses/answers \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --backend vllm \
  --gpu-memory-utilization 0.70
```

Set `--tensor-parallel-size` when you want to control the number of GPUs. If omitted, the scripts use the detected CUDA GPU count.

## Resume Behavior

All scripts skip existing valid outputs by default. Use `--force` to regenerate outputs.

Use `--start-idx` and `--end-idx` for chunked or single-image runs:

```bash
python IQA/answer.py --start-idx 0 --end-idx 1
```

## Teacher Outputs

T2IA question files contain:

```json
[
  {"question": "...", "answer": "yes"}
]
```

T2IA answer files contain:

```json
{
  "pair_id": "...",
  "prompt": "...",
  "answers": [
    {
      "question": "...",
      "target_answer": "yes",
      "reasoning": "...",
      "score": 5
    }
  ],
  "raw_response": "..."
}
```

IQA question outputs contain:

```json
{
  "pair_id": "...",
  "prompt": "...",
  "scene_graph": {},
  "questions": [
    {"question": "...", "answer": "no"}
  ]
}
```

IQA answer outputs contain:

```json
{
  "pair_id": "...",
  "prompt": "...",
  "answers": [
    {
      "question": "...",
      "answer": "no",
      "target_answer": "no",
      "correct": true
    }
  ],
  "score": 5
}
```

## Combining IQA and T2IA Scores

Use `compute_overall_scores.py` to combine the IQA and T2IA scores:

```bash
python compute_overall_scores.py \
  --iqa-dir iqa_answers \
  --t2ia-dir model-responses/answers \
  --output-file overall_scores.json \
  --alpha 0.5 \
  --beta 0.5
```

The formula is:

```text
overall_score = alpha * iqa_score + beta * t2ia_score
```

`--alpha` and `--beta` default to `0.5`.

The script joins files by `pair_id`, then `item_key`, then `image_id`, and finally the filename stem. It reads a top-level `score` when present. If a file has no top-level score, it averages per-answer `score` values. For boolean correctness fields, `true` is treated as `5` and `false` as `1`.

To write CSV instead of JSON:

```bash
python compute_overall_scores.py \
  --iqa-dir iqa_answers \
  --t2ia-dir model-responses/answers \
  --output-file overall_scores.csv \
  --format csv
```
