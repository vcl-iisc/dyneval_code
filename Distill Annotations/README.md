# Distill Annotations

This folder contains the teacher-VLM scripts that generate T2IA and IQA annotations, plus builders that convert those outputs into fine-tuning data for Step 4.

Annotation generation is done by **two self-contained scripts**:

| Script | What it does |
|--------|--------------|
| `t2ia.py` | Extracts prompt elements, generates one yes/no question per element, then scores each question 1–5 against the image. |
| `iqa.py`  | Builds a scene graph, generates rendering-quality yes/no questions per node, then answers/scores them 1–5 against the image. |

Both scripts run the full generate-then-score pipeline in a single pass, mirroring the prompts in [`run_inference.py`](../run_inference.py) so the teacher produces annotations in the exact style the student learns.

The teacher model is [`Qwen/Qwen3-VL-235B-A22B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct). The student DynEval model is trained with task tokens:

| Token | Token ID | Used in |
|-------|----------|---------|
| `<\|T2IA\|>` | 151669 | Element extraction / question generation (prompt only) |
| `<\|IQA\|>` | 151670 | Scene-graph / quality-question generation |
| `<\|EVALUATION\|>` | 151671 | Answering / 1–5 scoring |

These tokens are **not** sent to the teacher VLM. They are added later by the SFT builders when creating the fine-tuning data consumed in [Step 4](../README.md#step-4--training).

## End-to-End Workflow

```text
DYNEVAL-250K-PROMPTS.json
  -> t2ia.py   -> t2ia_questions/  (elements + per-element questions)
                  t2ia_answers/    (1-5 scores)
  -> iqa.py    -> iqa_outputs/     (scene graph + quality questions)
                  iqa_answers/     (1-5 scores)
  -> build SFT data (see "Build Fine-tuning Data")
  -> Qwen3-VL fine-tuning -> DynEval-2B / DynEval-4B
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
      "generation_model": "unknown"
    }
  ]
}
```

Relative paths (including `image_path`) are resolved from:

1. The current working directory
2. The script folder
3. The repository root

Use `--images-root` to point at the image directory when `image_path` is relative.

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

With `--backend auto`, FP8 models use vLLM automatically; non-FP8 models use Transformers.

## Common Arguments

Both scripts support:

```bash
--annotations-file DYNEVAL-250K-PROMPTS.json
--images-root path/to/images
--model Qwen/Qwen3-VL-235B-A22B-Instruct
--backend auto            # auto | transformers | vllm
--start-idx 0
--end-idx 10
--force                   # regenerate even if valid outputs already exist
```

vLLM-related arguments:

```bash
--gpu-memory-utilization 0.70
--tensor-parallel-size 8
--temperature 0.0
```

## T2IA: `t2ia.py`

For each prompt the script:

1. Extracts the important visual elements from the prompt (each classified by type, e.g. `bench (object)`).
2. Generates one yes/no verification question per element, with a target answer.
3. Scores each question 1–5 by visual evidence (`5 = yes, definitely` … `1 = no, definitely`), looking only at the image.

```bash
python t2ia.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --images-root DYNEVAL-250K-IMAGES \
  --questions-dir t2ia_questions \
  --answers-dir t2ia_answers \
  --start-idx 0 \
  --end-idx 10
```

Outputs:

- `--questions-dir/<pair_id>.json` — one entry per element:

```json
[
  {"element": "bench (object)", "question": "Is there a bench in the photo?", "answer": "yes"}
]
```

- `--answers-dir/<pair_id>.json` — per-question 1–5 scores:

```json
{
  "pair_id": "...",
  "prompt": "...",
  "image_path": "...",
  "answers": [
    {"question": "...", "element": "bench (object)", "target_answer": "yes", "reasoning": "...", "score": 5}
  ]
}
```

## IQA: `iqa.py`

For each image the script:

1. Generates a scene graph (`nodes` + `edges` only), using the prompt as reference.
2. Passes the node ids and labels back and generates rendering-quality yes/no questions (presence/attribute/color/count/alignment questions are forbidden), each with a `node_id` and a `target_answer`.
3. Answers each question from the image, marks correctness against the target, and produces an overall 1–5 quality score.

```bash
python iqa.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --images-root DYNEVAL-250K-IMAGES \
  --output-dir iqa_outputs \
  --answers-dir iqa_answers \
  --start-idx 0 \
  --end-idx 10
```

Outputs:

- `--output-dir/<pair_id>.json` — scene graph + quality questions:

```json
{
  "pair_id": "...",
  "prompt": "...",
  "scene_graph": {
    "nodes": [{"id": "object_1", "label": "bench", "attributes": ["wooden"]}],
    "edges": [{"source": "object_1", "relation": "on", "target": "object_2"}]
  },
  "questions": [
    {"node_id": "object_1", "question": "Are the bench slats straight and free of distortion?", "target_answer": "yes"}
  ]
}
```

- `--answers-dir/<pair_id>.json` — answers + overall quality score:

```json
{
  "pair_id": "...",
  "prompt": "...",
  "answers": [
    {"question": "...", "answer": "no", "target_answer": "yes", "correct": false, "reasoning": "..."}
  ],
  "score": 3
}
```

`iqa.py` also exposes per-stage generation limits: `--scene-graph-max-new-tokens`, `--questions-max-new-tokens`, and `--answer-max-new-tokens`.

## Running FP8 with vLLM

```bash
python t2ia.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --images-root DYNEVAL-250K-IMAGES \
  --questions-dir t2ia_questions \
  --answers-dir t2ia_answers \
  --model Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --backend vllm \
  --gpu-memory-utilization 0.70
```

Set `--tensor-parallel-size` to control the number of GPUs. If omitted, the scripts use the detected CUDA GPU count.

## Resume Behavior

Both scripts skip records whose question **and** answer outputs already exist and are valid. Use `--force` to regenerate. Use `--start-idx` / `--end-idx` for chunked or single-record runs:

```bash
python iqa.py --start-idx 0 --end-idx 1
```

Failed records write a `<pair_id>.error.txt` next to the answer output instead of stopping the run.

## Build Fine-tuning Data

After both scripts finish, build the SFT JSONL with the builders in [`../data`](../data):

```bash
python ../data/build_t2ia_sft.py \
  --questions-dir t2ia_questions \
  --answers-dir t2ia_answers \
  --images-root DYNEVAL-250K-IMAGES \
  --output-dir ../data/sft/my_dyneval_sft_data

python ../data/build_iqa_sft.py \
  --scene-graph-dir iqa_outputs \
  --answers-dir iqa_answers \
  --images-root DYNEVAL-250K-IMAGES \
  --output-dir ../data/sft/my_iqa_sft_data
```

Alternatively, `build_dynevalinstruct.py` wraps the same teacher outputs into DynEvalInstruct conversation JSON:

```bash
python build_dynevalinstruct.py \
  --annotations-file DYNEVAL-250K-PROMPTS.json \
  --t2ia-questions-dir t2ia_questions \
  --t2ia-answers-dir t2ia_answers \
  --iqa-outputs-dir iqa_outputs \
  --iqa-answers-dir iqa_answers \
  --output-t2ia dynevalinstruct_t2ia.json \
  --output-iqa dynevalinstruct_iqa.json
```

Human-turn templates are defined in `task_tokens.py`:

- `<\|T2IA\|>` samples are **prompt-only** (no `image` field).
- `<\|IQA\|>` and `<\|EVALUATION\|>` samples include an `image` and start from the generated image.

## Combining IQA and T2IA Scores

Use `compute_overall_scores.py` to combine the IQA and T2IA scores:

```bash
python compute_overall_scores.py \
  --iqa-dir iqa_answers \
  --t2ia-dir t2ia_answers \
  --output-file overall_scores.json \
  --alpha 0.5 \
  --beta 0.5
```

The formula is:

```text
overall_score = alpha * iqa_score + beta * t2ia_score
```

`--alpha` and `--beta` default to `0.5`. The script joins files by `pair_id`, then `item_key`, then `image_id`, and finally the filename stem. It reads a top-level `score` when present; otherwise it averages per-answer `score` values (treating boolean `correct` as `5`/`1`). Use `--format csv` to write CSV instead of JSON.
