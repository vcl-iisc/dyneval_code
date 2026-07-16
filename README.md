# DynEval: Holistic Evaluation of Text-to-Image Generative Models in the Wild (ECCV-26)

Shyam Marjit*, Dheeraj Baiju*, Anuj Shikarkhane*, Akhil Sakthieswaran, Sayak Paul, and Anirban Chakraborty

**Model Checkpoints:** [**DynEval-2B** & **DynEval-4B**](https://huggingface.co/vcl-iisc/DynEval-Evaluator) · **Dataset:** [**DynEval-1K**, **GenDB**, **DynEvalInstruct**](https://huggingface.co/datasets/vcl-iisc/DynEval-dataset) *(with teacher model responses)*

---

## Table of Contents

- [Main Contributions](#main-contributions)
- [Dataset Construction and Training](#dataset-construction-and-training)
  - [Step 1 — Filter Diverse Prompts](#step-1--filter-diverse-prompts)
  - [Step 2 — Generate Images](#step-2--generate-images)
  - [Step 3 — Distill Annotations with a Teacher VLM](#step-3--distill-annotations-with-a-teacher-vlm)
  - [Step 4 — Fine-tune DynEval](#step-4--fine-tune-dyneval)
    - [4a — Task Tokens](#4a--task-tokens)
    - [4b — SFT Annotation Format](#4b--sft-annotation-format)
    - [4c — Build SFT Data](#4c--build-sft-data)
    - [4d — Run Full Fine-tuning](#4d--run-full-fine-tuning)
    - [4e — Logging and Checkpoints](#4e--logging-and-checkpoints)
- [Inference](#inference)
  - [Install Dependencies](#install-dependencies)
  - [Run DynEval-2B from Hugging Face](#run-dyneval-2b-from-hugging-face)
  - [Run DynEval-4B from Hugging Face](#run-dyneval-4b-from-hugging-face)
  - [Run a Local Checkpoint](#run-a-local-checkpoint)
  - [Example Terminal Output](#example-terminal-output)
  - [Optional Debug Fields](#optional-debug-fields)
  - [Command-Line Arguments](#command-line-arguments)
  - [Notes](#notes)
- [Quantitative Results](#quantitative-results)
- [Qualitative Results](#qualitative-results)

---

## Main Contributions

**(i)** We construct two large-scale datasets, GenDB and DynEvalInstruct with well-balanced prompt coverage and image generations from 36 diverse T2I models for evaluator training. We construct GenDB, a large-scale prompt–image dataset with well-balanced prompt coverage and generations from 36 diverse T2I models, and derive DynEvalInstruct from GenDB for evaluator training.

**(ii)** Unlike static QA methods, we propose DynEval, a dynamic evaluator that jointly evaluates prompt-generated image alignment as well as builds scene graphs from generated images to compose structured, image-specific questions for fine-grained image quality assessment.

**(iii)** To obtain a robust evaluator, we introduce tier-based prompt categorization with tier-specific T2I model generation to cover T2I models failure modes across varying prompt complexities and model capabilities To train a robust evaluator, we introduce tier-matched prompt–model generation, pairing prompts of varying complexity with T2I models of corresponding capability to capture informative failure modes across the model spectrum.

**(iv)** Across multiple established benchmarks, DynEval achieves superior correlation with human judgments than prior works. Our extensive analysis over 36 T2I models reveals multiple sub-categories to identify attributes that remain challenging for current SOTA T2I models, offering insights for improving next-generation models.

<p align="center">
  <img src="assets/method.png" width="800"/>
  <br>
  <em>Overview of GenDB and DynEvalInstruct construction</em>
</p>

<p align="center">
  <img src="assets/method2.png" width="800"/>
  <br>
  <em>Overview of DynEvalInstruct construction and the DynEval evaluation framework.</em>
</p>

---

## Dataset Construction and Training

### Step 1 — Filter Diverse Prompts

Run `extract_diverse_prompts.py` to filter the required prompts from DiffusionDB (a database of 1.8 billion human-written prompts).

> **Note:** The publicly available version of `extract_diverse_prompts.py` performs basic filtering. The higher-performance version is kept private.

Once prompts are filtered, each prompt must be assigned to a category — **Tier 1**, **Tier 2**, or **Tier 3** — based on complexity.


All such prompt info is present in the [Hugging Face dataset](https://huggingface.co/datasets/vcl-iisc/DynEval-dataset).
---

### Step 2 — Generate Images

Using the prompt-to-tier assignments from Step 1, generate images by matching prompt tiers to model tiers (i.e., Tier 1 prompts are assigned to Tier 1 models only). Generation scripts for each model are available in the `image_gen_scripts` folder.

The output of this step is the **GenDB** dataset, containing ⟨image, text prompt⟩ pairs.

All used (prompt, image) pairs are present in the [Hugging Face dataset](https://huggingface.co/datasets/vcl-iisc/DynEval-dataset).
---

### Step 3 — Distill Annotations with a Teacher VLM

Pass each ⟨image, text prompt⟩ pair to the teacher VLM ([`Qwen/Qwen3-VL-235B-A22B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct)) to generate T2IA (Text-to-Image Alignment) and IQA (Image Quality Assessment) annotations.

Use the scripts in [`Distill Annotations/`](Distill%20Annotations/README.md):

1. Run the **T2IA** and **IQA** teacher workflows to generate questions and answers.
2. Run `build_dynevalinstruct.py` to convert those outputs into **DynEvalInstruct** JSON for fine-tuning.

The teacher model uses natural-language prompts only. The student DynEval model is trained with task tokens `<\|T2IA\|>`, `<\|IQA\|>`, and `<\|EVALUATION\|>` (IDs 151669–151671), which are inserted during the build step—not sent to the teacher VLM.

---

### Step 4 — Fine-tune DynEval

Fine-tune `Qwen/Qwen3-VL-4B-Instruct` (DynEval-4B) or `Qwen/Qwen3-VL-2B-Instruct` (DynEval-2B) using the training script in this repository:

```text
training/train_qwen3vl_dyneval.py
```

This code does **not** use the older `Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py` dataset-registration flow. You do not need to set `DYNEVALINSTRUCT_T2IA_ANNOTATION`, `DYNEVALINSTRUCT_T2IA_DATA`, `DYNEVALINSTRUCT_IQA_ANNOTATION`, or `DYNEVALINSTRUCT_IQA_DATA`.

Instead, training uses prebuilt SFT JSONL files:

```text
data/sft/<dataset_name>/train.jsonl
data/sft/<dataset_name>/val.jsonl
```

Each JSONL row already contains the full chat messages for one task. The trainer only needs:

```bash
--data-dir data/sft/<dataset_name>
```

#### 4a — Task Tokens

The training script registers the task tokens at startup:

| Token | Role |
|-------|------|
| `<T2IA>` | Prompt-only text-to-image alignment element extraction and question generation |
| `<IQA>` | Image-quality assessment question generation, including scene/quality checks |
| `<EVALUATION>` | Image-based question answering and 1--5 scoring |

Training uses:

- `<T2IA>` for text-to-image alignment element extraction and visual question generation.
- `<IQA>` for image-quality assessment question generation, including scene/quality checks.
- `<EVALUATION>` for image-based answering and 1--5 scoring of generated questions.

#### 4b — SFT Annotation Format

Each training row is a JSON object with a `messages` field in Qwen chat format.

**`<T2IA>` element extraction or question generation:**

```json
{
  "id": "sample_id",
  "task": "t2ia_element_extraction",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [{"type": "text", "text": "<T2IA>\nPrompt: a photo of a carrot\nElements:"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "[\"carrot (food)\"]"}]}
  ]
}
```

**`<IQA>` image-quality question generation:**

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
        {"type": "text", "text": "<IQA>\nGenerate image-quality assessment questions for visible artifacts, distortions, texture, shape consistency, and spatial quality."}
      ]
    },
    {"role": "assistant", "content": [{"type": "text", "text": "[{\"question\": \"Are object shapes visually coherent?\", \"answer\": \"yes\"}]"}]}
  ]
}
```

**`<EVALUATION>` image scoring:**

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
        {"type": "text", "text": "<EVALUATION>\nQuestions to score:\n1. Is there a carrot in the photo?"}
      ]
    },
    {"role": "assistant", "content": [{"type": "text", "text": "[{\"question\": \"Is there a carrot in the photo?\", \"score\": 5}]"}]}
  ]
}
```

Do not put the task token in the assistant response. The task token should appear in the user instruction.

#### 4c — Build SFT Data

Use `data_building/build_dyneval_sft.py` for DynEval-style folders containing:

- a question folder
- an answer folder
- an image root

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

This creates mixed SFT data for:

- `<T2IA>` element extraction
- `<T2IA>` single-question generation
- `<EVALUATION>` image scoring

To combine multiple already-built SFT folders:

```bash
python data_building/combine_sft_dirs.py \
  --input-dirs \
    data/sft/dataset_a \
    data/sft/dataset_b \
    data/sft/dataset_c \
  --output-dir data/sft/combined_dyneval_sft_data
```

#### 4d — Run Full Fine-tuning

Always pass `--finetune-mode full` for the full fine-tuning runs reported for DynEval.

**DynEval-2B:**

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29511 training/train_qwen3vl_dyneval.py \
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

**DynEval-4B:**

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29512 training/train_qwen3vl_dyneval.py \
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

---

## Inference

This repository includes `run_inference.py` for normal single-image inference. It can load either:

- a DynEval-2B or DynEval-4B checkpoint from [`vcl-iisc/DynEval-Evaluator`](https://huggingface.co/vcl-iisc/DynEval-Evaluator), or
- local checkpoint weights from a path on disk.

The script runs the complete alignment-evaluation flow:

1. Extract text-to-image elements internally with `<|T2IA|>`.
2. Generate one yes/no visual question for each element with `<|T2IA|>`.
3. Score the image against the generated questions with `<|EVALUATION|>`.

By default, the terminal displays only the generated questions and final scores. Intermediate elements and raw model responses are hidden unless explicitly requested.
When `--variant` is omitted, the script loads DynEval-4B by default.

### Install Dependencies

Use an environment with a recent Qwen3-VL-compatible version of `transformers`:

```bash
pip install torch transformers accelerate pillow
```

If the Hugging Face model requires access approval, accept it on the [model page](https://huggingface.co/vcl-iisc/DynEval-Evaluator) and then log in:

```bash
huggingface-cli login
```

Run the following commands from the repository root (the directory containing `run_inference.py`).

### Run DynEval-2B from Hugging Face

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  --variant 2b \
  --prompt "a photo of a carrot" \
  --image example.jpg \
  --output-file output.json
```

### Run DynEval-4B from Hugging Face

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  --prompt "a photo of a carrot" \
  --image example.jpg \
  --output-file output.json
```

### Run a Local Checkpoint

Use `--checkpoint` when the weights are already downloaded locally:

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  --checkpoint /path/to/local/checkpoint \
  --prompt "a photo of a carrot" \
  --image /path/to/image.jpg \
  --output-file output.json
```

`--checkpoint` overrides both `--variant` and `--repo-id`.

### Example Terminal Output

```text
========================================================================================
DynEval Evaluator Result
========================================================================================
Prompt: a photo of a carrot
Image: example.jpg

Questions (1)
  1. Is there a carrot in the photo?
     ground-truth answer: yes

Evaluation Scores (1)
  1. score=5 | Is there a carrot in the photo?
  Mean score: 5.000
========================================================================================
```

When `--output-file` is provided, the script also saves JSON with the following structure:

```json
{
  "prompt": "...",
  "image_path": "...",
  "questions": [],
  "answers": []
}
```

### Optional Debug Fields

Use `--include-elements` to add the intermediate extracted elements to the saved JSON, and `--include-raw` to add the raw model responses:

```bash
CUDA_VISIBLE_DEVICES=0 python run_inference.py \
  --variant 2b \
  --prompt "a photo of a carrot" \
  --image example.jpg \
  --output-file output_debug.json \
  --include-elements \
  --include-raw
```

These flags affect the saved JSON only; the terminal output remains concise.

### Command-Line Arguments

- `--variant {2b,4b}`: Hugging Face model variant; defaults to `4b`.
- `--checkpoint PATH`: local checkpoint to load instead of Hugging Face weights.
- `--repo-id REPO_ID`: Hugging Face repository; defaults to `vcl-iisc/DynEval-Evaluator`.
- `--prompt TEXT`: text-to-image prompt corresponding to the image (required).
- `--image PATH`: image to evaluate (required).
- `--output-file PATH`: optional path at which to save the JSON result.
- `--dtype {bfloat16,float16,float32,auto}`: model precision; defaults to `bfloat16`.
- `--device-map DEVICE_MAP`: model device placement; defaults to `auto`.
- `--max-new-tokens-elements N`: element-extraction generation limit; defaults to `256`.
- `--max-new-tokens-questions N`: per-question generation limit; defaults to `256`.
- `--max-new-tokens-answers N`: evaluation generation limit; defaults to `768`.
- `--include-elements`: include extracted elements in the saved JSON.
- `--include-raw`: include raw model responses in the saved JSON.

### Notes

- For the best reproducibility, use the exact text prompt associated with the evaluated image.
- The script adds the DynEval task tokens internally; do not add `<|T2IA|>` or `<|EVALUATION|>` to `--prompt`.
- The Hugging Face repository stores the variants in the `DynEval-2B` and `DynEval-4B` subfolders.
- `--output-file` is optional. The formatted result is always printed to the terminal.

## Quantitative Results

<p align="center">
  <img src="assets/zero_shot.png" width="800"/>
  <br>
  <em>Zero-shot cross-dataset evaluation across diverse benchmarks, comparing existing scoring methods with EvalMuse and DynEval variants.</em>
</p>

<p align="center">
  <img src="assets/zero_shot2.png" width="800"/>
  <br>
  <em>More recent Zero-shot cross-dataset evaluation across diverse benchmarks with newer T2I evaluators.</em>
</p>

## Qualitative Results

<p align="center">
  <img src="assets/geneval_dyneval.png" width="800"/>
  <br>
  <em>Evaluation on the GenEval dataset. Inputs consist of image–text prompt pairs from a mix of real and generated images, shown alongside human ratings, the mean human rating, and the DynEval score (scale: 1–5). Although DynEval is trained on synthetic images, the fine-tuned model demonstrates the ability to generalize to real images.</em>
</p>

<p align="center">
  <img src="assets/AGIKA-3K_dyneval.png" width="800"/>
  <br>
  <em>Evaluation on the AGIKA-3K dataset. Inputs consist of image–text prompt pairs shown alongside human ratings, the mean human rating, and the DynEval score (scale: 1–5).</em>
</p>

<p align="center">
  <img src="assets/genai_bench_dyneval.png" width="800"/>
  <br>
  <em>Evaluation on the GenAI-Bench dataset. Inputs consist of image–text prompt pairs shown alongside human ratings, the mean human rating, and the DynEval score (scale: 1–5).</em>
</p>



<p align="center">
  <img src="assets/fail.png" width="800"/>
  <br>
  <em>Alignment scores across prompt sub-categories in DynEval-1K evaluation dataset, grouped by model tier. The 42 sub-categories span nine semantic dimensions, and scores represent the average DynEval alignment score. Models are grouped into three tiers based on overall alignment performance, with bars showing the tier-averaged score for each sub-category. Tier-1 models consistently achieve stronger alignment across most sub-categories, with the largest performance gaps appearing in challenging categories such as counting, text rendering, and high-complexity prompts.
  </em>
</p>

---
