# DynEval: Holistic Evaluation of Text-to-Image Generative Models in the Wild (ECCV-26)

Shyam Marjit, Dheeraj Baiju, Anuj Shikarkhane, Akhil Sakthieswaran, Sayak Paul, and Anirban Chakraborty

**Model Checkpoints:** [**DynEval-2B** & **DynEval-4B**](https://huggingface.co/vcl-iisc/DynEval-Evaluator) · **Dataset:** [**DynEval-1K**, **GenDB**, **DynEvalInstruct**](https://huggingface.co/datasets/vcl-iisc/DynEval-dataset) *(with teacher model responses)*

---

## Table of Contents

- [Main Contributions](#main-contributions)
- [Dataset Construction and Training](#dataset-construction-and-training)
  - [Step 1 — Filter Diverse Prompts](#step-1--filter-diverse-prompts)
  - [Step 2 — Generate Images](#step-2--generate-images)
  - [Step 3 — Distill Annotations with a Teacher VLM](#step-3--distill-annotations-with-a-teacher-vlm)
  - [Step 4 — Training](#step-4--training)
    - [4a — Task Tokens](#4a--task-tokens)
    - [4b — SFT Annotation Format](#4b--sft-annotation-format)
    - [4c — Build SFT Data](#4c--build-sft-data)
    - [4d — Run Training](#4d--run-training)
    - [4e — Logging and Checkpoints](#4e--logging-and-checkpoints)
- [Inference](#inference)
  - [Install Dependencies](#install-dependencies)
  - [Run DynEval-2B from Hugging Face](#run-dyneval-2b-from-hugging-face)
  - [Run DynEval-4B from Hugging Face](#run-dyneval-4b-from-hugging-face)
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

### Step 4 — Training

Train DynEval-2B or DynEval-4B with:

```text
training/train_qwen3vl_dyneval.py
```

The trainer expects a prepared SFT data directory:

```text
data/sft/<dataset_name>/train.jsonl
data/sft/<dataset_name>/val.jsonl
```

Pass this directory with:

```bash
--data-dir data/sft/<dataset_name>
```

#### 4a — Task Tokens

The tokenizer is initialized with three task tokens:

| Token | Used for |
|-------|----------|
| `<T2IA>` | Text-to-image alignment elements and questions |
| `<IQA>` | Image-quality assessment questions |
| `<EVALUATION>` | Image-based scoring from 1 to 5 |

#### 4b — SFT Annotation Format

Training data is stored as JSONL. Each line is one sample. The `messages` field is what the trainer reads. The task token appears in the user message, and the assistant message is the target output.

**`<T2IA>` element extraction**

This task is prompt-only. It trains the model to output the important image-generation elements as a JSON list.

```json
{
  "id": "sample_id_elements",
  "prompt_id": "sample_id",
  "prompt": "a photo of a carrot",
  "elements": ["carrot (food)"],
  "task": "T2IA_element_extraction",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "Given an aigc prompt, extract the elements that are important for generating images."}]},
    {"role": "user", "content": [{"type": "text", "text": "<T2IA>\nPrompt: a photo of a carrot\nElements:"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "[\"carrot (food)\"]"}]}
  ]
}
```

**`<T2IA>` single-question generation**

This task is also prompt-only. It trains the model to generate one yes/no verification question for one extracted element.

```json
{
  "id": "sample_id_single_question_000",
  "prompt_id": "sample_id",
  "prompt": "a photo of a carrot",
  "element": "carrot (food)",
  "question": "Is there a carrot in the photo?",
  "task": "T2IA_single_question_generation",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "Given a prompt for image generation and one of its related elements, generate one easy Yes/No question to verify whether the element is represented in the image generated by the prompt."}]},
    {"role": "user", "content": [{"type": "text", "text": "<T2IA>\nDescription: a photo of a carrot\nElement: carrot (food)\nReturn JSON:\n{\"question\": \"...\", \"answer\": \"yes/no\"}"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "{\"question\": \"Is there a carrot in the photo?\", \"answer\": \"yes\"}"}]}
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

**`<EVALUATION>` image scoring**

This task uses the image. It trains the model to score each generated question from 1 to 5.

```json
{
  "id": "sample_id_model_evaluation",
  "prompt_id": "sample_id",
  "prompt": "a photo of a carrot",
  "image_ref": "model_name/image.png",
  "image_path": "/path/to/image.png",
  "questions": ["Is there a carrot in the photo?"],
  "answers": [{"question": "Is there a carrot in the photo?", "score": 5}],
  "task": "EVALUATION_scoring",
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "You are a strict visual evidence evaluator."}]},
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "<EVALUATION>\nQuestions to score:\n1. Question: Is there a carrot in the photo?"}
      ]
    },
    {"role": "assistant", "content": [{"type": "text", "text": "[{\"question\": \"Is there a carrot in the photo?\", \"score\": 5}]"}]}
  ]
}
```

Do not put task tokens in assistant responses.

#### 4c — Build SFT Data

Use `data_building/build_dyneval_sft.py` to convert teacher-generated question/answer files into SFT JSONL.

Expected inputs:

- `--questions-dir`: JSON files containing prompt-level elements, T2IA questions, and/or IQA questions
- `--answers-dir`: JSON files containing image paths and question scores
- `--images-root`: root folder used to resolve image paths

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

The output directory will contain:

```text
data/sft/my_dyneval_sft_data/train.jsonl
data/sft/my_dyneval_sft_data/val.jsonl
data/sft/my_dyneval_sft_data/manifest.json
```

The generated SFT data can include rows for:

- `<T2IA>` element extraction
- `<T2IA>` single-question generation
- `<IQA>` image-quality question generation
- `<EVALUATION>` image scoring

To combine multiple prepared SFT folders:

```bash
python data_building/combine_sft_dirs.py \
  --input-dirs \
    data/sft/dataset_a \
    data/sft/dataset_b \
    data/sft/dataset_c \
  --output-dir data/sft/combined_dyneval_sft_data
```

#### 4d — Run Training

Use `--finetune-mode full` for full-parameter training.

**DynEval-2B:**

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

**DynEval-4B:**

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

---

## Inference

This repository includes `run_inference.py` for normal single-image inference with DynEval-2B or DynEval-4B from [`vcl-iisc/DynEval-Evaluator`](https://huggingface.co/vcl-iisc/DynEval-Evaluator).

The script runs the complete alignment-evaluation flow:

1. Extract text-to-image elements internally with `<|T2IA|>`.
2. Generate one yes/no visual question for each element with `<|T2IA|>`.
3. Score the image against the generated questions with `<|EVALUATION|>`.

By default, the terminal displays extracted elements, generated questions, and final scores. Raw model responses are hidden unless explicitly requested.
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
- `--repo-id REPO_ID`: Hugging Face repository; defaults to `vcl-iisc/DynEval-Evaluator`.
- `--prompt TEXT`: text-to-image prompt corresponding to the image (required).
- `--image PATH`: image to evaluate (required).
- `--output-file PATH`: optional path at which to save the JSON result.
- `--dtype {bfloat16,float16,float32,auto}`: model precision; defaults to `bfloat16`.
- `--device-map DEVICE_MAP`: model device placement; defaults to `auto`.
- `--max-new-tokens-elements N`: element-extraction generation limit; defaults to `256`.
- `--max-new-tokens-questions N`: per-question generation limit; defaults to `256`.
- `--max-new-tokens-answers N`: evaluation generation limit; defaults to `768`.
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
