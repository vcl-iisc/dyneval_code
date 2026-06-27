# DynEval: Holistic Evaluation of Text-to-Image Generative Models in the Wild (ECCV-26)

Shyam Marjit, Dheeraj Baiju, Anuj Shikarkhane, Akhil Sakthieswaran, Sayak Paul, and Anirban Chakraborty

**Model Checkpoints:** [**DynEval-2B** & **DynEval-4B**](https://huggingface.co/vcl-iisc/DynEval-Evaluator) · **Dataset:** [**DynEval-1K**, **GenDB**, **DynEvalInstruct**](https://huggingface.co/datasets/vcl-iisc/DynEval-dataset) *(with teacher model responses)*

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

## How to Run

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

Fine-tune `Qwen/Qwen3-VL-4B-Instruct` (DynEval-4B) or `Qwen/Qwen3-VL-2B-Instruct` (DynEval-2B) on the DynEvalInstruct dataset produced in Step 3. DynEval is **not** distilled as a black-box regressor. Instead, training uses three task-specific tokens that trigger distinct evaluation procedures:

| Token | Token ID | Role |
|-------|----------|------|
| `<\|T2IA\|>` | 151669 | Prompt-only alignment question generation, including distortion checks |
| `<\|IQA\|>` | 151670 | Scene-graph construction and image-quality question generation |
| `<\|EVALUATION\|>` | 151671 | VQA-style answering and scoring in the range [1--5] |

These tokens are registered in the tokenizer before fine-tuning and must appear in the **human** turns of DynEvalInstruct. The same token strings and IDs apply to both DynEval-2B and DynEval-4B. Training follows a **curriculum learning** strategy:
- **Stage 1:** `<\|T2IA\|>` question generation, then `<\|EVALUATION\|>` scoring on the generated questions
- **Stage 2:** `<\|IQA\|>` scene-graph / quality-question generation, then `<\|EVALUATION\|>` scoring

After fine-tuning, the checkpoint should contain an `added_tokens.json` with:

```json
{
  "<|T2IA|>": 151669,
  "<|IQA|>": 151670,
  "<|EVALUATION|>": 151671
}
```

#### 4a — Configure Dataset Paths

DynEvalInstruct dataset entries are already registered in `Qwen3-VL/qwen-vl-finetune/qwenvl/data/__init__.py`. Set paths through environment variables before launching training:

```bash
export DYNEVALINSTRUCT_T2IA_ANNOTATION=/path/to/dynevalinstruct_t2ia.json
export DYNEVALINSTRUCT_T2IA_DATA=/path/to/image/data
export DYNEVALINSTRUCT_IQA_ANNOTATION=/path/to/dynevalinstruct_iqa.json
export DYNEVALINSTRUCT_IQA_DATA=/path/to/image/data
```

Alternatively, edit the default paths in `qwenvl/data/__init__.py` for `DYNEVALINSTRUCT_T2IA` and `DYNEVALINSTRUCT_IQA`.

Stage 1 annotations should contain only samples whose human turn starts with `<\|T2IA\|>` or `<\|EVALUATION\|>`. Stage 2 annotations should contain only samples whose human turn starts with `<\|IQA\|>` or `<\|EVALUATION\|>`.

#### 4b — Annotation Format

Each sample follows the Qwen-VL conversation format. The task token must be placed at the start of the human instruction, before the task-specific prompt text. Do **not** place `<\|T2IA\|>`, `<\|IQA\|>`, or `<\|EVALUATION\|>` in the assistant answer.

**`<\|T2IA\|>` question generation (prompt only, no image):**
```python
{
    "conversations": [
        {
            "from": "human",
            "value": "<|T2IA|>\nPrompt: a photo of a bench\nGenerate atomic yes/no verification questions for text-to-image alignment."
        },
        {
            "from": "gpt",
            "value": "[{\"question\": \"Is a bench visible?\", \"answer\": \"yes\"}]"
        }
    ]
}
```

**`<\|IQA\|>` scene-graph and quality-question generation (image required):**
```python
{
    "image": "images/001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n<|IQA|>\nParse the image into a scene graph and generate image-quality questions for shape consistency, distortions, texture fidelity, and spatial cues."
        },
        {
            "from": "gpt",
            "value": "{\"scene_graph\": {...}, \"questions\": [...]}"
        }
    ]
}
```

**`<\|EVALUATION\|>` answering and scoring:**
```python
{
    "image": "images/001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n<|EVALUATION|>\nPrompt: a photo of a bench\nQuestions:\n1. Is a bench visible?\nAnswer each question and assign a score from 1 to 5."
        },
        {
            "from": "gpt",
            "value": "{\"answers\": [{\"question\": \"Is a bench visible?\", \"score\": 5, \"reasoning\": \"...\"}]}"
        }
    ]
}
```

At inference time, DynEval triggers `<\|T2IA\|>` first, merges the generated questions, runs `<\|EVALUATION\|>` for the T2IA score, then triggers `<\|IQA\|>`, and finally runs `<\|EVALUATION\|>` again for the IQA score.

#### 4c — Run Fine-tuning

Launch curriculum fine-tuning from `Qwen3-VL/qwen-vl-finetune/`. First run the preflight check, then train in two stages.

**DynEval-4B:**

```bash
cd Qwen3-VL/qwen-vl-finetune
python3 scripts/validate_dyneval_train_setup.py

export DYNEVALINSTRUCT_T2IA_ANNOTATION=/path/to/dynevalinstruct_t2ia.json
export DYNEVALINSTRUCT_T2IA_DATA=/path/to/image/data

# Stage 1: <|T2IA|> + <|EVALUATION|>
bash scripts/sft_qwen_3_vl_4b.sh 1

export DYNEVALINSTRUCT_IQA_ANNOTATION=/path/to/dynevalinstruct_iqa.json
export DYNEVALINSTRUCT_IQA_DATA=/path/to/image/data

# Stage 2: <|IQA|> + <|EVALUATION|>, initialized from Stage 1 checkpoint
bash scripts/sft_qwen_3_vl_4b.sh 2
```

**DynEval-2B** uses the same tokens, IDs, and curriculum. Replace the launch script with `scripts/sft_qwen_3_vl_2b.sh`:

```bash
cd Qwen3-VL/qwen-vl-finetune
python3 scripts/validate_dyneval_train_setup.py

export DYNEVALINSTRUCT_T2IA_ANNOTATION=/path/to/dynevalinstruct_t2ia.json
export DYNEVALINSTRUCT_T2IA_DATA=/path/to/image/data
bash scripts/sft_qwen_3_vl_2b.sh 1

export DYNEVALINSTRUCT_IQA_ANNOTATION=/path/to/dynevalinstruct_iqa.json
export DYNEVALINSTRUCT_IQA_DATA=/path/to/image/data
bash scripts/sft_qwen_3_vl_2b.sh 2
```

Both scripts pass:

```bash
--additional_special_tokens "<|T2IA|>,<|IQA|>,<|EVALUATION|>"
```

Keep the hyperparameter configuration unchanged unless noted above. Key settings:
- Base models: `Qwen/Qwen3-VL-4B-Instruct` or `Qwen/Qwen3-VL-2B-Instruct`
- Learning rate: `1e-5`
- Task tokens: `<\|T2IA\|>` (151669), `<\|IQA\|>` (151670), `<\|EVALUATION\|>` (151671)
- Stage 1 dataset: `dynevalinstruct_t2ia`
- Stage 2 dataset: `dynevalinstruct_iqa`
- Stage 1 output: `./output/dyneval_4b_stage1_t2ia` or `./output/dyneval_2b_stage1_t2ia`
- Stage 2 output: `./output/dyneval_4b_stage2_iqa` or `./output/dyneval_2b_stage2_iqa`

Published checkpoints are available at [`vcl-iisc/DynEval-Evaluator`](https://huggingface.co/vcl-iisc/DynEval-Evaluator) under `DynEval-2B/` and `DynEval-4B/`.

---

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

---

## Inference

### Load the 2B Evaluator

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

repo_id = "vcl-iisc/DynEval-Evaluator"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    repo_id,
    subfolder="DynEval-2B",
    dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    repo_id,
    subfolder="DynEval-2B",
)
```

### Load the 4B Evaluator

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

repo_id = "vcl-iisc/DynEval-Evaluator"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    repo_id,
    subfolder="DynEval-4B",
    dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    repo_id,
    subfolder="DynEval-4B",
)
```

---

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
