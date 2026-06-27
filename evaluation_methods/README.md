# Evaluation Methods

This directory collects third-party and baseline evaluators used for text-to-image evaluation experiments. Each subfolder is a mostly self-contained method with its own dependencies, scripts, data format, and original README.

Use this README as a common index. For exact reproduction details, check the README inside the specific evaluator folder.

## Folder Overview

| Folder | Main purpose | Typical entry points |
|---|---|---|
| `EvalMuse/` | Fine-grained text-to-image alignment and fidelity evaluation from EvalMuse. | `eval.py` |
| `GenEval2/` | GenEval 2 benchmark and Soft-TIFA style evaluation for compositional prompts. | `evaluation.py` |
| `LongT2IBench/` | Long-prompt text-to-image benchmark with graph-structured annotations. | `test_generation.py`, `test_score.py` |
| `T2I-CoReBench/` | Compositional and reasoning benchmark for text-to-image generation. | `evaluate.py` |
| `T2I-Eval/` | VLM-based T2I evaluator with online and offline inference paths. | `t2i_eval.py`, `t2i_eval_offline.py` |
| `TIIF-Bench/` | Text-in-image fidelity benchmark and OCR/VLM based evaluation scripts. | `eval/eval_with_vlm.py` |
| `UniGenBench/` | Unified generation benchmark with Gemini and vLLM evaluation workflows. |  `eval/eval_vllm.sh` |
| `dpg-bench/` | DPG-Bench score computation. | `compute-dpg-score.py` |
| `geneval/` | Original GenEval object-focused T2I alignment benchmark. | `evaluation/evaluate_images.py` |
| `tifa/` | TIFA question-answering based faithfulness evaluation. | `run_tifa.py`, `run_tifa.sh` |
| `vqascore/` | VQAScore and related image/video evaluation utilities. | `eval.py` |

## General Usage Pattern

Most evaluators follow this workflow:

1. Prepare prompt metadata in the evaluator's expected format.
2. Generate images with your text-to-image model.
3. Create an image path mapping or place images in the required folder layout.
4. Install the evaluator-specific dependencies.
5. Run the evaluator script.
6. Aggregate or summarize scores.

Because these evaluators come from different projects, their input formats are not standardized. Do not assume one evaluator's image directory layout works for another.

## Recommended Setup

Use one isolated environment per evaluator when possible. Several methods pin different versions of `torch`, `transformers`, `vllm`, `mmdet`, or VLM dependencies.

```bash
cd dyneval_code/evaluation_methods/<method>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some folders use `environment.yml`, shell scripts, or package-specific install instructions instead of `requirements.txt`. Follow the method's local README in those cases.

## Common Inputs

The common conceptual inputs are:

| Input | Meaning |
|---|---|
| Prompt file | Text prompts to evaluate. Often JSON, JSONL, CSV, or TXT. |
| Image folder | Generated images from one or more T2I models. |
| Image mapping | JSON/CSV mapping prompt IDs or prompt strings to image paths. |
| Model/evaluator checkpoint | Local or hosted VLM/reward/evaluator model. |
| Output path | JSON/CSV/text scores written by the evaluator. |

## Running Selected Evaluators

### GenEval2

```bash
cd dyneval_code/evaluation_methods/GenEval2
python evaluation.py \
  --benchmark_data geneval2_data.jsonl \
  --image_filepath_data your_image_paths.json \
  --method soft_tifa_gm \
  --output_file score_lists.json
```

Then run analysis:

```bash
python soft_tifa_analysis.py \
  --benchmark_data geneval2_data.jsonl \
  --score_data score_lists.json
```

### GenEval

```bash
cd dyneval_code/evaluation_methods/geneval
python evaluation/evaluate_images.py \
  <image_folder> \
  --outfile results.jsonl

python evaluation/summary_scores.py results.jsonl
```

The GenEval image folder layout is specific; see `geneval/README.md`.

### DPG-Bench

```bash
cd dyneval_code/evaluation_methods/dpg-bench
python compute-dpg-score.py \
  --image-root <image_root> \
  --output <output_file>
```

Check `compute-dpg-score.py --help` for the exact arguments in this copy.

### TIFA

```bash
cd dyneval_code/evaluation_methods/tifa
bash run_tifa.sh
```

For custom usage, see `tifa/README.md`; TIFA expects question-answer files and image mappings.

### VQAScore

```bash
cd dyneval_code/evaluation_methods/vqascore
python eval.py --help
```

VQAScore has multiple scripts for image, video, API, and local model evaluation. Start with `vqascore/README.md`.

### T2I-CoReBench

```bash
cd dyneval_code/evaluation_methods/T2I-CoReBench
bash scripts/eval.sh
```

The evaluator supports several VLM backends, including vLLM and API-based models. See `T2I-CoReBench/README.md`.

### UniGenBench

Gemini-based evaluation:

```bash
cd dyneval_code/evaluation_methods/UniGenBench
bash eval/eval_gemini.sh --help
```

vLLM-based evaluation:

```bash
bash eval/eval_vllm.sh --help
```

The vLLM script expects a running OpenAI-compatible vLLM server.

## API Keys And External Services

Some evaluators call hosted VLMs. Common environment variables include:

```bash
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export GEMINI_BASE_URL=...
export VLLM_API_URL=http://localhost:8080
```




