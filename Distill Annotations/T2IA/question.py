import argparse
import json
from pathlib import Path

from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Instruct"
FP8_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_FILE = Path("DYNEVAL-250K-PROMPTS.json")
QUESTIONS_DIR = None
IMAGES_ROOT = None
OUTPUT_DIR = None


QUESTION_TEMPLATE = """
Here is a prompt: "{prompt}"

Generate atomic yes/no verification questions for text-to-image alignment, including distortion checks.
Also generate the target answer for each question.
Make sure there is no repeated questions per element/attribute, and all questions are unique, to maintain atomicity.


Return JSON as a list:
[
  {{"question": "...", "answer": "yes/no"}},
  ...
]
"""


## step-1b
DISTORTION_QUESTION_TEMPLATE = """
Here is a prompt: "{prompt}"

Generate yes or no distortion-based questions for text-to-image alignment checks and also generate the target answer.


Return JSON as a list:
[
  {{"question": "...", "answer": "yes/no"}},
  ...
]
"""


def load_prompt_records(annotations_file: str | Path) -> list[dict[str, str]]:
    path = resolve_input_path(Path(annotations_file))
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        records: list[dict[str, str]] = []
        for item in data["prompts"]:
            if not isinstance(item, dict):
                continue
            pair_id = str(item.get("pair_id", "")).strip()
            prompt = str(item.get("prompt", "")).strip()
            if not pair_id or not prompt:
                continue
            records.append(
                {
                    "item_id": pair_id,
                    "prompt": prompt,
                    "image_path": str(item.get("image_path", "")).strip(),
                    "questions_file": str(item.get("questions_file", "")).strip(),
                }
            )
        if records:
            return records
        raise ValueError("No valid pair_id/prompt records found in prompt mapping JSON")

    raise ValueError("annotations file must be a prompt mapping JSON object with a prompts list")


def resolve_input_path(path: Path) -> Path:
    if path.exists():
        return path
    script_relative = BASE_DIR / path
    if script_relative.exists():
        return script_relative
    repo_relative = BASE_DIR.parents[1] / path
    if repo_relative.exists():
        return repo_relative
    return path


def use_vllm_backend(model_name: str, backend: str) -> bool:
    if backend == "vllm":
        return True
    if backend == "transformers":
        return False
    return model_name == FP8_MODEL or model_name.endswith("-FP8")


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    has_vision = any(
        isinstance(content, dict) and content.get("type") in {"image", "video"}
        for message in messages
        for content in message.get("content", [])
    )
    if not has_vision:
        return {"prompt": text}

    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    inputs = {"prompt": text}
    if mm_data:
        inputs["multi_modal_data"] = mm_data
    if video_kwargs:
        inputs["mm_processor_kwargs"] = video_kwargs
    return inputs


class QwenGenerator:
    def __init__(
        self,
        model_name: str,
        backend: str,
        gpu_memory_utilization: float,
        tensor_parallel_size: int | None,
        temperature: float,
    ):
        self.model_name = model_name
        self.backend = "vllm" if use_vllm_backend(model_name, backend) else "transformers"
        self.temperature = temperature
        self.processor = AutoProcessor.from_pretrained(model_name)

        if self.backend == "vllm":
            import os

            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

            import torch
            from vllm import LLM

            tp_size = tensor_parallel_size or torch.cuda.device_count() or 1
            self.model = LLM(
                model=model_name,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization,
                enforce_eager=False,
                tensor_parallel_size=tp_size,
                seed=0,
            )
        else:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name,
                dtype="auto",
                device_map="auto",
            )

    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        if self.backend == "vllm":
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=max_new_tokens,
                top_k=-1,
                stop_token_ids=[],
            )
            outputs = self.model.generate(
                [prepare_inputs_for_vllm(messages, self.processor)],
                sampling_params=sampling_params,
            )
            return outputs[0].outputs[0].text.strip() if outputs else ""

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip() if output_text else ""

def qwen_generate(generator: QwenGenerator, prompt: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
    ]
    return generator.generate(messages, max_new_tokens=max_new_tokens)


def parse_response_json(raw: str):
    text = (raw or "").strip()
    if text == "":
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for opener in ("[", "{"):
        start = text.find(opener)
        while start >= 0:
            try:
                data, _ = decoder.raw_decode(text[start:])
                if isinstance(data, (list, dict)):
                    return data
            except json.JSONDecodeError:
                pass
            start = text.find(opener, start + 1)

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        try:
            return json.loads("\n".join(lines).strip())
        except json.JSONDecodeError:
            return None

    return None


def normalize_questions(data) -> list[dict[str, str]]:
    if isinstance(data, dict):
        data = data.get("questions", data.get("items", []))
    if not isinstance(data, list):
        raise ValueError("model did not return a JSON list")

    questions: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip().lower()
        if answer.startswith("yes"):
            answer = "yes"
        elif answer.startswith("no"):
            answer = "no"
        else:
            continue
        if question:
            questions.append({"question": question, "answer": answer})

    if not questions:
        raise ValueError("model returned no valid question/answer pairs")
    return questions


def generate_questions(generator: QwenGenerator, prompt: str, max_new_tokens: int) -> list[dict[str, str]]:
    """Generate standard and distortion-based questions for one prompt."""
    standard_raw = qwen_generate(
        generator,
        QUESTION_TEMPLATE.format(prompt=prompt),
        max_new_tokens=max_new_tokens,
    )
    standard_data = parse_response_json(standard_raw)
    if standard_data is None:
        raise ValueError(f"model did not return parseable JSON for standard questions: {standard_raw}")

    distortion_raw = qwen_generate(
        generator,
        DISTORTION_QUESTION_TEMPLATE.format(prompt=prompt),
        max_new_tokens=max_new_tokens,
    )
    distortion_data = parse_response_json(distortion_raw)
    if distortion_data is None:
        raise ValueError(f"model did not return parseable JSON for distortion questions: {distortion_raw}")

    questions = normalize_questions(standard_data) + normalize_questions(distortion_data)
    unique_questions: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for item in questions:
        key = item["question"].strip().lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        unique_questions.append(item)
    return unique_questions


def output_file_for_record(record: dict[str, str], annotations_file: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir / f"{record['item_id']}.json"
    questions_file = record.get("questions_file", "")
    if questions_file:
        return resolve_input_path(annotations_file).parent / questions_file
    return BASE_DIR / "questions" / f"{record['item_id']}.json"


def error_file_for_record(record: dict[str, str], annotations_file: Path, output_dir: Path | None) -> Path:
    output_file = output_file_for_record(record, annotations_file, output_dir)
    return output_file.with_suffix(".error.txt")


def has_valid_output(record: dict[str, str], annotations_file: Path, output_dir: Path | None) -> bool:
    output_file = output_file_for_record(record, annotations_file, output_dir)
    if not output_file.exists():
        return False

    try:
        data = json.loads(output_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, dict):
                return False
            if not {"question", "answer"}.issubset(item):
                return False
        return True
    except (json.JSONDecodeError, OSError, ValueError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-file", type=Path, default=ANNOTATIONS_FILE)
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--images-root", type=Path, default=IMAGES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--backend", choices=("auto", "transformers", "vllm"), default="auto")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    annotations_file = resolve_input_path(args.annotations_file)
    questions_output_dir = args.questions_dir if args.questions_dir is not None else args.output_dir
    prompt_records = load_prompt_records(annotations_file)
    if questions_output_dir is not None:
        questions_output_dir.mkdir(parents=True, exist_ok=True)

    pending_jobs: list[dict[str, str]] = []
    skipped_count = 0
    start = max(0, args.start_idx)
    end = len(prompt_records) if args.end_idx is None else min(len(prompt_records), args.end_idx)
    for record in prompt_records[start:end]:
        output_file = output_file_for_record(record, annotations_file, questions_output_dir)
        if (not args.force) and has_valid_output(record, annotations_file, questions_output_dir):
            skipped_count += 1
            print(f"Skipping existing valid output for {output_file}")
            continue
        pending_jobs.append(record)

    print(
        f"Found {len(prompt_records)} prompt records. Processing [{start}, {end}) one prompt at a time. "
        f"Skipping {skipped_count} completed prompts."
    )
    if not pending_jobs:
        return

    generator = QwenGenerator(
        args.model,
        args.backend,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
        args.temperature,
    )
    print(f"Loaded model={args.model}, backend={generator.backend}")
    for record in pending_jobs:
        try:
            questions = generate_questions(generator, record["prompt"], args.max_new_tokens)
            output_file = output_file_for_record(record, annotations_file, questions_output_dir)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(json.dumps(questions, indent=2), encoding="utf-8")
            print(f"Saved {output_file} for prompt: {record['prompt']}")
        except Exception as exc:
            err_file = error_file_for_record(record, annotations_file, questions_output_dir)
            err_file.parent.mkdir(parents=True, exist_ok=True)
            err_file.write_text(str(exc), encoding="utf-8")
            print(f"Failed {record['item_id']}: {exc}")


if __name__ == "__main__":
    main()
