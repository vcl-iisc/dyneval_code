import argparse
import json
import re
from pathlib import Path

from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_FILE = Path("DYNEVAL-250K-PROMPTS.json")
QUESTIONS_DIR = None
IMAGES_ROOT = None
OUTPUT_DIR = None
DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
FP8_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"


def load_json(path: Path):
    with resolve_input_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_annotations(annotations_file: Path) -> list[dict]:
    data = load_json(annotations_file)

    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        records: list[dict] = []
        for item in data["prompts"]:
            if not isinstance(item, dict):
                continue
            pair_id = str(item.get("pair_id", "")).strip()
            prompt = str(item.get("prompt", "")).strip()
            image_path = str(item.get("image_path", "")).strip()
            if not pair_id or not prompt or not image_path:
                continue
            records.append(
                {
                    "item_key": pair_id,
                    "pair_id": pair_id,
                    "text_id": str(item.get("text_id", pair_id)).strip() or pair_id,
                    "text": prompt,
                    "model": str(item.get("generation_model", "unknown")).strip() or "unknown",
                    "image_id": pair_id,
                    "image_path": image_path,
                    "questions_file": str(item.get("questions_file", "")).strip(),
                    "response_file": str(item.get("response_file", "")).strip(),
                    "group_id": str(item.get("group_id", "")).strip(),
                    "source_item_id": str(item.get("source_item_id", "")).strip(),
                }
            )
        if records:
            return records
        raise ValueError("No valid prompt/image records found in prompt mapping JSON")

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


def parse_questions(question_path: Path) -> list[dict[str, str]]:
    data = load_json(question_path)
    if not isinstance(data, list):
        raise ValueError("questions file must be a JSON list")

    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip().lower()
        if answer.startswith("yes"):
            answer = "yes"
        elif answer.startswith("no"):
            answer = "no"
        else:
            answer = ""
        if q:
            out.append({"question": q, "answer": answer})

    if out:
        return out
    raise ValueError("no valid questions found")


def build_questions_text(questions: list[dict[str, str]]) -> str:
    lines = []
    for i, q in enumerate(questions, start=1):
        answer_text = q.get("answer", "")
        if answer_text:
            lines.append(f"{i}. Question: {q['question']}\n   Target answer: {answer_text}")
        else:
            lines.append(f"{i}. Question: {q['question']}")
    return "\n".join(lines)


def build_user_text(prompt: str, questions: list[dict[str, str]]) -> str:
    return (
        f'Original prompt: "{prompt}"\n\n'
        "You are given an image and a set of Yes/No questions.\n\n"
        "Now answer these questions for this image and generate a score given the target answers on a scale of 1-5.\n\n"
        "Instructions:\n"
        "1. Answer each question with yes or no based ONLY on the image.\n"
        "2. Compare your answers with the expected target answers.\n"
        "3. Assign a per-question score from 1 to 5 based on correctness.\n"
        "4. Compute one overall score from 1 to 5.\n\n"
        "Return JSON in the following format:\n"
        "{\n"
        '  "answers": [\n'
        '    {"question": "...", "answer": "yes/no", "target_answer": "yes/no", "score": 5, "reasoning": "..."},\n'
        '    ...\n'
        "  ],\n"
        '  "score": <number between 1 and 5>\n'
        "}\n\n"
        f"Questions:\n{build_questions_text(questions)}\n\n"
        "Return ONLY JSON. No extra text."
    )


def parse_response_json(raw: str):
    text = (raw or "").strip()
    if text == "":
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for opener in ("{", "["):
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


def normalize_confidence_score(value) -> int:
    """Normalize a value to a confidence score 1-5. Returns 0 if invalid."""
    try:
        score = int(value)
    except (TypeError, ValueError):
        s = str(value).strip().lower()
        if s.startswith("yes") or re.search(r"\byes\b", s):
            return 5
        if s.startswith("no") or re.search(r"\bno\b", s):
            return 1
        return 0
    if 1 <= score <= 5:
        return score
    return 0


def extract_answers(raw: str, questions: list[dict[str, str]]) -> list[dict]:
    parsed = parse_response_json(raw)
    answers_payload = parsed

    if isinstance(parsed, dict):
        answers_payload = parsed.get("answers", parsed)

    out = []
    if isinstance(answers_payload, list):
        for i, q in enumerate(questions):
            score = 0
            reasoning = ""
            if i < len(answers_payload):
                item = answers_payload[i]
                if isinstance(item, dict):
                    raw_score = item.get("score", item.get("answer", ""))
                    score = normalize_confidence_score(raw_score)
                    reasoning = item.get("reasoning", "")
                else:
                    score = normalize_confidence_score(item)
            out.append(
                {
                    "question": q["question"],
                    "target_answer": q.get("answer", ""),
                    "reasoning": reasoning,
                    "score": score,
                }
            )
        return out

    tokens = re.findall(r"\b([1-5])\b", raw or "")
    for i, q in enumerate(questions):
        score = normalize_confidence_score(tokens[i]) if i < len(tokens) else 0
        out.append(
            {
                "question": q["question"],
                "target_answer": q.get("answer", ""),
                "reasoning": "Failed to parse JSON reasoning.",
                "score": score,
            }
        )
    return out


def answer_payload_valid(path: Path) -> bool:
    if not path.exists():
        return False

    try:
        data = load_json(path)
    except Exception:
        return False

    if not isinstance(data, dict) or not isinstance(data.get("answers"), list):
        return False
    rubric = data.get("rubric")
    return rubric is None or isinstance(rubric, dict)


def resolve_image_path(image_ref: str, images_root: Path | None, annotations_file: Path) -> Path:
    ref = Path(image_ref)

    if ref.exists():
        return ref

    candidates = [
        annotations_file.parent / ref,
    ]
    if images_root is not None:
        candidates.extend([images_root / ref, images_root / ref.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found for image_path={image_ref}")


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

def run_one(generator: QwenGenerator, prompt: str, questions, image_path: Path, max_new_tokens: int) -> str:
    user_text = build_user_text(prompt, questions)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    return generator.generate(messages, max_new_tokens=max_new_tokens)


def questions_file_for_record(record: dict, questions_dir: Path | None, annotations_file: Path) -> Path:
    if questions_dir is not None:
        return questions_dir / f"{record.get('pair_id') or record['text_id']}.json"
    questions_file = record.get("questions_file", "")
    if questions_file:
        return resolve_input_path(annotations_file).parent / questions_file
    return BASE_DIR / "questions" / f"{record['text_id']}.json"


def output_file_for_image(record: dict, output_dir: Path | None, annotations_file: Path) -> Path:
    if output_dir is not None:
        output_id = record.get("pair_id") or f"{record['model']}_{record['text_id']}_{record['image_id']}"
        return output_dir / f"{output_id}.json"
    response_file = record.get("response_file", "")
    if response_file:
        return resolve_input_path(annotations_file).parent / response_file
    return BASE_DIR / "answers" / f"{record['model']}_{record['text_id']}_{record['image_id']}.json"


def process_record(
    generator: QwenGenerator,
    record: dict,
    questions_dir: Path | None,
    annotations_file: Path,
    images_root: Path | None,
    output_dir: Path | None,
    force: bool,
    max_new_tokens: int,
):
    text_id = record["text_id"]
    prompt_text = record["text"]
    image_ref = record["image_path"]

    q_path = questions_file_for_record(record, questions_dir, annotations_file)
    if not q_path.exists():
        return f"skip text_id={text_id}: missing questions file {q_path}"

    try:
        questions = parse_questions(q_path)
    except Exception as exc:
        return f"skip text_id={text_id}: bad questions ({exc})"

    out_path = output_file_for_image(record, output_dir, annotations_file)
    if (not force) and answer_payload_valid(out_path):
        return f"skip image={image_ref}: already done"

    try:
        image_path = resolve_image_path(image_ref, images_root, annotations_file)
        raw = run_one(generator, prompt_text, questions, image_path, max_new_tokens)
        answers = extract_answers(raw, questions)

        payload = {
            "item_key": record.get("item_key", ""),
            "pair_id": record.get("pair_id", ""),
            "text_id": text_id,
            "model": record.get("model", "unknown"),
            "image_id": record.get("image_id", ""),
            "image_ref": image_ref,
            "image_path": str(image_path),
            "prompt": prompt_text,
            "questions_file": str(q_path),
            "answers": answers,
            "raw_response": raw,
            "human_quality_mean": record.get("human_quality_mean"),
            "group_id": record.get("group_id", ""),
            "source_item_id": record.get("source_item_id", ""),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return f"saved {out_path}"
    except Exception as exc:
        err_path = out_path.with_suffix(".error.txt")
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(str(exc), encoding="utf-8")
        return f"error image={image_ref}: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-file", type=Path, default=ANNOTATIONS_FILE)
    parser.add_argument("--questions-dir", type=Path, default=QUESTIONS_DIR)
    parser.add_argument("--images-root", type=Path, default=IMAGES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
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
    records = load_annotations(annotations_file)
    total = len(records)

    start = max(0, args.start_idx)
    end = total if args.end_idx is None else min(total, args.end_idx)

    if start >= end:
        print(f"nothing to process: start={start}, end={end}, total={total}")
        return

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = QwenGenerator(
        args.model,
        args.backend,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
        args.temperature,
    )

    print(
        f"processing [{start}, {end}) from total={total}, one image at a time, "
        f"model={args.model}, backend={generator.backend}"
    )
    for idx in range(start, end):
        print(
            process_record(
                generator,
                records[idx],
                args.questions_dir,
                annotations_file,
                args.images_root,
                args.output_dir,
                args.force,
                args.max_new_tokens,
            )
        )


if __name__ == "__main__":
    main()
