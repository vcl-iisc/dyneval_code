import argparse
import json
from pathlib import Path

from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_FILE = Path("DYNEVAL-250K-PROMPTS.json")
QUESTIONS_DIR = None
IMAGES_ROOT = None
OUTPUT_DIR = Path("iqa_outputs")
DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
FP8_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"

SCENE_GRAPH_PROMPT = """Here is an image generated for this prompt: "{prompt}".

Now generate a scene graph by considering only the image. Consider the text prompt for your reference. The scene graph should represent the objects present in the image as nodes, and edges should represent the relationship between the nodes (objects).

We are doing an image quality assessment. We have already done the text-to-image alignment, so you don't need to generate questions to check the alignment. To assess the quality of each object in the image, decompose each object into its respective part-label object attributes and evaluate each segment individually with respect to several objectives, such as shape, distortion, texture, etc.

Now generate yes- or no-based questions to do the above analysis.
Score image between 1-5.

Return ONLY JSON in this format:
{{
  "scene_graph": {{
    "nodes": [
      {{"id": "object_1", "label": "...", "attributes": ["..."]}}
    ],
    "edges": [
      {{"source": "object_1", "target": "object_2", "relationship": "..."}}
    ]
  }},
  "questions": [
    {{"question": "...", "answer": "yes/no"}}
  ],
  "score": 1
}}
"""

QUALITY_QUESTION_PROMPT = """We are doing an image quality assessment. We have already done the text-to-image alignment, so you don't need to generate questions to check the alignment. To check the quality of each object present in the image, decompose each object into its respective part-label object attributes and evaluate each segment individually with respect to several objectives, such as shape, distortions, texture, etc.

Now generate yes or no based questions to do the above analysis.

Return ONLY JSON as a list:
[
  {{"question": "...", "answer": "yes/no"}}
]
"""


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
                    "prompt": prompt,
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


def normalize_questions(data) -> list[dict[str, str]]:
    if isinstance(data, dict):
        data = data.get("questions", data.get("items", []))
    if not isinstance(data, list):
        return []

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
    return questions


def output_file_for_record(record: dict, output_dir: Path | None) -> Path:
    output_id = record.get("pair_id") or record["image_id"]
    if output_dir is not None:
        return output_dir / f"{output_id}.json"
    return BASE_DIR / "iqa_outputs" / f"{output_id}.json"


def questions_file_for_record(record: dict, questions_dir: Path | None, annotations_file: Path) -> Path | None:
    if questions_dir is not None:
        return questions_dir / f"{record.get('pair_id') or record['image_id']}.json"
    return None


def resolve_image_path(image_ref: str, images_root: Path | None, annotations_file: Path) -> Path:
    ref = Path(image_ref)

    if ref.exists():
        return ref

    candidates = [resolve_input_path(annotations_file).parent / ref]
    if images_root is not None:
        candidates.extend([images_root / ref, images_root / ref.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Image not found for image_path={image_ref}")


def payload_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and "scene_graph" in data and "questions" in data


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


def run_scene_graph(generator: QwenGenerator, prompt: str, image_path: Path, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": SCENE_GRAPH_PROMPT.format(prompt=prompt)},
            ],
        }
    ]
    return generator.generate(messages, max_new_tokens=max_new_tokens)


def run_quality_questions(
    generator: QwenGenerator,
    prompt: str,
    image_path: Path,
    scene_graph,
    max_new_tokens: int,
) -> str:
    scene_graph_text = json.dumps(scene_graph, ensure_ascii=False, indent=2)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": QUALITY_QUESTION_PROMPT.format(
                        prompt=prompt,
                        scene_graph=scene_graph_text,
                    ),
                },
            ],
        }
    ]
    return generator.generate(messages, max_new_tokens=max_new_tokens)


def process_record(
    generator: QwenGenerator,
    record: dict,
    annotations_file: Path,
    images_root: Path | None,
    questions_dir: Path | None,
    output_dir: Path | None,
    force: bool,
    scene_graph_max_new_tokens: int,
    questions_max_new_tokens: int,
) -> str:
    out_path = output_file_for_record(record, output_dir)
    if (not force) and payload_valid(out_path):
        return f"skip image={record['image_path']}: already done"

    try:
        image_path = resolve_image_path(record["image_path"], images_root, annotations_file)
        scene_graph_raw = run_scene_graph(
            generator,
            record["prompt"],
            image_path,
            scene_graph_max_new_tokens,
        )
        first_pass_response = parse_response_json(scene_graph_raw)
        if isinstance(first_pass_response, dict) and "scene_graph" in first_pass_response:
            scene_graph = first_pass_response["scene_graph"]
        elif first_pass_response is not None:
            scene_graph = first_pass_response
        else:
            scene_graph = {"raw_response": scene_graph_raw}

        questions_raw = run_quality_questions(
            generator,
            record["prompt"],
            image_path,
            scene_graph,
            questions_max_new_tokens,
        )
        questions_parsed = parse_response_json(questions_raw)
        questions = normalize_questions(questions_parsed)

        q_path = questions_file_for_record(record, questions_dir, annotations_file)
        if q_path is not None and questions:
            q_path.parent.mkdir(parents=True, exist_ok=True)
            q_path.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")

        payload = {
            "item_key": record.get("item_key", ""),
            "pair_id": record.get("pair_id", ""),
            "text_id": record.get("text_id", ""),
            "model": record.get("model", "unknown"),
            "image_id": record.get("image_id", ""),
            "image_ref": record.get("image_path", ""),
            "image_path": str(image_path),
            "prompt": record["prompt"],
            "scene_graph": scene_graph,
            "first_pass_response": first_pass_response,
            "scene_graph_raw_response": scene_graph_raw,
            "questions": questions,
            "questions_raw_response": questions_raw,
            "questions_file": str(q_path) if q_path is not None else "",
            "group_id": record.get("group_id", ""),
            "source_item_id": record.get("source_item_id", ""),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"saved {out_path}"
    except Exception as exc:
        err_path = out_path.with_suffix(".error.txt")
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(str(exc), encoding="utf-8")
        return f"error image={record.get('image_path', '')}: {exc}"


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
    parser.add_argument("--scene-graph-max-new-tokens", type=int, default=1024)
    parser.add_argument("--questions-max-new-tokens", type=int, default=1024)
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
    if args.questions_dir is not None:
        args.questions_dir.mkdir(parents=True, exist_ok=True)

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
                annotations_file,
                args.images_root,
                args.questions_dir,
                args.output_dir,
                args.force,
                args.scene_graph_max_new_tokens,
                args.questions_max_new_tokens,
            )
        )


if __name__ == "__main__":
    main()
