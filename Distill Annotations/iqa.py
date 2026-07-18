import argparse
import json
import re
from pathlib import Path

from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_FILE = Path("DYNEVAL-250K-PROMPTS.json")
IMAGES_ROOT = None
OUTPUT_DIR = Path("iqa_outputs")
ANSWERS_DIR = Path("iqa_answers")
DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
FP8_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"

# Prompts mirror the two-step IQA flow in inference/run-inference.py. Step 1 builds
# a scene graph (nodes + edges only). Step 2 decomposes each node into
# rendering-quality yes/no questions. Step 3 answers/scores them against the image.
# The <IQA> task token is intentionally NOT sent to the teacher VLM.
SCENE_GRAPH_PROMPT = """Here is an image generated for this prompt "{prompt}".
Now generate a scene graph by considering only the image. Consider the text prompt for your reference.
Return only valid JSON with exactly these keys: "nodes" and "edges".
"nodes" must be an array of visible objects, where each object has "id", "label", and "attributes".
"edges" must be an array of relationships, where each relationship has "source", "relation", and "target".
Do not write markdown, bullets, or prose."""

QUALITY_QUESTION_PROMPT = """We are doing an image quality assessment. The generated image is provided to you along with a list of object nodes JSON.
We have ALREADY done the text-to-image alignment. Therefore you must NOT ask presence, attribute, color, count, or alignment questions (for example: "Is there a bench?", "Is the bench wooden?", "Are the flowers red?", "Is the wall white?"). Such questions are forbidden.
Instead, for each object node, decompose it into its parts and generate yes/no questions that judge ONLY the rendering QUALITY of how that object is drawn in the image. Focus on perceptual defects such as: shape correctness and geometry, structural distortions or deformities, unnatural or broken proportions, blur or smearing, texture artifacts, melting/warping, incorrect or impossible 3D spatial structure, and boundary/edge coherence.
Each question must be phrased so that a well-rendered, high-quality object gets the target answer "yes".
Examples of the required style: "Are the bench slats straight and free of distortion?", "Are the flower petals rendered with clean edges and no smearing?", "Does the wall surface have a consistent texture without artifacts?".
Inspect the provided image to decide the target answer.
Return only valid JSON with exactly one key: "questions".
Each question object must contain "node_id", "question", and "target_answer".
Do not write markdown, bullets, or prose."""

ANSWER_PROMPT = """Here is a prompt, an image, and a list of image-quality yes/no questions with their target answers.

Original prompt: "{prompt}"

Questions:
{questions}

Answer each question by looking ONLY at the image, then score how well the image matches the target answers on a scale of 1-5.

Instructions:
1. Answer each question with "yes" or "no" based ONLY on what you can see in the image.
2. Compare each answer with its target answer and mark "correct" true when they match.
3. Generate one overall score from 1 to 5, where 5 means the image matches all or almost all target answers (high rendering quality) and 1 means a severe quality failure.

Return ONLY JSON in this format:
{{
  "answers": [
    {{"question": "...", "answer": "yes/no", "target_answer": "yes/no", "correct": true, "reasoning": "..."}}
  ],
  "score": 1
}}
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


def normalize_yes_no(value) -> str:
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return ""


def normalize_score(value) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return score if 1 <= score <= 5 else 0


def normalize_scene_graph(raw) -> dict[str, list[dict]]:
    """Coerce the teacher scene graph into {nodes:[{id,label,attributes}], edges:[{source,relation,target}]}."""
    nodes: list[dict] = []
    edges: list[dict] = []
    if isinstance(raw, dict) and "scene_graph" in raw and isinstance(raw["scene_graph"], dict):
        raw = raw["scene_graph"]
    if not isinstance(raw, dict):
        return {"nodes": nodes, "edges": edges}

    raw_nodes = raw.get("nodes", raw.get("Nodes", []))
    raw_edges = raw.get("edges", raw.get("Edges", []))

    if isinstance(raw_nodes, list):
        for idx, item in enumerate(raw_nodes, start=1):
            if isinstance(item, dict):
                node_id = str(item.get("id", item.get("node_id", f"object_{idx}"))).strip() or f"object_{idx}"
                label = str(item.get("label", item.get("name", item.get("object", "")))).strip()
                attributes = item.get("attributes", [])
            else:
                node_id = f"object_{idx}"
                label = str(item).strip()
                attributes = []
            if label:
                nodes.append(
                    {
                        "id": node_id,
                        "label": label,
                        "attributes": attributes if isinstance(attributes, list) else [str(attributes)],
                    }
                )

    if isinstance(raw_edges, list):
        for item in raw_edges:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            relation = str(item.get("relation", item.get("relationship", item.get("predicate", "")))).strip()
            target = str(item.get("target", "")).strip()
            if source and relation and target:
                edges.append({"source": source, "relation": relation, "target": target})

    return {"nodes": nodes, "edges": edges}


def normalize_questions(data) -> list[dict[str, str]]:
    if isinstance(data, dict):
        data = data.get("questions", data.get("items", []))
    if not isinstance(data, list):
        return []

    questions: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        node_id = str(item.get("node_id", item.get("node", ""))).strip()
        target = normalize_yes_no(item.get("target_answer", item.get("answer", "")))
        if not question or not target:
            continue
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        questions.append({"node_id": node_id, "question": question, "target_answer": target})
    return questions


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
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip() if output_text else ""


def build_questions_text(questions: list[dict[str, str]]) -> str:
    lines = []
    for i, item in enumerate(questions, start=1):
        lines.append(f"{i}. Question: {item['question']}\n   Target answer: {item['target_answer']}")
    return "\n".join(lines)


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
    scene_graph: dict,
    max_new_tokens: int,
) -> str:
    # Match run_inference.py: pass only node ids and labels (no attributes/edges) so
    # the model generates rendering-quality questions rather than re-verifying
    # attributes, which would produce forbidden alignment questions.
    raw_nodes = scene_graph.get("nodes", []) if isinstance(scene_graph, dict) else []
    nodes = [
        {"node_id": str(node.get("id", "")), "label": str(node.get("label", ""))}
        for node in raw_nodes
        if isinstance(node, dict) and str(node.get("label", "")).strip()
    ]
    user_text = (
        f"Prompt: {prompt}\n\n"
        f"Object nodes JSON:\n{json.dumps(nodes, ensure_ascii=False, indent=2)}\n\n"
        f"{QUALITY_QUESTION_PROMPT}"
    )
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


def run_answer(
    generator: QwenGenerator,
    prompt: str,
    image_path: Path,
    questions: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": ANSWER_PROMPT.format(prompt=prompt, questions=build_questions_text(questions)),
                },
            ],
        }
    ]
    return generator.generate(messages, max_new_tokens=max_new_tokens)


def normalize_answers(raw: str, questions: list[dict[str, str]]) -> tuple[list[dict], int, object]:
    parsed = parse_response_json(raw)
    payload = parsed.get("answers", []) if isinstance(parsed, dict) else parsed

    answers: list[dict] = []
    if isinstance(payload, list):
        for i, question in enumerate(questions):
            model_answer = ""
            reasoning = ""
            correct = None
            if i < len(payload):
                item = payload[i]
                if isinstance(item, dict):
                    model_answer = normalize_yes_no(item.get("answer", ""))
                    reasoning = str(item.get("reasoning", "")).strip()
                    if isinstance(item.get("correct"), bool):
                        correct = item["correct"]
                else:
                    model_answer = normalize_yes_no(item)
            if correct is None and model_answer:
                correct = model_answer == question["target_answer"]
            answers.append(
                {
                    "question": question["question"],
                    "answer": model_answer,
                    "target_answer": question["target_answer"],
                    "correct": correct,
                    "reasoning": reasoning,
                }
            )
    else:
        for question in questions:
            answers.append(
                {
                    "question": question["question"],
                    "answer": "",
                    "target_answer": question["target_answer"],
                    "correct": None,
                    "reasoning": "Failed to parse JSON answers.",
                }
            )

    score = normalize_score(parsed.get("score")) if isinstance(parsed, dict) else 0
    if score == 0 and answers:
        correct_count = sum(1 for item in answers if item.get("correct") is True)
        score = max(1, min(5, round((correct_count / len(answers)) * 4 + 1)))

    return answers, score, parsed


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


def output_file_for_record(record: dict, output_dir: Path) -> Path:
    return output_dir / f"{record.get('pair_id') or record['image_id']}.json"


def answers_file_for_record(record: dict, answers_dir: Path) -> Path:
    return answers_dir / f"{record.get('pair_id') or record['image_id']}.json"


def scene_graph_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and "scene_graph" in data and "questions" in data


def answers_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return isinstance(data, dict) and isinstance(data.get("answers"), list) and data.get("score") is not None


def process_record(
    generator: QwenGenerator,
    record: dict,
    annotations_file: Path,
    images_root: Path | None,
    output_dir: Path,
    answers_dir: Path,
    force: bool,
    scene_graph_max_new_tokens: int,
    questions_max_new_tokens: int,
    answer_max_new_tokens: int,
) -> str:
    out_path = output_file_for_record(record, output_dir)
    a_path = answers_file_for_record(record, answers_dir)
    if (not force) and scene_graph_valid(out_path) and answers_valid(a_path):
        return f"skip pair_id={record['pair_id']}: already done"

    try:
        image_path = resolve_image_path(record["image_path"], images_root, annotations_file)

        # Stage 1: scene graph (nodes + edges only).
        scene_graph_raw = run_scene_graph(generator, record["prompt"], image_path, scene_graph_max_new_tokens)
        scene_graph = normalize_scene_graph(parse_response_json(scene_graph_raw))

        # Stage 2: rendering-quality decomposition questions.
        questions_raw = run_quality_questions(
            generator, record["prompt"], image_path, scene_graph, questions_max_new_tokens
        )
        questions = normalize_questions(parse_response_json(questions_raw))

        out_payload = {
            "item_key": record.get("item_key", ""),
            "pair_id": record.get("pair_id", ""),
            "text_id": record.get("text_id", ""),
            "model": record.get("model", "unknown"),
            "image_id": record.get("image_id", ""),
            "image_ref": record.get("image_path", ""),
            "image_path": str(image_path),
            "prompt": record["prompt"],
            "scene_graph": scene_graph,
            "scene_graph_raw_response": scene_graph_raw,
            "questions": questions,
            "questions_raw_response": questions_raw,
            "group_id": record.get("group_id", ""),
            "source_item_id": record.get("source_item_id", ""),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        # Stage 3: answer + score the quality questions against the image.
        if questions:
            answer_raw = run_answer(generator, record["prompt"], image_path, questions, answer_max_new_tokens)
            answers, score, parsed = normalize_answers(answer_raw, questions)
        else:
            answer_raw, answers, score, parsed = "", [], 0, None

        a_payload = {
            "item_key": record.get("item_key", ""),
            "pair_id": record.get("pair_id", ""),
            "text_id": record.get("text_id", ""),
            "model": record.get("model", "unknown"),
            "image_id": record.get("image_id", ""),
            "image_ref": record.get("image_path", ""),
            "image_path": str(image_path),
            "prompt": record["prompt"],
            "questions_file": str(out_path),
            "answers": answers,
            "score": score,
            "raw_response": answer_raw,
            "parsed_response": parsed,
            "group_id": record.get("group_id", ""),
            "source_item_id": record.get("source_item_id", ""),
        }
        a_path.parent.mkdir(parents=True, exist_ok=True)
        a_path.write_text(json.dumps(a_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"saved {out_path.name} + {a_path.name} for image={record['image_path']}"
    except Exception as exc:
        err_path = a_path.with_suffix(".error.txt")
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(str(exc), encoding="utf-8")
        return f"error pair_id={record.get('pair_id', '')}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher IQA annotation: build a scene graph, generate rendering-quality yes/no questions, then answer/score them 1-5 against the image."
    )
    parser.add_argument("--annotations-file", type=Path, default=ANNOTATIONS_FILE)
    parser.add_argument("--images-root", type=Path, default=IMAGES_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Scene-graph + questions output directory (consumed by build_iqa_sft.py --scene-graph-dir).")
    parser.add_argument("--answers-dir", type=Path, default=ANSWERS_DIR, help="Answer/score output directory (consumed by build_iqa_sft.py --answers-dir).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("auto", "transformers", "vllm"), default="auto")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--scene-graph-max-new-tokens", type=int, default=1024)
    parser.add_argument("--questions-max-new-tokens", type=int, default=1024)
    parser.add_argument("--answer-max-new-tokens", type=int, default=1024)
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.answers_dir.mkdir(parents=True, exist_ok=True)

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
                args.output_dir,
                args.answers_dir,
                args.force,
                args.scene_graph_max_new_tokens,
                args.questions_max_new_tokens,
                args.answer_max_new_tokens,
            )
        )


if __name__ == "__main__":
    main()
