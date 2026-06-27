#!/usr/bin/env python3
"""Convert teacher distillation outputs into DynEvalInstruct training JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from task_tokens import format_evaluation_human, format_iqa_human, format_t2ia_human


def resolve_input_path(path: Path, base_dir: Path) -> Path:
    if path.exists():
        return path
    for candidate in (base_dir / path, base_dir.parent / path):
        if candidate.exists():
            return candidate
    return path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_prompt_records(annotations_file: Path) -> list[dict]:
    data = load_json(annotations_file)
    if not isinstance(data, dict) or not isinstance(data.get("prompts"), list):
        raise ValueError("annotations file must contain a prompts list")

    records: list[dict] = []
    for item in data["prompts"]:
        if not isinstance(item, dict):
            continue
        pair_id = str(item.get("pair_id", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        image_path = str(item.get("image_path", "")).strip()
        if not pair_id or not prompt:
            continue
        records.append(
            {
                "pair_id": pair_id,
                "prompt": prompt,
                "image_path": image_path,
                "questions_file": str(item.get("questions_file", "")).strip(),
                "response_file": str(item.get("response_file", "")).strip(),
            }
        )
    if not records:
        raise ValueError("no valid prompt records found")
    return records


def resolve_record_path(
    record: dict,
    field_name: str,
    fallback_dir: Path | None,
    pair_id: str,
    annotations_file: Path,
) -> Path | None:
    field_value = record.get(field_name, "")
    if field_value:
        return resolve_input_path(Path(field_value), annotations_file.parent)
    if fallback_dir is not None:
        candidate = fallback_dir / f"{pair_id}.json"
        if candidate.exists():
            return candidate
    return None


def format_gpt_evaluation_value(answers: list[dict]) -> str:
    payload = {"answers": answers}
    scores = [item.get("score") for item in answers if isinstance(item.get("score"), int)]
    if scores:
        payload["score"] = round(sum(scores) / len(scores))
    return json.dumps(payload, ensure_ascii=False)


def build_stage1_samples(
    record: dict,
    annotations_file: Path,
    t2ia_questions_dir: Path | None,
    t2ia_answers_dir: Path | None,
) -> list[dict]:
    pair_id = record["pair_id"]
    prompt = record["prompt"]
    samples: list[dict] = []

    questions_path = resolve_record_path(
        record, "questions_file", t2ia_questions_dir, pair_id, annotations_file
    )
    if questions_path is not None and questions_path.exists():
        questions = load_json(questions_path)
        if isinstance(questions, list) and questions:
            samples.append(
                {
                    "id": f"{pair_id}_t2ia_questions",
                    "conversations": [
                        {"from": "human", "value": format_t2ia_human(prompt)},
                        {"from": "gpt", "value": json.dumps(questions, ensure_ascii=False)},
                    ],
                }
            )

    answers_path = resolve_record_path(
        record, "response_file", t2ia_answers_dir, pair_id, annotations_file
    )
    if (
        answers_path is not None
        and answers_path.exists()
        and questions_path is not None
        and questions_path.exists()
    ):
        answer_payload = load_json(answers_path)
        questions = load_json(questions_path)
        answers = answer_payload.get("answers", []) if isinstance(answer_payload, dict) else []
        if isinstance(questions, list) and questions and isinstance(answers, list) and answers:
            image_path = str(record.get("image_path", "")).strip()
            if not image_path:
                return samples
            samples.append(
                {
                    "id": f"{pair_id}_t2ia_evaluation",
                    "image": image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": format_evaluation_human(prompt, questions),
                        },
                        {"from": "gpt", "value": format_gpt_evaluation_value(answers)},
                    ],
                }
            )

    return samples


def build_stage2_samples(
    record: dict,
    annotations_file: Path,
    iqa_outputs_dir: Path | None,
    iqa_answers_dir: Path | None,
) -> list[dict]:
    pair_id = record["pair_id"]
    prompt = record["prompt"]
    image_path = str(record.get("image_path", "")).strip()
    if not image_path:
        return []

    samples: list[dict] = []
    iqa_output_path = iqa_outputs_dir / f"{pair_id}.json" if iqa_outputs_dir else None
    if iqa_output_path is not None and iqa_output_path.exists():
        iqa_payload = load_json(iqa_output_path)
        scene_graph = iqa_payload.get("scene_graph", {})
        questions = iqa_payload.get("questions", [])
        if isinstance(questions, list) and questions:
            samples.append(
                {
                    "id": f"{pair_id}_iqa_questions",
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": format_iqa_human()},
                        {
                            "from": "gpt",
                            "value": json.dumps(
                                {"scene_graph": scene_graph, "questions": questions},
                                ensure_ascii=False,
                            ),
                        },
                    ],
                }
            )

    iqa_answer_path = iqa_answers_dir / f"{pair_id}.json" if iqa_answers_dir else None
    if iqa_answer_path is not None and iqa_answer_path.exists():
        answer_payload = load_json(iqa_answer_path)
        questions = answer_payload.get("answers", [])
        question_items = [
            {"question": item.get("question", "")}
            for item in questions
            if isinstance(item, dict) and item.get("question")
        ]
        if not question_items and iqa_output_path is not None and iqa_output_path.exists():
            iqa_payload = load_json(iqa_output_path)
            question_items = iqa_payload.get("questions", [])

        answers = answer_payload.get("answers", []) if isinstance(answer_payload, dict) else []
        if isinstance(question_items, list) and question_items and isinstance(answers, list) and answers:
            gpt_value = {
                "answers": answers,
            }
            if isinstance(answer_payload.get("score"), int):
                gpt_value["score"] = answer_payload["score"]
            samples.append(
                {
                    "id": f"{pair_id}_iqa_evaluation",
                    "image": image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": format_evaluation_human(prompt, question_items),
                        },
                        {"from": "gpt", "value": json.dumps(gpt_value, ensure_ascii=False)},
                    ],
                }
            )

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dynevalinstruct_t2ia.json and dynevalinstruct_iqa.json"
    )
    parser.add_argument("--annotations-file", type=Path, required=True)
    parser.add_argument("--t2ia-questions-dir", type=Path, default=None)
    parser.add_argument("--t2ia-answers-dir", type=Path, default=None)
    parser.add_argument("--iqa-outputs-dir", type=Path, default=None)
    parser.add_argument("--iqa-answers-dir", type=Path, default=None)
    parser.add_argument(
        "--output-t2ia",
        type=Path,
        default=Path("dynevalinstruct_t2ia.json"),
    )
    parser.add_argument(
        "--output-iqa",
        type=Path,
        default=Path("dynevalinstruct_iqa.json"),
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    annotations_file = resolve_input_path(args.annotations_file, base_dir)
    records = load_prompt_records(annotations_file)

    stage1_samples: list[dict] = []
    stage2_samples: list[dict] = []
    for record in records:
        stage1_samples.extend(
            build_stage1_samples(
                record,
                annotations_file,
                args.t2ia_questions_dir,
                args.t2ia_answers_dir,
            )
        )
        stage2_samples.extend(
            build_stage2_samples(
                record,
                annotations_file,
                args.iqa_outputs_dir,
                args.iqa_answers_dir,
            )
        )

    args.output_t2ia.write_text(
        json.dumps(stage1_samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.output_iqa.write_text(
        json.dumps(stage2_samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(stage1_samples)} Stage 1 samples to {args.output_t2ia}")
    print(f"Wrote {len(stage2_samples)} Stage 2 samples to {args.output_iqa}")


if __name__ == "__main__":
    main()
