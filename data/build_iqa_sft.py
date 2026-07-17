
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


# These prompt strings are copied verbatim from inference/run-inference.py so the
# assistant targets built here match exactly what the model is asked for at
# inference time. Do not edit one without updating the other.

IQA_TOKEN = "<IQA>"

IQA_SCENE_GRAPH_PROMPT = """Here is an image generated for this prompt "{prompt}".
Now generate a scene graph by considering only the image. Consider the text prompt for your reference.
Return only valid JSON with exactly these keys: "nodes" and "edges".
"nodes" must be an array of visible objects, where each object has "id", "label", and "attributes".
"edges" must be an array of relationships, where each relationship has "source", "relation", and "target".
Do not write markdown, bullets, or prose."""

IQA_DECOMPOSITION_PROMPT = """We are doing an image quality assessment. The generated image is provided to you along with a list of object nodes JSON.
We have ALREADY done the text-to-image alignment. Therefore you must NOT ask presence, attribute, color, count, or alignment questions (for example: "Is there a bench?", "Is the bench wooden?", "Are the flowers red?", "Is the wall white?"). Such questions are forbidden.
Instead, for each object node, decompose it into its parts and generate yes/no questions that judge ONLY the rendering QUALITY of how that object is drawn in the image. Focus on perceptual defects such as: shape correctness and geometry, structural distortions or deformities, unnatural or broken proportions, blur or smearing, texture artifacts, melting/warping, incorrect or impossible 3D spatial structure, and boundary/edge coherence.
Each question must be phrased so that a well-rendered, high-quality object gets the target answer "yes".
Examples of the required style: "Are the bench slats straight and free of distortion?", "Are the flower petals rendered with clean edges and no smearing?", "Does the wall surface have a consistent texture without artifacts?".
Inspect the provided image to decide the target answer.
Return only valid JSON with exactly one key: "questions".
Each question object must contain "node_id", "question", and "target_answer".
Do not write markdown, bullets, or prose."""

IQA_FINAL_PROMPT = """Now answer these questions for this image and generate a score given the target answers on a scale of 1-5.
Use the provided question JSON directly.
Do not write paragraphs, markdown, summaries, or bullet-point analysis.
Return only valid JSON in this exact format:
{
  "questions": [
    {"question": "...", "target_answer": "yes/no", "answer": "yes/no", "score": 1, "reasoning": "..."}
  ]
}
Each "score" must be a number from 1 to 5 based on how well the image answer matches the target answer."""


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image_path(image_ref: str, images_root: Path | None) -> Path | None:
    ref = Path(image_ref)
    candidates = [ref] if ref.is_absolute() else []
    if images_root is not None:
        candidates.extend([images_root / ref, images_root / ref.name])
    else:
        candidates.append(ref)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def normalize_yes_no(value: Any) -> str:
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return ""


def normalize_scene_graph(raw: Any) -> dict[str, list[dict[str, Any]]]:
    # Convert a teacher scene graph into the canonical inference format:
    # nodes as {"id","label","attributes"} and edges as {"source","relation","target"}.
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
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


def normalize_generated_questions(raw: Any) -> list[dict[str, str]]:
    # Extract teacher scene-graph questions as [{node_id, question, target_answer}].
    if isinstance(raw, dict):
        raw = raw.get("questions", raw.get("items", []))
    out: list[dict[str, str]] = []
    seen = set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        target_answer = normalize_yes_no(item.get("target_answer", item.get("answer", "")))
        node_id = str(item.get("node_id", item.get("node", ""))).strip()
        if not question or not target_answer:
            continue
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"node_id": node_id, "question": question, "target_answer": target_answer})
    return out


def score_from_correct(correct: Any, overall_score: Any, mode: str) -> int:
    # Map a teacher per-question judgement to a 1-5 score.
    if mode == "overall":
        try:
            score = int(round(float(overall_score)))
        except (TypeError, ValueError):
            score = 0
        if 1 <= score <= 5:
            return score
    if isinstance(correct, bool):
        return 5 if correct else 1
    return 3


def scene_graph_messages(prompt: str, scene_graph: dict[str, Any]) -> list[dict[str, Any]]:
    user_text = f"{IQA_TOKEN}\n{IQA_SCENE_GRAPH_PROMPT.format(prompt=prompt)}"
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": json_dumps(scene_graph)}]},
    ]


def question_messages(prompt: str, scene_graph: dict[str, Any], questions: list[dict[str, str]]) -> list[dict[str, Any]]:
    # Match inference: pass only node ids and labels (no attributes/edges) so the
    # model learns to generate quality questions rather than re-verify attributes.
    raw_nodes = scene_graph.get("nodes", []) if isinstance(scene_graph, dict) else []
    nodes = [
        {"node_id": str(node.get("id", "")), "label": str(node.get("label", ""))}
        for node in raw_nodes
        if isinstance(node, dict) and str(node.get("label", "")).strip()
    ]
    user_text = (
        f"{IQA_TOKEN}\nPrompt: {prompt}\n\n"
        f"Object nodes JSON:\n{json_dumps(nodes)}\n\n"
        f"{IQA_DECOMPOSITION_PROMPT}"
    )
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": json_dumps({"questions": questions})}]},
    ]


def final_messages(
    prompt: str,
    scene_graph: dict[str, Any],
    questions: list[dict[str, str]],
    scored_questions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    user_text = (
        f"{IQA_TOKEN}\nPrompt: {prompt}\n\n"
        f"Scene graph JSON:\n{json_dumps(scene_graph)}\n\n"
        f"IQA question JSON:\n{json_dumps({'questions': questions})}\n\n"
        f"{IQA_FINAL_PROMPT}"
    )
    return [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": json_dumps({"questions": scored_questions})}]},
    ]


def load_teacher_answers(answers_dir: Path | None, pair_id: str) -> tuple[list[dict[str, Any]], Any]:
    # Load the teacher answer payload for one pair_id, if available.
    if answers_dir is None:
        return [], None
    path = answers_dir / f"{pair_id}.json"
    if not path.exists():
        return [], None
    data = read_json(path)
    if not isinstance(data, dict):
        return [], None
    answers = data.get("answers", [])
    return (answers if isinstance(answers, list) else []), data.get("score")


def build_iqa_rows(
    scene_graph_dir: Path,
    answers_dir: Path | None,
    images_root: Path | None,
    score_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    # Build scene-graph, question-generation, and scoring rows from teacher outputs.
    rows: list[dict[str, Any]] = []
    counts = {"scene_graph": 0, "question_generation": 0, "evaluation": 0, "skipped": 0}

    for path in sorted(scene_graph_dir.glob("*.json")):
        data = read_json(path)
        if not isinstance(data, dict):
            counts["skipped"] += 1
            continue

        pair_id = str(data.get("pair_id") or data.get("text_id") or path.stem).strip()
        prompt = str(data.get("prompt") or data.get("text") or "").strip()
        image_ref = str(data.get("image_path") or data.get("image_ref") or "").strip()
        scene_graph = normalize_scene_graph(data.get("scene_graph"))

        if not pair_id or not prompt or not image_ref or not scene_graph["nodes"]:
            counts["skipped"] += 1
            continue

        image_path = resolve_image_path(image_ref, images_root)
        if image_path is None:
            counts["skipped"] += 1
            continue

        base = {"prompt_id": pair_id, "prompt": prompt, "image_ref": image_ref, "image_path": str(image_path)}

        rows.append(
            {
                **base,
                "id": f"{pair_id}_iqa_scene_graph",
                "scene_graph": scene_graph,
                "task": "IQA_scene_graph",
                "messages": scene_graph_messages(prompt, scene_graph),
            }
        )
        counts["scene_graph"] += 1

        questions = normalize_generated_questions(data.get("questions"))
        if not questions:
            continue

        rows.append(
            {
                **base,
                "id": f"{pair_id}_iqa_questions",
                "scene_graph": scene_graph,
                "questions": questions,
                "task": "IQA_question_generation",
                "messages": question_messages(prompt, scene_graph, questions),
            }
        )
        counts["question_generation"] += 1

        teacher_answers, overall_score = load_teacher_answers(answers_dir, pair_id)
        if not teacher_answers:
            continue

        answer_by_question = {}
        for item in teacher_answers:
            if isinstance(item, dict):
                key = str(item.get("question", "")).strip().lower()
                if key:
                    answer_by_question[key] = item

        scored_questions: list[dict[str, Any]] = []
        for question in questions:
            teacher = answer_by_question.get(question["question"].strip().lower())
            if not isinstance(teacher, dict):
                continue
            model_answer = normalize_yes_no(teacher.get("answer", ""))
            target_answer = normalize_yes_no(teacher.get("target_answer", question["target_answer"])) or question["target_answer"]
            if not model_answer:
                continue
            correct = teacher.get("correct")
            if not isinstance(correct, bool):
                correct = model_answer == target_answer
            scored_questions.append(
                {
                    "question": question["question"],
                    "target_answer": target_answer,
                    "answer": model_answer,
                    "score": score_from_correct(correct, overall_score, score_mode),
                    "reasoning": str(teacher.get("reasoning", "")).strip(),
                }
            )

        if not scored_questions:
            continue

        rows.append(
            {
                **base,
                "id": f"{pair_id}_iqa_evaluation",
                "scene_graph": scene_graph,
                "questions": questions,
                "scored_questions": scored_questions,
                "task": "IQA_evaluation",
                "messages": final_messages(prompt, scene_graph, questions, scored_questions),
            }
        )
        counts["evaluation"] += 1

    return rows, counts


def split_rows(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rng.shuffle(rows)
    val_count = max(1, int(len(rows) * val_ratio)) if len(rows) > 1 else 0
    return rows[val_count:], rows[:val_count]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DynEval <IQA> SFT JSONL from teacher scene-graph/answer folders.")
    parser.add_argument("--scene-graph-dir", type=Path, required=True, help="Directory of teacher IQA scene-graph/question JSON files (iqa_outputs).")
    parser.add_argument("--answers-dir", type=Path, default=None, help="Directory of teacher IQA answer JSON files (iqa_answers). Optional; enables scoring rows.")
    parser.add_argument("--images-root", type=Path, default=None, help="Root folder used to resolve image paths.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-mode",
        choices=["correct", "overall"],
        default="correct",
        help="Per-question 1-5 score source: 'correct' maps correct/incorrect to 5/1; 'overall' uses the teacher's overall score.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows, counts = build_iqa_rows(args.scene_graph_dir, args.answers_dir, args.images_root, args.score_mode)
    if not rows:
        raise SystemExit("No IQA rows were built. Check the teacher output folders and image paths.")

    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)
    write_jsonl(train_rows, args.output_dir / "train.jsonl")
    write_jsonl(val_rows, args.output_dir / "val.jsonl")

    manifest = {
        "task": "IQA",
        "scene_graph_dir": str(args.scene_graph_dir),
        "answers_dir": str(args.answers_dir) if args.answers_dir is not None else None,
        "images_root": str(args.images_root) if args.images_root is not None else None,
        "score_mode": args.score_mode,
        "output_dir": str(args.output_dir),
        "rows": {
            "iqa_scene_graph": counts["scene_graph"],
            "iqa_question_generation": counts["question_generation"],
            "iqa_evaluation": counts["evaluation"],
            "skipped": counts["skipped"],
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
