import argparse
import csv
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def record_id(path: Path, payload: dict) -> str:
    for key in ("pair_id", "item_key", "image_id"):
        value = str(payload.get(key, "")).strip()
        if value:
            return value
    return path.stem


def normalize_score(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def average_answer_scores(payload: dict) -> float | None:
    answers = payload.get("answers", [])
    if not isinstance(answers, list):
        return None

    scores: list[float] = []
    for item in answers:
        if not isinstance(item, dict):
            continue
        score = normalize_score(item.get("score"))
        if score is not None:
            scores.append(score)
            continue
        correct = item.get("correct")
        if isinstance(correct, bool):
            scores.append(5.0 if correct else 1.0)

    if not scores:
        return None
    return sum(scores) / len(scores)


def extract_score(payload: dict) -> float | None:
    score = normalize_score(payload.get("score"))
    if score is not None:
        return score
    return average_answer_scores(payload)


def load_scores(directory: Path) -> dict[str, dict]:
    scores: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = load_json(path)
        except Exception as exc:
            scores[path.stem] = {
                "score": None,
                "path": str(path),
                "error": str(exc),
            }
            continue

        if not isinstance(payload, dict):
            scores[path.stem] = {
                "score": None,
                "path": str(path),
                "error": "JSON payload is not an object",
            }
            continue

        rid = record_id(path, payload)
        scores[rid] = {
            "score": extract_score(payload),
            "path": str(path),
            "error": "",
        }
    return scores


def write_json(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id",
        "iqa_score",
        "t2ia_score",
        "alpha",
        "beta",
        "overall_score",
        "iqa_file",
        "t2ia_file",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iqa-dir", type=Path, default=Path("iqa_answers"))
    parser.add_argument("--t2ia-dir", type=Path, default=Path("model-responses/answers"))
    parser.add_argument("--output-file", type=Path, default=Path("overall_scores.json"))
    parser.add_argument("--format", choices=("json", "csv"), default="json")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--include-missing", action="store_true")
    args = parser.parse_args()

    iqa_scores = load_scores(args.iqa_dir)
    t2ia_scores = load_scores(args.t2ia_dir)

    if args.include_missing:
        pair_ids = sorted(set(iqa_scores) | set(t2ia_scores))
    else:
        pair_ids = sorted(set(iqa_scores) & set(t2ia_scores))

    rows: list[dict] = []
    for pair_id in pair_ids:
        iqa = iqa_scores.get(pair_id, {})
        t2ia = t2ia_scores.get(pair_id, {})
        iqa_score = iqa.get("score")
        t2ia_score = t2ia.get("score")

        if iqa_score is None or t2ia_score is None:
            overall_score = None
            status = "missing_score"
        else:
            overall_score = args.alpha * iqa_score + args.beta * t2ia_score
            status = "ok"

        rows.append(
            {
                "pair_id": pair_id,
                "iqa_score": iqa_score,
                "t2ia_score": t2ia_score,
                "alpha": args.alpha,
                "beta": args.beta,
                "overall_score": overall_score,
                "iqa_file": iqa.get("path", ""),
                "t2ia_file": t2ia.get("path", ""),
                "status": status,
            }
        )

    if args.format == "csv":
        write_csv(rows, args.output_file)
    else:
        write_json(rows, args.output_file)

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"wrote {len(rows)} rows to {args.output_file} ({ok_count} scored)")


if __name__ == "__main__":
    main()
