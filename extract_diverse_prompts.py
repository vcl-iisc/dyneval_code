from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
RESULT_ROOT = PIPELINE_DIR / "complexity-based-scoring-part2"
PREPROCESS_DIR = RESULT_ROOT / "preprocess"
SCORES_DIR = RESULT_ROOT / "scores"
TIERS_DIR = RESULT_ROOT / "tiers"
CATEGORIES_DIR = RESULT_ROOT / "categories"

# Keep the default input portable by resolving it relative to this script's
# repository. Users can still point elsewhere with the --input argument.
DEFAULT_INPUT = PIPELINE_DIR / "data" / "diffusiondb-prompts.txt"
DEFAULT_UNIQUE_OUT = PREPROCESS_DIR / "stage1-unique.txt"
DEFAULT_MINLEN_OUT = PREPROCESS_DIR / "stage2-min30chars.txt"
DEFAULT_KEPT_OUT = PREPROCESS_DIR / "stage3-deduped.txt"
DEFAULT_REMOVED_OUT = PREPROCESS_DIR / "stage3-overlap-removed.txt"

DEFAULT_JSONL = SCORES_DIR / "nine-factor-complexity.jsonl"
DEFAULT_TSV = SCORES_DIR / "nine-factor-complexity.tsv"
DEFAULT_SUMMARY = SCORES_DIR / "nine-factor-summary.json"
DEFAULT_ERROR_JSONL = SCORES_DIR / "nine-factor-errors.jsonl"
DEFAULT_SORTED_JSON = SCORES_DIR / "nine-factor-complexity-sorted-desc.json"

DEFAULT_MODEL = "Qwen/Qwen3.6-27B"
DEFAULT_CORPUS_SIZE = 500_000
DEFAULT_TIER1_MIN = 200.0  # score >= this → tier1
DEFAULT_TIER2_MIN = 100.0  # tier2_min <= score < tier1_min → tier2; else tier3


#####
# Load a Python file dynamically so this pipeline can reuse helper modules
# without requiring them to be installed as packages.
####
def load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


#####
# Load the module that implements preprocessing and complexity scoring.
####
def load_part1():
    return load_module_from_path(
        "complexity_based_scoring_part1",
        SCRIPT_DIR / "complexity-based-scoring-part1.py",
    )


#####
# Load the module that assigns categories to the tiered prompts.
####
def load_assign():
    return load_module_from_path(
        "assign_prompt_categories_qwen3_6_27b",
        SCRIPT_DIR / "assign_prompt_categories_qwen3_6_27b.py",
    )


#####
# Read scored prompts, preferring the already-sorted JSON file and falling
# back to the line-delimited JSON scoring output when needed.
####
def load_scored_records(
    jsonl_path: Path | None, sorted_json_path: Path | None
) -> list[dict[str, Any]]:
    """Load scored records from sorted JSON (preferred) or score JSONL."""
    if sorted_json_path is not None and sorted_json_path.exists():
        data = json.loads(sorted_json_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise SystemExit(f"Sorted JSON must be a list: {sorted_json_path}")
        return data

    if jsonl_path is None or not jsonl_path.exists():
        raise SystemExit(
            f"Need scored prompts. Missing sorted JSON and JSONL:\n"
            f"  sorted={sorted_json_path}\n  jsonl={jsonl_path}"
        )

    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON on line {line_number} of {jsonl_path}: {exc}"
                ) from exc
            if "score" not in record:
                raise SystemExit(f"Missing 'score' on line {line_number} of {jsonl_path}")
            records.append(record)
    return records


#####
# Sort all scored prompts from highest to lowest complexity and save them as
# one JSON array for later tier construction.
####
def stage4b_sort_scores(jsonl_path: Path, sorted_json_path: Path) -> int:
    """Sort scored prompts by final score descending → JSON array."""
    records = load_scored_records(jsonl_path, sorted_json_path=None)
    records.sort(key=lambda r: float(r["score"]), reverse=True)
    sorted_json_path.parent.mkdir(parents=True, exist_ok=True)
    with sorted_json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return len(records)


#####
# Convert a numeric complexity score into its configured tier name.
####
def tier_name_for_score(score: float, tier1_min: float, tier2_min: float) -> str:
    if score >= tier1_min:
        return "tier1"
    if score >= tier2_min:
        return "tier2"
    return "tier3"


#####
# Select the highest-scoring prompts up to the requested corpus size, split
# them into three tiers, and write prompt, record, and summary files.
####
def stage4c_build_tiered_corpus(
    records: list[dict[str, Any]],
    tiers_dir: Path,
    corpus_size: int,
    tier1_min: float,
    tier2_min: float,
) -> dict[str, Any]:
    """Build up to ``corpus_size`` prompts into tier1/2/3 txt (+ jsonl) files.

    Prompts are taken in descending score order so the highest-complexity prompts
    fill the corpus first. Each prompt is routed to a tier by score thresholds:
      tier1: score >= tier1_min
      tier2: tier2_min <= score < tier1_min
      tier3: score < tier2_min
    """
    tiers_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: float(r["score"]), reverse=True)

    buckets: dict[str, list[dict[str, Any]]] = {"tier1": [], "tier2": [], "tier3": []}
    for record in ordered:
        if sum(len(v) for v in buckets.values()) >= corpus_size:
            break
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            continue
        score = float(record["score"])
        name = tier_name_for_score(score, tier1_min, tier2_min)
        buckets[name].append(record)

    summary: dict[str, Any] = {
        "corpus_size_target": corpus_size,
        "corpus_size_actual": sum(len(v) for v in buckets.values()),
        "tier1_min_score": tier1_min,
        "tier2_min_score": tier2_min,
        "thresholds": {
            "tier1": f"score >= {tier1_min}",
            "tier2": f"{tier2_min} <= score < {tier1_min}",
            "tier3": f"score < {tier2_min}",
        },
        "tiers": {},
    }

    for name in ("tier1", "tier2", "tier3"):
        tier_records = buckets[name]
        txt_path = tiers_dir / f"{name}.txt"
        jsonl_path = tiers_dir / f"{name}.jsonl"
        meta_path = tiers_dir / f"{name}-meta.json"

        with txt_path.open("w", encoding="utf-8", newline="\n") as txt, jsonl_path.open(
            "w", encoding="utf-8"
        ) as jsonl:
            for record in tier_records:
                prompt = str(record.get("prompt", "")).replace("\n", " ").strip()
                txt.write(prompt + "\n")
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

        scores = [float(r["score"]) for r in tier_records]
        meta = {
            "tier": name,
            "num_prompts": len(tier_records),
            "txt": str(txt_path),
            "jsonl": str(jsonl_path),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "score_mean": (sum(scores) / len(scores)) if scores else None,
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        summary["tiers"][name] = meta
        print(
            f"    {name}: {len(tier_records):,} prompts -> {txt_path.name}"
            + (
                f" (score {meta['score_min']:.2f} .. {meta['score_max']:.2f})"
                if scores
                else " (empty)"
            )
        )

    summary_path = tiers_dir / "tiers-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"    total corpus: {summary['corpus_size_actual']:,} / {corpus_size:,}")
    print(f"    summary -> {summary_path}")
    return summary


#####
# Assign semantic categories to every prompt in one tier. Existing JSONL
# output can be resumed, and a loaded model is reused across tiers.
####
def stage5_assign_categories(
    assign,
    input_path: Path,
    jsonl_output: Path,
    json_output: Path,
    summary_output: Path,
    error_output: Path,
    model: str,
    max_new_tokens: int,
    dtype: str,
    device_map: str,
    category_limit: int | None,
    no_resume: bool,
    strict_json: bool,
    processor=None,
    model_obj=None,
    scorer=None,
) -> tuple[int, int, Any, Any, Any]:
    """Stage 5 for one tier. Reuses a preloaded model when provided.

    Returns (processed_this_run, completed_total, scorer, processor, model_obj).
    """
    for path in (jsonl_output, json_output, summary_output, error_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    if no_resume:
        jsonl_output.unlink(missing_ok=True)
        error_output.unlink(missing_ok=True)

    records = assign.load_input_prompts(input_path)
    total_input = len(records)
    already_done = assign.count_jsonl_records(jsonl_output)
    todo = records[already_done:]
    if category_limit is not None:
        todo = todo[:category_limit]

    print(f"    input prompts : {total_input:,} ({input_path})")
    print(f"    already done  : {already_done:,}")
    print(f"    this run      : {len(todo):,}")

    if not todo:
        print("    nothing to do for this tier.")
        if jsonl_output.exists():
            assign.write_full_json(jsonl_output, json_output)
            summary_args = SimpleNamespace(
                input=input_path,
                jsonl_output=jsonl_output,
                json_output=json_output,
                error_output=error_output,
                model=model,
            )
            assign.write_summary(
                summary_output, summary_args, total_input, already_done, jsonl_output
            )
        return 0, already_done, scorer, processor, model_obj

    if processor is None or model_obj is None or scorer is None:
        print("    Loading model for Stage 5...")
        scorer = assign.load_scorer_helpers()
        processor, model_obj = scorer.load_model(model, dtype, device_map)

    from tqdm import tqdm

    processed = 0
    try:
        with tqdm(total=len(todo), desc=f"Categories [{input_path.stem}]", unit="prompt") as progress:
            with jsonl_output.open("a", encoding="utf-8") as jsonl, error_output.open(
                "a", encoding="utf-8"
            ) as err_handle:
                for record in todo:
                    out, error = assign.assign_categories_for_prompt(
                        scorer,
                        processor,
                        model_obj,
                        record,
                        max_new_tokens,
                        strict_json,
                    )
                    jsonl.write(json.dumps(out, ensure_ascii=False) + "\n")
                    jsonl.flush()
                    if error is not None:
                        err_handle.write(json.dumps(error, ensure_ascii=False) + "\n")
                        err_handle.flush()
                    processed += 1
                    progress.update(1)
    except KeyboardInterrupt:
        print("\nInterrupted during Stage 5. Category JSONL is resumable.", file=sys.stderr)
        raise
    finally:
        completed = already_done + processed
        if jsonl_output.exists():
            n = assign.write_full_json(jsonl_output, json_output)
            summary_args = SimpleNamespace(
                input=input_path,
                jsonl_output=jsonl_output,
                json_output=json_output,
                error_output=error_output,
                model=model,
            )
            assign.write_summary(
                summary_output, summary_args, total_input, completed, jsonl_output
            )
            print(f"    Wrote full category JSON ({n:,} records): {json_output}")

    return processed, already_done + processed, scorer, processor, model_obj


#####
# Define command-line options for file locations, model settings, thresholds,
# processing limits, resume behavior, and optional stage skipping.
####
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Preprocess I/O (stages 1-3) ----
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--unique-output", type=Path, default=DEFAULT_UNIQUE_OUT)
    parser.add_argument("--minlen-output", type=Path, default=DEFAULT_MINLEN_OUT)
    parser.add_argument("--kept-output", type=Path, default=DEFAULT_KEPT_OUT)
    parser.add_argument("--removed-output", type=Path, default=DEFAULT_REMOVED_OUT)
    parser.add_argument("--min-chars", type=int, default=30)
    parser.add_argument("--case-sensitive", action="store_true")

    # ---- Stage 3 knobs ----
    parser.add_argument("--ngram", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--bands", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-bucket-pairs", type=int, default=64)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--preprocess-limit", type=int, default=None)

    # ---- Stage 4 (complexity scoring) ----
    parser.add_argument("--jsonl-output", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--tsv-output", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--error-output", type=Path, default=DEFAULT_ERROR_JSONL)
    parser.add_argument("--sorted-json-output", type=Path, default=DEFAULT_SORTED_JSON)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens-elements", type=int, default=400)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--score-limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--strict-json", action="store_true")

    # ---- Stage 4c (tiered corpus) ----
    parser.add_argument("--tiers-dir", type=Path, default=TIERS_DIR)
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=DEFAULT_CORPUS_SIZE,
        help="Max prompts in the combined tier1+tier2+tier3 corpus (default 500000).",
    )
    parser.add_argument(
        "--tier1-min-score",
        type=float,
        default=DEFAULT_TIER1_MIN,
        help="Tier1 if score >= this (default 200).",
    )
    parser.add_argument(
        "--tier2-min-score",
        type=float,
        default=DEFAULT_TIER2_MIN,
        help="Tier2 if this <= score < tier1-min; else tier3 (default 100).",
    )

    # ---- Stage 5 (category assignment, per tier) ----
    parser.add_argument("--categories-dir", type=Path, default=CATEGORIES_DIR)
    parser.add_argument("--category-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--category-limit",
        type=int,
        default=None,
        help="Max prompts to categorize PER TIER this run (default: all remaining).",
    )
    parser.add_argument("--no-resume-categories", action="store_true")

    # ---- Flow control ----
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--skip-sort", action="store_true")
    parser.add_argument("--skip-tiers", action="store_true")
    parser.add_argument("--skip-categories", action="store_true")
    return parser.parse_args()


#####
# Run the complete prompt-processing pipeline in order: preprocess, score,
# sort, create tiers, and assign categories.
####
def main() -> None:
    args = parse_args()
    part1 = load_part1()
    assign = load_assign()

    PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    args.tiers_dir.mkdir(parents=True, exist_ok=True)
    args.categories_dir.mkdir(parents=True, exist_ok=True)

    print(f"Result root: {RESULT_ROOT}")
    print(f"  preprocess  -> {PREPROCESS_DIR}")
    print(f"  scores      -> {SCORES_DIR}")
    print(f"  tiers       -> {args.tiers_dir}")
    print(f"  categories  -> {args.categories_dir}")

    total1 = kept1 = total2 = kept2 = total3 = kept3 = removed3 = 0

    #####
    # Steps 1-3: remove exact duplicates, short prompts, and near-duplicates
    # to create a clean prompt collection for scoring.
    ####
    if not args.skip_preprocess:
        overlap = part1.load_overlap_module()

        print("=" * 70)
        print("Stage 1/5: removing EXACT duplicate prompts")
        print(f"    input : {args.input}")
        total1, kept1 = part1.stage1_remove_exact_duplicates(
            args.input, args.unique_output, case_insensitive=not args.case_sensitive
        )
        print(f"    output: {args.unique_output}")
        print(f"    read {total1:,} -> kept {kept1:,} (removed {total1 - kept1:,} exact duplicates)")

        print("=" * 70)
        print(f"Stage 2/5: removing prompts shorter than {args.min_chars} characters")
        total2, kept2 = part1.stage2_filter_min_length(
            args.unique_output, args.minlen_output, args.min_chars
        )
        print(f"    output: {args.minlen_output}")
        print(f"    read {total2:,} -> kept {kept2:,} (removed {total2 - kept2:,} too-short prompts)")

        print("=" * 70)
        print("Stage 3/5: removing NEAR-duplicate (overlapping) prompts")
        print(
            f"    params: ngram={args.ngram} threshold={args.threshold} "
            f"num_perm={args.num_perm} bands={args.bands} threads={args.threads}"
        )
        total3, kept3, removed3 = part1.stage3_remove_overlaps(
            overlap,
            args.minlen_output,
            args.kept_output,
            args.removed_output,
            ngram=args.ngram,
            threshold=args.threshold,
            num_perm=args.num_perm,
            bands=args.bands,
            seed=args.seed,
            max_bucket_pairs=args.max_bucket_pairs,
            limit=args.preprocess_limit,
            threads=args.threads,
        )
        print(f"    kept    -> {args.kept_output} ({kept3:,} prompts)")
        print(f"    removed -> {args.removed_output} ({removed3:,} near-duplicates)")
    else:
        print("=" * 70)
        print("Stages 1-3 SKIPPED (--skip-preprocess)")
        if not args.kept_output.exists() and not args.skip_score:
            raise SystemExit(
                f"--skip-preprocess set but cleaned file not found: {args.kept_output}"
            )

    #####
    # Step 4: score each cleaned prompt across nine complexity factors.
    ####
    score_processed = score_completed = 0
    if not args.skip_score:
        print("=" * 70)
        print("Stage 4/5: scoring with Qwen3.6-27B (nine complexity factors)")
        print(f"    input : {args.kept_output}")
        scorer = part1.load_scorer_module()
        score_processed, score_completed = part1.stage4_score_with_qwen27b(
            scorer,
            input_path=args.kept_output,
            jsonl_output=args.jsonl_output,
            tsv_output=args.tsv_output,
            summary_output=args.summary_output,
            error_output=args.error_output,
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            max_new_tokens_elements=args.max_new_tokens_elements,
            dtype=args.dtype,
            device_map=args.device_map,
            score_limit=args.score_limit,
            no_resume=args.no_resume,
            strict_json=args.strict_json,
        )
        print(f"    scored this run : {score_processed:,}")
        print(f"    scored total    : {score_completed:,}")
    else:
        print("=" * 70)
        print("Stage 4 SKIPPED (--skip-score)")
        print(f"    using existing scores: {args.jsonl_output}")

    #####
    # Step 4b: sort scored prompts from most to least complex.
    ####
    if not args.skip_sort:
        print("=" * 70)
        print("Stage 4b/5: sorting prompts by complexity score (descending)")
        if not args.jsonl_output.exists():
            raise SystemExit(f"Score JSONL not found for sorting: {args.jsonl_output}")
        n_sorted = stage4b_sort_scores(args.jsonl_output, args.sorted_json_output)
        print(f"    sorted {n_sorted:,} prompts -> {args.sorted_json_output}")
    else:
        print("=" * 70)
        print("Stage 4b SKIPPED (--skip-sort)")

    #####
    # Step 4c: build the requested-size corpus and split it into three
    # complexity tiers using the configured score thresholds.
    ####
    if not args.skip_tiers:
        print("=" * 70)
        print("Stage 4c/5: building tiered corpus (tier1 / tier2 / tier3)")
        print(
            f"    thresholds: tier1 >= {args.tier1_min_score}, "
            f"tier2 [{args.tier2_min_score}, {args.tier1_min_score}), "
            f"tier3 < {args.tier2_min_score}"
        )
        print(f"    corpus size target: {args.corpus_size:,}")
        records = load_scored_records(args.jsonl_output, args.sorted_json_output)
        stage4c_build_tiered_corpus(
            records,
            tiers_dir=args.tiers_dir,
            corpus_size=args.corpus_size,
            tier1_min=args.tier1_min_score,
            tier2_min=args.tier2_min_score,
        )
    else:
        print("=" * 70)
        print("Stage 4c SKIPPED (--skip-tiers)")
        print(f"    using existing tiers in: {args.tiers_dir}")

    #####
    # Step 5: assign the supported semantic categories within each tier,
    # reusing one loaded model to avoid unnecessary reloads.
    ####
    if args.skip_categories:
        print("=" * 70)
        print("Stage 5 SKIPPED (--skip-categories)")
        print("DONE.")
        print(f"Result root: {RESULT_ROOT}")
        return

    print("=" * 70)
    print("Stage 5/5: assigning 42 categories per tier (one category at a time)")

    scorer = processor = model_obj = None
    total_cat_processed = 0
    total_cat_completed = 0

    for tier_name in ("tier1", "tier2", "tier3"):
        tier_jsonl = args.tiers_dir / f"{tier_name}.jsonl"
        tier_txt = args.tiers_dir / f"{tier_name}.txt"
        # Prefer jsonl (keeps scores); fall back to txt-wrapped prompts.
        if tier_jsonl.exists() and tier_jsonl.stat().st_size > 0:
            input_path = tier_jsonl
        elif tier_txt.exists() and tier_txt.stat().st_size > 0:
            input_path = tier_txt
        else:
            print(f"\n  [{tier_name}] empty/missing — skipping category assignment")
            continue

        tier_cat_dir = args.categories_dir / tier_name
        print(f"\n  [{tier_name}] classifying -> {tier_cat_dir}")
        processed, completed, scorer, processor, model_obj = stage5_assign_categories(
            assign,
            input_path=input_path,
            jsonl_output=tier_cat_dir / "prompt-category-assignments.jsonl",
            json_output=tier_cat_dir / "prompt-category-assignments.json",
            summary_output=tier_cat_dir / "prompt-category-assignments-summary.json",
            error_output=tier_cat_dir / "prompt-category-assignments-errors.jsonl",
            model=args.model,
            max_new_tokens=args.category_max_new_tokens,
            dtype=args.dtype,
            device_map=args.device_map,
            category_limit=args.category_limit,
            no_resume=args.no_resume_categories,
            strict_json=args.strict_json,
            processor=processor,
            model_obj=model_obj,
            scorer=scorer,
        )
        total_cat_processed += processed
        total_cat_completed += completed

    print("=" * 70)
    print("DONE.")
    print(f"Result root: {RESULT_ROOT}")
    if not args.skip_preprocess:
        print(
            f"    preprocess: {total1:,} raw -> {kept1:,} unique -> "
            f"{kept2:,} >= {args.min_chars} chars -> {kept3:,} after overlap removal"
        )
    if not args.skip_score:
        print(f"    scored this run : {score_processed:,}")
        print(f"    scored total    : {score_completed:,}")
    print(f"    categorized this run : {total_cat_processed:,}")
    print(f"    categorized total    : {total_cat_completed:,}")
    print(f"    tiers dir      : {args.tiers_dir}")
    print(f"    categories dir : {args.categories_dir}")


#####
# Start the pipeline only when this file is executed directly.
####
if __name__ == "__main__":
    main()
