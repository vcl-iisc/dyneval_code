from __future__ import annotations
import argparse
import ast
import importlib.util
import re
import sys
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = SCRIPT_DIR / "diverse-prompt-results"
PREPROCESS_DIR = RESULT_ROOT / "preprocess"
SCORES_DIR = RESULT_ROOT / "scores"
TIERS_DIR = RESULT_ROOT / "tiers"
CATEGORIES_DIR = RESULT_ROOT / "categories"

DEFAULT_INPUT = SCRIPT_DIR / "data" / "diffusiondb-prompts.txt"
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


# ---------------------------------------------------------------------------
# Inlined preprocessing and nine-factor scoring orchestration
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dynamic loaders (hyphenated / sibling scripts cannot use a normal import)
# ---------------------------------------------------------------------------
####
# Load a sibling Python file dynamically and return the imported module.
####
def load_module_from_path(module_name: str, path: Path):
    # Register in sys.modules before exec so @dataclass / typing work correctly
    # when the sibling script is loaded by file path.
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


####
# Load the MinHash/Jaccard helper used to detect near-duplicate prompts.
####
def load_overlap_module():
    return load_module_from_path(
        "jaccard_overlap", SCRIPT_DIR / "jacordian-overlap-removel.py"
    )


####
# Load the Qwen helper used for nine-factor prompt complexity scoring.
####
def load_scorer_module():
    return load_module_from_path(
        "score_qwen3_6_27b",
        SCRIPT_DIR / "score_prompts_qwen3_6_27b_one_dimension.py",
    )

# ---------------------------------------------------------------------------
# Stage 1: remove exact duplicate prompts
# ---------------------------------------------------------------------------
####
# Remove repeated prompts while preserving their first-seen order.
####
def stage1_remove_exact_duplicates(
    input_path: Path, output_path: Path, case_insensitive: bool
) -> tuple[int, int]:
    """Write every prompt exactly once, preserving first-seen order.

    Returns (total_read, kept).
    """
    seen: set[str] = set()
    total = 0
    kept = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", errors="replace") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            total += 1
            prompt = line.rstrip("\r\n")
            if not prompt:
                continue
            key = prompt.casefold() if case_insensitive else prompt
            if key in seen:
                continue
            seen.add(key)
            dst.write(prompt + "\n")
            kept += 1
    return total, kept


# ---------------------------------------------------------------------------
# Stage 2: drop prompts that are too short
# ---------------------------------------------------------------------------
####
# Keep only prompts that meet the configured minimum character count.
####
def stage2_filter_min_length(
    input_path: Path, output_path: Path, min_chars: int
) -> tuple[int, int]:
    """Keep only prompts with at least `min_chars` characters.

    Returns (total_read, kept).
    """
    total = 0
    kept = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", errors="replace") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            total += 1
            prompt = line.rstrip("\r\n")
            if len(prompt) >= min_chars:
                dst.write(prompt + "\n")
                kept += 1
    return total, kept


# ---------------------------------------------------------------------------
# Stage 3: remove near-duplicate (overlapping) prompts
# ---------------------------------------------------------------------------
####
# Use MinHash signatures and Jaccard similarity to remove near-duplicates.
####
def stage3_remove_overlaps(
    overlap,
    input_path: Path,
    kept_path: Path,
    removed_path: Path,
    ngram: int,
    threshold: float,
    num_perm: int,
    bands: int,
    seed: int,
    max_bucket_pairs: int,
    limit: int | None,
    threads: int = 32,
) -> tuple[int, int, int]:
    """Collapse clusters of similar prompts down to a single representative.

    Returns (total, kept, removed).
    """
    if num_perm % bands != 0:
        raise SystemExit(
            f"--num-perm ({num_perm}) must be divisible by --bands ({bands})."
        )
    rows = num_perm // bands

    prompts = overlap.load_prompts(input_path, limit)
    print(f"    loaded {len(prompts):,} prompts")
    print(f"    MinHash threads: {threads}")
    a, b = overlap.build_permutations(num_perm, seed)
    signatures = overlap.compute_signatures(prompts, ngram, a, b, threads=threads)
    keep_mask, _removed_pairs = overlap.dedupe(
        prompts, signatures, bands, rows, threshold, max_bucket_pairs
    )

    kept_path.parent.mkdir(parents=True, exist_ok=True)
    removed_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    removed = 0
    with kept_path.open("w", encoding="utf-8", newline="\n") as kf, removed_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as rf:
        for idx, prompt in enumerate(prompts):
            if keep_mask[idx]:
                kf.write(prompt + "\n")
                kept += 1
            else:
                rf.write(prompt + "\n")
                removed += 1
    return len(prompts), kept, removed


# ---------------------------------------------------------------------------
# Stage 4: score cleaned prompts with Qwen3.6-27B
# ---------------------------------------------------------------------------
####
# Score cleaned prompts, write resumable results, and record a run summary.
####
def stage4_score_with_qwen27b(
    scorer,
    input_path: Path,
    jsonl_output: Path,
    tsv_output: Path,
    summary_output: Path,
    error_output: Path,
    model: str,
    max_new_tokens: int,
    max_new_tokens_elements: int,
    dtype: str,
    device_map: str,
    score_limit: int | None,
    no_resume: bool,
    strict_json: bool,
) -> tuple[int, int]:
    """Score prompts with the nine-factor Qwen3.6-27B scorer.

    Returns (processed_this_run, completed_total).
    """
    for path in (jsonl_output, tsv_output, summary_output, error_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    if no_resume:
        jsonl_output.unlink(missing_ok=True)
        tsv_output.unlink(missing_ok=True)
        error_output.unlink(missing_ok=True)

    total_input = scorer.count_lines(input_path)
    already_done = scorer.count_jsonl_records(jsonl_output)
    write_tsv_header = not tsv_output.exists() or tsv_output.stat().st_size == 0

    print(f"    input prompts : {total_input:,}")
    print(f"    already done  : {already_done:,}")
    print(
        f"    score limit   : {score_limit:,}"
        if score_limit is not None
        else "    score limit   : none (all remaining)"
    )
    print(
        f"    Qwen calls/prompt: {len(scorer.LLM_DIMENSIONS) + 1} "
        f"({len(scorer.LLM_DIMENSIONS)} per-dim "
        f"[{', '.join(scorer.LLM_DIMENSIONS)}] + 1 element-extraction). "
        f"Python factors: f1, f3."
    )
    print(f"    model: {model}")
    print("    Loading model...")
    processor, model_obj = scorer.load_model(model, dtype, device_map)

    processed = 0
    progress_total = (
        score_limit if score_limit is not None else max(0, total_input - already_done)
    )

    try:
        from tqdm import tqdm

        with tqdm(total=progress_total, desc="Stage 4 scoring", unit="prompt") as progress:
            for line_number, prompt in scorer.iter_prompts(input_path, already_done):
                record, errors = scorer.score_prompt_one_dimension_at_a_time(
                    processor,
                    model_obj,
                    line_number,
                    prompt,
                    max_new_tokens,
                    strict_json,
                    max_new_tokens_elements=max_new_tokens_elements,
                )
                scorer.append_record(jsonl_output, tsv_output, record, write_tsv_header)
                scorer.append_errors(error_output, errors)
                write_tsv_header = False

                processed += 1
                progress.update(1)
                if score_limit is not None and processed >= score_limit:
                    break
    except KeyboardInterrupt:
        print("\nInterrupted during Stage 4. Score outputs are resumable.", file=sys.stderr)
        raise

    completed = already_done + processed
    # write_summary expects an argparse-like namespace with these fields
    summary_args = SimpleNamespace(
        input=input_path,
        jsonl_output=jsonl_output,
        tsv_output=tsv_output,
        error_output=error_output,
        model=model,
    )
    scorer.write_summary(summary_output, summary_args, total_input, completed)
    return processed, completed

# ---------------------------------------------------------------------------
# Inlined prompt category assignment
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 42 categories: definition + in-context positive examples for Qwen.
# ---------------------------------------------------------------------------
CATEGORY_SPECS: dict[str, dict[str, Any]] = {
    "single_object": {
        "definition": "Prompt focuses on exactly ONE main object/entity (clothing/parts of that same entity do not count as extra objects).",
        "examples": [
            "a red apple on a white background",
            "portrait of a woman, highly detailed",
            "a single vintage camera",
        ],
        "not_examples": [
            "a cat and a dog",  # multi_object
            "a girl sitting at a table building a radio",  # multi_object (girl+table+radio)
        ],
    },
    "multi_object": {
        "definition": "Prompt mentions TWO OR MORE distinct objects/entities (not just parts/attributes of one entity).",
        "examples": [
            "a cat and a dog on a sofa",
            "a girl sitting at a table building a radio",
            "car parked beside a tall building",
        ],
        "not_examples": [
            "a red sports car",  # single_object
        ],
    },
    "human_present": {
        "definition": "A human / person / people / clearly human body part is mentioned.",
        "examples": [
            "a young woman with short hair",
            "two children playing in a park",
            "close-up of a man's hands",
        ],
    },
    "animal_present": {
        "definition": "An animal is mentioned (real animals; also mythical beasts if used as creatures).",
        "examples": [
            "a fluffy white cat",
            "a herd of horses running",
            "a dragon perched on a cliff",
        ],
    },
    "vehicle_present": {
        "definition": "A vehicle is mentioned (car, ship, plane, bike, train, starship, etc.).",
        "examples": [
            "a red sports car on the highway",
            "an ancient trojan ship in crashing waves",
            "a spaceship orbiting earth",
        ],
    },
    "food_present": {
        "definition": "Food or drink is mentioned.",
        "examples": [
            "a bowl of ramen with chopsticks",
            "fresh strawberries and cream",
            "a glass of red wine on a table",
        ],
    },
    "plant_present": {
        "definition": "Plants, trees, flowers, grass, or vegetation are mentioned.",
        "examples": [
            "a field of sunflowers",
            "an oak tree beside a river",
            "potted ferns in a bright room",
        ],
    },
    "landmark_present": {
        "definition": "A named landmark, monument, or famous place is mentioned.",
        "examples": [
            "the eiffel tower at night",
            "mount fuji in spring",
            "times square neon lights",
        ],
    },
    "count_exact": {
        "definition": "An exact numeric count is specified.",
        "examples": [
            "three cats sitting on a fence",
            "2 red apples and 1 green apple",
            "exactly five candles on a cake",
        ],
    },
    "count_approx": {
        "definition": "An approximate / vague quantity is specified (not an exact number).",
        "examples": [
            "several birds in the sky",
            "a few books on a shelf",
            "many lanterns floating over a lake",
        ],
    },
    "count_multi_object": {
        "definition": "Counting spans multiple different object types or groups.",
        "examples": [
            "two dogs and three cats",
            "four red apples and two green pears",
            "one knight and several horses",
        ],
    },
    "color_binding": {
        "definition": "A color is bound to a specific object/entity (not just a vague palette note).",
        "examples": [
            "a red sports car",
            "blue eyes and black hair",
            "a girl in a yellow raincoat",
        ],
        "not_examples": [
            "pastel color palette",  # global palette, not object-bound
            "vibrant colors",
        ],
    },
    "shape_binding": {
        "definition": "A shape is bound to a specific object.",
        "examples": [
            "a round wooden table",
            "square windows on a brick building",
            "a triangular sandwich",
        ],
    },
    "material_binding": {
        "definition": "A material is bound to a specific object.",
        "examples": [
            "a wooden chair with blue cushions",
            "a glass vase holding flowers",
            "black leather jacket",
        ],
    },
    "texture_binding": {
        "definition": "A texture is bound to a specific object.",
        "examples": [
            "a fluffy white cat",
            "rough stone walls",
            "smooth polished marble floor",
        ],
    },
    "size_binding": {
        "definition": "A size is bound to a specific object.",
        "examples": [
            "a tiny hummingbird",
            "a huge mountain under stormy clouds",
            "a tall and thin hooded alien",
        ],
    },
    "style_binding": {
        "definition": "A named visual style/aesthetic is applied to the image content (genre/medium attached to the scene).",
        "examples": [
            "cyberpunk neon city",
            "watercolor illustration of a forest",
            "studio ghibli style girl",
        ],
    },
    "attribute_binding_multi": {
        "definition": "Multiple different attributes are bound to object(s) (e.g. color+material+size together).",
        "examples": [
            "a small red wooden toy car",
            "tall thin gray-skinned alien in a dark ornate cloak",
            "fluffy white cat on a wooden chair with blue cushions",
        ],
    },
    "spatial_2d": {
        "definition": "2D layout words: left/right, above/below, beside, next to, top/bottom of frame.",
        "examples": [
            "a cat to the left of a dog",
            "the moon above the mountains",
            "a lamp beside the bed",
        ],
    },
    "spatial_3d": {
        "definition": "3D depth/containment: in front of, behind, inside, under, on top of, through.",
        "examples": [
            "a cat sitting on a dog's head",
            "a man standing behind a desk",
            "books inside a glass cabinet",
        ],
    },
    "relative_position": {
        "definition": "Relative positioning between entities is specified (often overlaps spatial_2d/3d).",
        "examples": [
            "the tower is next to the river",
            "a bird flying above the trees",
            "a child in front of her mother",
        ],
    },
    "perspective": {
        "definition": "Camera / viewpoint is specified.",
        "examples": [
            "aerial view of a coastal city",
            "extreme close-up of an eye",
            "wide angle shot, low angle, 35mm",
        ],
    },
    "object_interaction": {
        "definition": "Objects physically interact (touching, colliding, connected, attacking, intertwined).",
        "examples": [
            "red wires attack a man",
            "two trains colliding",
            "vines wrapping around a statue",
        ],
    },
    "comparative_relation": {
        "definition": "An explicit comparison between entities.",
        "examples": [
            "a dog taller than a cat",
            "the left building is bigger than the right one",
            "more stars than clouds in the sky",
        ],
    },
    "multi_relation": {
        "definition": "Two or more distinct relations/interactions are present in the same prompt.",
        "examples": [
            "a cat sitting on a dog and looking at the moon",
            "a girl holding a radio while sitting at a table",
            "a bird above a lake next to a wooden dock",
        ],
    },
    "human_action": {
        "definition": "A human is performing an action.",
        "examples": [
            "a woman sitting on the moon",
            "a girl building a radio",
            "people dancing in the street",
        ],
    },
    "animal_action": {
        "definition": "An animal is performing an action.",
        "examples": [
            "a cat sitting on a dog's head",
            "horses running across a field",
            "a bird diving toward the water",
        ],
    },
    "object_manipulation": {
        "definition": "A human/agent is manipulating an object (holding, building, using, carrying).",
        "examples": [
            "a girl building a radio",
            "a chef chopping vegetables",
            "a knight holding a glowing spear",
        ],
    },
    "indoor_scene": {
        "definition": "The setting is indoors / interior.",
        "examples": [
            "in a red laboratory",
            "family portrait in the main room of a castle",
            "sitting at a table in a dark room",
        ],
    },
    "outdoor_scene": {
        "definition": "The setting is outdoors.",
        "examples": [
            "a landscape of sparse grassland",
            "city street at night",
            "a beach under a sunset sky",
        ],
    },
    "urban_scene": {
        "definition": "Urban / city environment.",
        "examples": [
            "neon tokyo alleyway",
            "skyline of new york",
            "busy traffic on a downtown avenue",
        ],
    },
    "natural_scene": {
        "definition": "Natural environment (forest, mountain, ocean, desert, sky, grassland).",
        "examples": [
            "misty forest at dawn",
            "sparse grassland with conifers",
            "ocean waves under a stormy sky",
        ],
    },
    "art_style": {
        "definition": "An art medium / illustration style is requested (painting, illustration, sketch, digital art, watercolor).",
        "examples": [
            "digital painting, concept art",
            "oil paint on canvas",
            "watercolor illustration",
        ],
    },
    "photographic_style": {
        "definition": "Photographic / camera fidelity cues (photo, polaroid, 35mm, bokeh, HDR, photorealistic).",
        "examples": [
            "vintage polaroid photo, hyper realistic",
            "35mm lens, depth of field, bokeh",
            "photograph of a real-life ice queen, 8k",
        ],
    },
    "artist_style": {
        "definition": "A named artist or 'by / in the style of <artist>' is mentioned.",
        "examples": [
            "by greg rutkowski and alphonse mucha",
            "in the style of beksinski",
            "painting by edward hopper",
        ],
        "not_examples": [
            "trending on artstation",  # platform, not artist
        ],
    },
    "genre_style": {
        "definition": "A genre / aesthetic movement is mentioned (cyberpunk, gothic, anime, baroque, sci-fi, noir, etc.).",
        "examples": [
            "science fiction, retrofuturistic",
            "gothic, dark fantasy",
            "anime style portrait",
        ],
    },
    "surreal_scene": {
        "definition": "Surreal / dreamlike / illogical imagery.",
        "examples": [
            "melting clocks in a desert",
            "a man with a fish head in a red laboratory",
            "doors floating in the sky",
        ],
    },
    "anti_realism": {
        "definition": "Explicitly non-photoreal / abstract / distorted / anti-real content.",
        "examples": [
            "abstract geometric shapes, anti-realism",
            "distorted faces, unreal proportions",
            "cubist fragmentation of a city",
        ],
    },
    "fantasy_content": {
        "definition": "Fantasy / mythical / magical content.",
        "examples": [
            "an elf mage casting a spell",
            "a dragon flying over a castle",
            "enchanted forest with glowing runes",
        ],
    },
    "text_in_image": {
        "definition": "Readable text / words / letters are requested to appear in the image.",
        "examples": [
            "a billboard that says 'OPEN 24 HOURS'",
            "a book cover with the title 'DREAMS'",
            "neon sign reading 'HOTEL'",
        ],
    },
    "symbol_rendering": {
        "definition": "Symbols, icons, glyphs, emblems, or glyphs should appear.",
        "examples": [
            "an ankh symbol glowing on a wall",
            "alchemical glyphs circling a portal",
            "warning icons on a control panel",
        ],
    },
    "logo_or_sign": {
        "definition": "A logo, brand mark, or signage should appear.",
        "examples": [
            "the nike logo on a sneaker",
            "a street sign saying Main Street",
            "a cafe storefront with a logo sign",
        ],
    },
}

CATEGORIES: tuple[str, ...] = tuple(CATEGORY_SPECS.keys())
assert len(CATEGORIES) == 42, f"Expected 42 categories, got {len(CATEGORIES)}"

# Backward-compatible alias used elsewhere in this file.
CATEGORY_DEFINITIONS = {name: spec["definition"] for name, spec in CATEGORY_SPECS.items()}


####
# Build the focused yes/no instruction used to test one semantic category.
####
def build_category_system_prompt(category: str) -> str:
    """Build a focused yes/no prompt for ONE category (reduces multi-label hallucination)."""
    spec = CATEGORY_SPECS[category]
    lines = [
        "You are an expert AIGC prompt taxonomy labeler.",
        f'Decide whether THIS ONE category applies to the given prompt: "{category}".',
        "",
        f"Definition: {spec['definition']}",
        "",
        "Positive examples (these SHOULD be labeled true):",
    ]
    for ex in spec.get("examples", []):
        lines.append(f'  - "{ex}" -> {{"applies": true}}')
    not_examples = spec.get("not_examples") or []
    if not_examples:
        lines.append("")
        lines.append("NOT examples (these SHOULD be labeled false):")
        for ex in not_examples:
            lines.append(f'  - "{ex}" -> {{"applies": false}}')
    lines.extend(
        [
            "",
            "Rules:",
            "- Answer ONLY for this single category. Ignore all other categories.",
            "- Mark true ONLY if there is clear evidence in the prompt text.",
            "- Do NOT invent content that is not stated or strongly implied.",
            '- Output ONLY one compact JSON object: {"applies": true} or {"applies": false}',
            "- No markdown, no explanations, no thinking text.",
        ]
    )
    return "\n".join(lines)


####
# Load shared model-loading and text-generation utilities for categorization.
####
def load_scorer_helpers():
    """Reuse Qwen3.6-27B load/generate helpers from the nine-factor scorer."""
    path = SCRIPT_DIR / "score_prompts_qwen3_6_27b_one_dimension.py"
    spec = importlib.util.spec_from_file_location("score_qwen3_6_27b", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["score_qwen3_6_27b"] = module
    spec.loader.exec_module(module)
    return module


####
# Extract and parse the first complete JSON-like object in model output.
####
def extract_first_json_object(text: str) -> dict[str, Any]:
    cleaned = (
        text.replace("```json", "")
        .replace("```", "")
        .replace("<think>", "")
        .replace("</think>", "")
        .strip()
    )
    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in model output: {text[:200]!r}")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                json_text = cleaned[start : index + 1]
                try:
                    parsed = json.loads(json_text)
                except json.JSONDecodeError:
                    parsed = ast.literal_eval(json_text)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Parsed JSON is not an object: {json_text[:200]!r}")
                return parsed

    raise ValueError(f"No complete JSON object found in model output: {text[:200]!r}")


####
# Convert common model-produced boolean representations into a Python bool.
####
def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


APPLIES_RE = re.compile(r'"?applies"?\s*[:=]\s*(true|false|1|0)', re.IGNORECASE)
TRUE_FALSE_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)


####
# Parse whether a model response says that the requested category applies.
####
def parse_single_category_applies(generated: str) -> bool:
    """Parse a single-category yes/no response into a boolean."""
    try:
        raw = extract_first_json_object(generated)
        for key in ("applies", "apply", "label", "present", "value", "result"):
            if key in raw:
                return _as_bool(raw[key])
        # If the model returned a single bool-like value under any key
        if len(raw) == 1:
            return _as_bool(next(iter(raw.values())))
        for value in raw.values():
            if isinstance(value, (bool, int, float, str)):
                return _as_bool(value)
    except Exception:
        pass

    match = APPLIES_RE.search(generated)
    if match:
        return match.group(1).lower() in {"true", "1"}

    # Prefer the LAST true/false/yes/no so reasoning text before the answer is ignored.
    matches = TRUE_FALSE_RE.findall(generated)
    if matches:
        return matches[-1].lower() in {"true", "yes"}

    raise ValueError(f"Could not parse applies yes/no from model output: {generated[:200]!r}")


####
# Resolve the conflict where both single-object and multi-object are selected.
####
def enforce_single_multi_exclusion(labels: dict[str, bool]) -> dict[str, bool]:
    """Prefer multi_object when both single_object and multi_object are true."""
    if labels.get("single_object") and labels.get("multi_object"):
        labels["single_object"] = False
    return labels


####
# Make one model call to classify one prompt against one category.
####
def classify_one_category(
    scorer: Any,
    processor: Any,
    model: Any,
    prompt: str,
    category: str,
    max_new_tokens: int,
) -> bool:
    """One Qwen call: does `category` apply to `prompt`?"""
    system_prompt = build_category_system_prompt(category)
    user_prompt = (
        f"Category: {category}\n"
        f"Prompt: {json.dumps(prompt, ensure_ascii=False)}\n\n"
        'Return only JSON: {"applies": true} or {"applies": false}'
    )
    generated = scorer.generate_dimension(
        processor, model, system_prompt, user_prompt, max_new_tokens
    )
    return parse_single_category_applies(generated)


####
# Load prompts from plain text, JSONL, or supported JSON structures.
####
def load_input_prompts(path: Path) -> list[dict[str, Any]]:
    """Accept sorted JSON array, JSONL records, or plain .txt (one prompt per line)."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Plain prompt dump: one prompt per line (tier1/2/3 .txt files).
    if path.suffix.lower() == ".txt":
        return [
            {"prompt": line.rstrip("\r\n"), "score": None}
            for line in text.splitlines()
            if line.strip()
        ]

    if path.suffix.lower() == ".jsonl" or (not text.startswith("[") and not text.startswith("{")):
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL line {line_number}: {exc}") from exc
        return records

    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "prompts" in data and isinstance(data["prompts"], list):
        return data["prompts"]
    raise SystemExit(f"Unsupported input JSON shape in {path}")


####
# Count completed non-empty records in a JSONL output file.
####
def count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


####
# Produce a short single-line model-output excerpt for error reporting.
####
def generated_excerpt(text: str, limit: int = 500) -> str:
    return text.replace("\n", "\\n")[:limit]


####
# Evaluate all 42 categories for one prompt and collect any parsing errors.
####
def assign_categories_for_prompt(
    scorer: Any,
    processor: Any,
    model: Any,
    record: dict[str, Any],
    max_new_tokens: int,
    strict_json: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Classify one prompt with 42 separate Qwen calls (one category at a time)."""
    prompt = str(record.get("prompt", "")).strip()
    labels: dict[str, bool] = {name: False for name in CATEGORIES}
    category_errors: list[dict[str, Any]] = []

    # One focused yes/no call per category — avoids the model juggling all 42 at once.
    for category in CATEGORIES:
        generated = ""
        try:
            system_prompt = build_category_system_prompt(category)
            user_prompt = (
                f"Category: {category}\n"
                f"Prompt: {json.dumps(prompt, ensure_ascii=False)}\n\n"
                'Return only JSON: {"applies": true} or {"applies": false}'
            )
            generated = scorer.generate_dimension(
                processor, model, system_prompt, user_prompt, max_new_tokens
            )
            labels[category] = parse_single_category_applies(generated)
        except Exception as exc:
            if strict_json:
                raise
            labels[category] = False
            category_errors.append(
                {
                    "category": category,
                    "error": str(exc),
                    "generated_excerpt": generated_excerpt(generated),
                }
            )

    labels = enforce_single_multi_exclusion(labels)
    assigned = [name for name, flag in labels.items() if flag]
    out = {
        "line_number": record.get("line_number"),
        "prompt": prompt,
        "complexity_score": record.get("score"),
        "num_categories": len(assigned),
        "assigned_categories": assigned,
        "categories": labels,
        "qwen_calls": len(CATEGORIES),
    }
    for key in (
        "f1_prompt_length",
        "object_count",
        "attribute_count",
        "f8_color_specifications",
    ):
        if key in record:
            out[key] = record[key]

    error = None
    if category_errors:
        out["category_errors"] = category_errors
        error = {
            "line_number": record.get("line_number"),
            "prompt": prompt,
            "num_category_errors": len(category_errors),
            "category_errors": category_errors,
        }
    return out, error


####
# Combine resumable JSONL category records into one formatted JSON array.
####
def write_full_json(jsonl_path: Path, json_path: Path) -> int:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return len(records)


####
# Summarize category frequencies, completion progress, and output locations.
####
def write_summary(
    summary_path: Path,
    args: argparse.Namespace,
    total_input: int,
    completed: int,
    jsonl_path: Path,
) -> None:
    # Aggregate category frequencies from whatever has been written so far.
    counts = {name: 0 for name in CATEGORIES}
    n = 0
    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                n += 1
                for name in CATEGORIES:
                    if record.get("categories", {}).get(name):
                        counts[name] += 1

    summary = {
        "input": str(args.input),
        "jsonl_output": str(args.jsonl_output),
        "json_output": str(args.json_output),
        "error_jsonl": str(args.error_output),
        "model": args.model,
        "num_categories": len(CATEGORIES),
        "categories": list(CATEGORIES),
        "total_input_prompts": total_input,
        "completed_prompts": completed,
        "remaining_prompts": max(0, total_input - completed),
        "category_positive_counts": counts,
        "category_positive_rates": {
            name: (counts[name] / n if n else 0.0) for name in CATEGORIES
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


####
# Load scored prompts from sorted JSON when available, otherwise from JSONL.
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


####
# Sort scored prompt records from highest to lowest complexity.
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


####
# Map a complexity score to Tier 1, Tier 2, or Tier 3.
####
def tier_name_for_score(score: float, tier1_min: float, tier2_min: float) -> str:
    if score >= tier1_min:
        return "tier1"
    if score >= tier2_min:
        return "tier2"
    return "tier3"


####
# Select the top-scoring prompts and write tier files plus score statistics.
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


####
# Categorize one tier with resume support and reuse the loaded model.
####
def stage5_assign_categories(
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

    records = load_input_prompts(input_path)
    total_input = len(records)
    already_done = count_jsonl_records(jsonl_output)
    todo = records[already_done:]
    if category_limit is not None:
        todo = todo[:category_limit]

    print(f"    input prompts : {total_input:,} ({input_path})")
    print(f"    already done  : {already_done:,}")
    print(f"    this run      : {len(todo):,}")

    if not todo:
        print("    nothing to do for this tier.")
        if jsonl_output.exists():
            write_full_json(jsonl_output, json_output)
            summary_args = SimpleNamespace(
                input=input_path,
                jsonl_output=jsonl_output,
                json_output=json_output,
                error_output=error_output,
                model=model,
            )
            write_summary(
                summary_output, summary_args, total_input, already_done, jsonl_output
            )
        return 0, already_done, scorer, processor, model_obj

    if processor is None or model_obj is None or scorer is None:
        print("    Loading model for Stage 5...")
        scorer = load_scorer_helpers()
        processor, model_obj = scorer.load_model(model, dtype, device_map)

    from tqdm import tqdm

    processed = 0
    try:
        with tqdm(total=len(todo), desc=f"Categories [{input_path.stem}]", unit="prompt") as progress:
            with jsonl_output.open("a", encoding="utf-8") as jsonl, error_output.open(
                "a", encoding="utf-8"
            ) as err_handle:
                for record in todo:
                    out, error = assign_categories_for_prompt(
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
            n = write_full_json(jsonl_output, json_output)
            summary_args = SimpleNamespace(
                input=input_path,
                jsonl_output=jsonl_output,
                json_output=json_output,
                error_output=error_output,
                model=model,
            )
            write_summary(
                summary_output, summary_args, total_input, completed, jsonl_output
            )
            print(f"    Wrote full category JSON ({n:,} records): {json_output}")

    return processed, already_done + processed, scorer, processor, model_obj


####
# Define all command-line paths, thresholds, model options, and stage controls.
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


####
# Run preprocessing, scoring, sorting, tiering, and category assignment.
####
def main() -> None:
    args = parse_args()
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

    # ==================================================================
    # Stages 1-3: preprocessing
    # ==================================================================
    if not args.skip_preprocess:
        overlap = load_overlap_module()

        print("=" * 70)
        print("Stage 1/5: removing EXACT duplicate prompts")
        print(f"    input : {args.input}")
        total1, kept1 = stage1_remove_exact_duplicates(
            args.input, args.unique_output, case_insensitive=not args.case_sensitive
        )
        print(f"    output: {args.unique_output}")
        print(f"    read {total1:,} -> kept {kept1:,} (removed {total1 - kept1:,} exact duplicates)")

        print("=" * 70)
        print(f"Stage 2/5: removing prompts shorter than {args.min_chars} characters")
        total2, kept2 = stage2_filter_min_length(
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
        total3, kept3, removed3 = stage3_remove_overlaps(
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

    # ==================================================================
    # Stage 4: nine-factor complexity scoring
    # ==================================================================
    score_processed = score_completed = 0
    if not args.skip_score:
        print("=" * 70)
        print("Stage 4/5: scoring with Qwen3.6-27B (nine complexity factors)")
        print(f"    input : {args.kept_output}")
        scorer = load_scorer_module()
        score_processed, score_completed = stage4_score_with_qwen27b(
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

    # ==================================================================
    # Stage 4b: sort by complexity score (descending)
    # ==================================================================
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

    # ==================================================================
    # Stage 4c: build tiered 500k corpus → tier1/2/3 txt files
    # ==================================================================
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

    # ==================================================================
    # Stage 5: 42-category assignment UNDER EACH tier file
    # ==================================================================
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


if __name__ == "__main__":
    main()
