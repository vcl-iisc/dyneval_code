import random
import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_FILE = SCRIPT_DIR / "prompts" / "diffusiondb-prompts.txt"
PROMPTS_DIR = SCRIPT_DIR / "prompts"
DEFAULT_SORTED_OUTPUT = PROMPTS_DIR / "diffusiondb-prompts-sorted.txt"
TIER_OUTPUT_FILES = {
    1: PROMPTS_DIR / "tier-1-prompts.txt",
    2: PROMPTS_DIR / "tier-2-prompts.txt",
    3: PROMPTS_DIR / "tier-3-prompts.txt",
}

MIN_PROMPT_LENGTH = 30
PROGRESS_INTERVAL = 250_000
NUM_TIERS = 3
TIER2_RANDOM_SEED = 42

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "it", "its",
    "this", "that", "these", "those", "as", "into", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "very", "so", "than", "too", "also", "just", "about", "while", "where", "when",
}

STYLE_ATTRIBUTION_PATTERNS = [
    r"\bin the style of\b",
    r"\bart by\b",
    r"\binspired by\b",
    r"\bpainted by\b",
    r"\bdrawn by\b",
    r"\bstyle of\b",
]

TECHNICAL_TERMS = {
    "octane render", "unreal engine", "ray tracing", "raytrace", "volumetric lighting",
    "cinematic lighting", "studio lighting", "global illumination", "depth of field",
    "lens flare", "bokeh", "photorealistic", "hyperrealistic", "hyper realistic",
    "8k", "4k", "uhd", "hd", "render", "cgi", "3d render", "octane", "unreal engine 5",
    "sharp focus", "macro", "wide angle", "f/1.8", "55mm", "85mm", "35mm",
    "chiaroscuro", "subsurface scattering", "post-processing", "matte painting",
}

DETAIL_DESCRIPTORS = {
    "highly detailed", "hyper detailed", "hyper-detailed", "ultra detailed",
    "extremely detailed", "very detailed", "intricate", "intricate complexity",
    "fine details", "fine detail", "sharp focus", "masterpiece", "ultra hd",
    "detailed portrait", "detailed illustration", "detailed painting",
}

STYLE_KEYWORDS = {
    "cyberpunk", "baroque", "impressionist", "impressionism", "surreal", "surrealism",
    "steampunk", "art deco", "gothic", "neo-noir", "noir", "synthwave", "vaporwave",
    "realism", "abstract", "minimalist", "retrofuturism", "retrofuture", "fantasy",
    "sci-fi", "anime", "pixel art", "low poly", "art nouveau", "renaissance",
}

COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "violet", "pink", "black",
    "white", "gray", "grey", "brown", "gold", "golden", "silver", "crimson", "scarlet",
    "teal", "turquoise", "cyan", "magenta", "maroon", "beige", "ivory", "amber",
    "emerald", "sapphire", "ruby", "indigo", "lavender", "fuchsia", "neon",
}

INTERACTION_VERBS = {
    "how", "emphasize", "support", "explain", "describe", "depict", "show", "highlight",
    "illustrate", "demonstrate", "convey", "express", "represent", "visualize",
}

# Attribute phrases must include descriptive content after the marker.
ATTRIBUTE_PATTERNS = [
    r"\bwith\s+(?!out\b)(?:(?:a|an|the)\s+)?[\w'-]+(?:\s+[\w'-]+){0,5}",
    r"\bwearing\s+(?:(?:a|an|the)\s+)?[\w'-]+(?:\s+[\w'-]+){0,5}",
    r"\bholding\s+(?:(?:a|an|the)\s+)?[\w'-]+(?:\s+[\w'-]+){0,5}",
    r"\bfeaturing\s+(?:(?:a|an|the)\s+)?[\w'-]+(?:\s+[\w'-]+){0,5}",
    r"\bsurrounded\s+by\s+(?:(?:a|an|the)\s+)?[\w'-]+(?:\s+[\w'-]+){0,5}",
    r"\bcovered\s+(?:in|with)\s+(?:(?:a|an|the)\s+)?[\w'-]+(?:\s+[\w'-]+){0,5}",
    r"\bmade\s+of\s+[\w'-]+(?:\s+[\w'-]+){0,5}",
    r"\bcomposed\s+of\s+[\w'-]+(?:\s+[\w'-]+){0,5}",
    r"\bdecorated\s+with\s+[\w'-]+(?:\s+[\w'-]+){0,5}",
    r"\badorned\s+with\s+[\w'-]+(?:\s+[\w'-]+){0,5}",
]

OBJECT_PHRASE_PATTERN = re.compile(r"\b(?:a|an|the)\s+([\w'-]+)")
SUBJECT_BEFORE_ATTRIBUTE_PATTERN = re.compile(
    r"\b([\w'-]+(?:\s+[\w'-]+){0,3})\s+(?:with|wearing|holding|featuring)\s+"
)

ATTRIBUTE_MARKER_WORDS = {
    "with", "wearing", "holding", "featuring", "surrounded", "by", "covered", "in",
    "made", "of", "composed", "decorated", "adorned", "a", "an", "the",
}

SUBJECT_LEAD_STOPWORDS = {
    "within", "without", "through", "during", "into", "onto", "upon", "from", "of",
    "in", "on", "at", "by", "for", "to", "the", "a", "an", "and", "or",
}


def _count_pattern_matches(text, patterns):
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def _count_phrase_matches(text, phrases):
    return sum(text.count(phrase) for phrase in phrases)


def _count_word_matches(words, vocabulary):
    return sum(1 for word in words if word in vocabulary)


def _content_word_count(fragment):
    words = re.findall(r"[a-z0-9']+", fragment.lower())
    return sum(
        1 for word in words
        if word not in STOPWORDS and word not in ATTRIBUTE_MARKER_WORDS and len(word) > 2
    )


def _is_valid_subject_phrase(phrase):
    words = re.findall(r"[a-z0-9']+", phrase.lower())
    if not words or words[0] in SUBJECT_LEAD_STOPWORDS:
        return False
    return _content_word_count(phrase) > 0


def _spans_overlap(first, second):
    return not (first[1] <= second[0] or second[1] <= first[0])


def _count_attribute_phrases(text):
    attribute_hits = 0
    used_spans = []

    for pattern in ATTRIBUTE_PATTERNS:
        for match in re.finditer(pattern, text):
            span = match.span()
            if any(_spans_overlap(span, used) for used in used_spans):
                continue
            if _content_word_count(match.group()) == 0:
                continue
            attribute_hits += 1
            used_spans.append(span)

    return attribute_hits


def _count_object_phrases(text):
    object_hits = 0
    used_spans = []

    for match in OBJECT_PHRASE_PATTERN.finditer(text):
        phrase = match.group(1)
        if _content_word_count(phrase) == 0:
            continue
        span = match.span(1)
        if any(_spans_overlap(span, used) for used in used_spans):
            continue
        object_hits += 1
        used_spans.append(span)

    for match in SUBJECT_BEFORE_ATTRIBUTE_PATTERN.finditer(text):
        phrase = match.group(1)
        if not _is_valid_subject_phrase(phrase):
            continue
        span = match.span(1)
        if any(_spans_overlap(span, used) for used in used_spans):
            continue
        object_hits += 1
        used_spans.append(span)

    if object_hits == 0:
        leading_words = re.findall(r"[a-z0-9']+", text)
        leading_subject = next(
            (
                word for word in leading_words[:4]
                if word not in STOPWORDS
                and word not in SUBJECT_LEAD_STOPWORDS
                and len(word) > 2
            ),
            None,
        )
        if leading_subject:
            object_hits = 1

    return object_hits


def _parse_objects_and_attributes(text):
    """
    Parse object and attribute mentions using bounded phrase patterns instead of
    naive substring matching (which mis-counts words like 'without' as 'with').
    """
    attribute_hits = _count_attribute_phrases(text)
    object_hits = _count_object_phrases(text)
    return object_hits, attribute_hits


def compute_complexity_score(prompt):
    """
    Assign a heuristic complexity score from nine factors:
    (i) prompt length, (ii) object/attribute counts, (iii) compositional density,
    (iv) artist/style attribution, (v) technical rendering terminology,
    (vi) explicit detail descriptors, (vii) high-level style keywords,
    (viii) color specifications, and (ix) interaction verbs.
    """
    text = prompt.lower()
    words = re.findall(r"[a-z0-9']+", text)
    word_count = len(words)

    # (i) Prompt length
    length_score = min(word_count // 12, 4)

    # (ii) Object and attribute counts
    object_hits, attribute_hits = _parse_objects_and_attributes(text)
    object_attribute_score = min(min(object_hits, 3) + min(attribute_hits, 3), 5)

    # (iii) Compositional density via comma-separated clauses
    clause_count = len([clause.strip() for clause in prompt.split(",") if clause.strip()])
    compositional_score = min(max(clause_count - 1, 0), 6)

    # (iv) Artist or style-attribution patterns
    style_attribution_score = min(_count_pattern_matches(text, STYLE_ATTRIBUTION_PATTERNS), 3)

    # (v) Technical rendering and fidelity terminology
    technical_score = min(_count_phrase_matches(text, TECHNICAL_TERMS), 4)

    # (vi) Explicit detail descriptors
    detail_score = min(_count_phrase_matches(text, DETAIL_DESCRIPTORS), 3)

    # (vii) High-level style keywords
    style_keyword_score = min(_count_phrase_matches(text, STYLE_KEYWORDS), 3)

    # (viii) Color specifications
    color_score = min(_count_word_matches(words, COLORS), 3)

    # (ix) Interaction verbs
    interaction_score = min(_count_word_matches(words, INTERACTION_VERBS), 2)

    breakdown = {
        "length": length_score,
        "object_attributes": object_attribute_score,
        "compositional_density": compositional_score,
        "style_attribution": style_attribution_score,
        "technical_terms": technical_score,
        "detail_descriptors": detail_score,
        "style_keywords": style_keyword_score,
        "color_specs": color_score,
        "interaction_verbs": interaction_score,
    }
    total = sum(breakdown.values())
    return total, breakdown


def are_prompts_similar(prompt1, prompt2, similarity_threshold=0.3):
    """
    Check if two prompts are too similar based on word overlap.
    Returns True if they share too many words (indicating similarity).
    """
    words1 = set(re.findall(r"[a-z0-9']+", prompt1.lower()))
    words2 = set(re.findall(r"[a-z0-9']+", prompt2.lower()))

    # Remove common style keywords that shouldn't affect uniqueness
    style_words = {"artstation", "trending", "8k", "4k", "octane", "render",
                   "detailed", "intricate", "highly", "digital", "art", "painting"}
    words1 = words1 - style_words
    words2 = words2 - style_words

    if not words1 or not words2:
        return False

    intersection = len(words1 & words2)
    union = len(words1 | words2)
    if union == 0:
        return False

    return (intersection / union) > similarity_threshold


def _score_and_sort_corpus(input_file, min_prompt_length=MIN_PROMPT_LENGTH):
    """
    Filter prompts shorter than min_prompt_length, score the remainder with the
    nine-factor heuristic, and return the full corpus sorted by complexity.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading prompts from {input_path}...")

    scored_prompts = []
    total_prompts = 0
    removed_short = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if not prompt:
                continue

            total_prompts += 1
            if len(prompt) < min_prompt_length:
                removed_short += 1
                continue

            score, _ = compute_complexity_score(prompt)
            scored_prompts.append((score, prompt))

            if total_prompts % PROGRESS_INTERVAL == 0:
                print(
                    f"  Processed {total_prompts:,} prompts "
                    f"({len(scored_prompts):,} kept after length filter)..."
                )

    scored_prompts.sort(key=lambda item: item[0], reverse=True)
    print(f"Total prompts available: {total_prompts:,}")
    print(
        f"Removed prompts shorter than {min_prompt_length} characters: "
        f"{removed_short:,}"
    )
    print(f"Scored and sorted corpus size: {len(scored_prompts):,}")
    return scored_prompts


def _chunk_sizes(total, num_chunks):
    base_size = total // num_chunks
    remainder = total % num_chunks
    return [base_size + (1 if index < remainder else 0) for index in range(num_chunks)]


def assign_prompt_tiers(ranked_prompts, num_tiers=NUM_TIERS, random_seed=TIER2_RANDOM_SEED):
    """
    Split the sorted corpus into three chunks, then:
    - Tier 1: top half of the highest-score chunk
    - Tier 2: middle chunk shuffled randomly
    - Tier 3: bottom half of the lowest-score chunk
    """
    total = len(ranked_prompts)
    if total == 0:
        return {}

    sizes = _chunk_sizes(total, num_tiers)
    start = 0
    chunks = []
    for size in sizes:
        chunks.append(ranked_prompts[start:start + size])
        start += size

    top_chunk, middle_chunk, bottom_chunk = chunks

    tier1 = [prompt for _, prompt in top_chunk[: len(top_chunk) // 2]]
    tier3 = [prompt for _, prompt in bottom_chunk[len(bottom_chunk) // 2 :]]

    tier2_candidates = [prompt for _, prompt in middle_chunk]
    rng = random.Random(random_seed)
    rng.shuffle(tier2_candidates)
    tier2 = tier2_candidates

    return {1: tier1, 2: tier2, 3: tier3}


def _write_prompt_lines(output_path, prompts):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")


def tier_and_save_prompts(
    input_file,
    sorted_output_file,
    tier_output_files,
    min_prompt_length=MIN_PROMPT_LENGTH,
    random_seed=TIER2_RANDOM_SEED,
):
    """
    Filter by length, score the full corpus, sort by complexity, and save tiers.
    """
    ranked_prompts = _score_and_sort_corpus(input_file, min_prompt_length)
    if not ranked_prompts:
        print("No prompts remained after filtering.")
        return

    sorted_prompts = [prompt for _, prompt in ranked_prompts]
    _write_prompt_lines(sorted_output_file, sorted_prompts)
    print(f"Saved {len(sorted_prompts):,} sorted prompts to {sorted_output_file}")

    tiers = assign_prompt_tiers(ranked_prompts, random_seed=random_seed)
    sizes = _chunk_sizes(len(ranked_prompts), NUM_TIERS)
    chunk_starts = [0]
    for size in sizes[:-1]:
        chunk_starts.append(chunk_starts[-1] + size)

    tier_meta = {
        1: (chunk_starts[0], chunk_starts[0] + sizes[0] // 2),
        2: (chunk_starts[1], chunk_starts[1] + sizes[1]),
        3: (chunk_starts[2] + sizes[2] // 2, chunk_starts[2] + sizes[2]),
    }

    for tier, prompts in tiers.items():
        output_path = tier_output_files[tier]
        _write_prompt_lines(output_path, prompts)
        start, end = tier_meta[tier]
        tier_scores = [score for score, _ in ranked_prompts[start:end]]
        print(
            f"Tier {tier}: {len(prompts):,} prompts "
            f"(score range {min(tier_scores)}-{max(tier_scores)}) -> {output_path}"
        )

    print(
        f"\nSuccessfully tiered {len(sorted_prompts):,} prompts under {PROMPTS_DIR}"
    )


def extract_diverse_prompts(
    input_file,
    output_file,
    num_prompts=10,
    sample_size=1000,
    min_prompt_length=MIN_PROMPT_LENGTH,
):
    """
    Score prompts with the nine-factor heuristic, discard short prompts,
    rank by complexity, and greedily select diverse prompts.
    """
    ranked_prompts = _score_and_sort_corpus(input_file, min_prompt_length)
    if not ranked_prompts:
        print("No prompts remained after filtering.")
        return

    candidate_pool = ranked_prompts[: min(sample_size, len(ranked_prompts))]
    diverse_prompts = []

    for score, prompt in candidate_pool:
        if all(not are_prompts_similar(prompt, selected) for selected in diverse_prompts):
            diverse_prompts.append(prompt)
            print(f"Selected ({len(diverse_prompts)}/{num_prompts}) "
                  f"[score={score}]: {prompt[:60]}...")
            if len(diverse_prompts) >= num_prompts:
                break

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for prompt in diverse_prompts:
            f.write(f"{prompt}\n")

    print(
        f"\nSuccessfully extracted {len(diverse_prompts)} diverse, "
        f"high-complexity prompts to {output_path}"
    )


def _cleanup_generated_artifacts():
    cache_dir = SCRIPT_DIR / "__pycache__"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    try:
        tier_and_save_prompts(
            input_file=DEFAULT_INPUT_FILE,
            sorted_output_file=DEFAULT_SORTED_OUTPUT,
            tier_output_files=TIER_OUTPUT_FILES,
            min_prompt_length=MIN_PROMPT_LENGTH,
            random_seed=TIER2_RANDOM_SEED,
        )
    finally:
        _cleanup_generated_artifacts()
