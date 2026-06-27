"""DynEval task tokens shared by training scripts and validation."""

from __future__ import annotations

from typing import Dict, List

# Registered on Qwen3-VL-2B/4B after the base special-token block (151643-151668).
DYNEVAL_TASK_TOKENS: List[str] = ["<|T2IA|>", "<|IQA|>", "<|EVALUATION|>"]

DYNEVAL_TASK_TOKEN_IDS: Dict[str, int] = {
    "<|T2IA|>": 151669,
    "<|IQA|>": 151670,
    "<|EVALUATION|>": 151671,
}

DYNEVAL_ADDITIONAL_SPECIAL_TOKENS = ",".join(DYNEVAL_TASK_TOKENS)


def parse_task_tokens(additional_special_tokens: str | None) -> List[str]:
    if not additional_special_tokens:
        return list(DYNEVAL_TASK_TOKENS)
    return [
        token.strip()
        for token in additional_special_tokens.split(",")
        if token.strip()
    ]


def validate_task_token_ids(tokenizer) -> Dict[str, int]:
    """Ensure tokenizer IDs match the DynEval checkpoint convention."""
    token_ids = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in DYNEVAL_TASK_TOKENS
    }
    mismatches = [
        f"{token}: expected {expected}, got {token_ids[token]}"
        for token, expected in DYNEVAL_TASK_TOKEN_IDS.items()
        if token_ids.get(token) != expected
    ]
    if mismatches:
        raise ValueError(
            "DynEval task token IDs do not match the expected mapping:\n  - "
            + "\n  - ".join(mismatches)
            + "\nUse "
            + DYNEVAL_ADDITIONAL_SPECIAL_TOKENS
            + " when fine-tuning from Qwen3-VL-2B/4B-Instruct."
        )
    return token_ids
