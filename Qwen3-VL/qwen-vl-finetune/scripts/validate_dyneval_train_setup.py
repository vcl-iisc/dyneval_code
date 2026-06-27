#!/usr/bin/env python3
"""Preflight checks for DynEval curriculum fine-tuning."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from qwenvl.train.dyneval_tokens import (
    DYNEVAL_ADDITIONAL_SPECIAL_TOKENS,
    DYNEVAL_TASK_TOKEN_IDS,
    DYNEVAL_TASK_TOKENS,
    validate_task_token_ids,
)
from qwenvl.train.model_utils import resolve_model_backend


def check_model_backend_resolution() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint = Path(tmp_dir) / "stage1"
        checkpoint.mkdir()
        config = {
            "model_type": "qwen3_vl",
            "architectures": ["Qwen3VLForConditionalGeneration"],
        }
        (checkpoint / "config.json").write_text(json.dumps(config), encoding="utf-8")

        assert resolve_model_backend(str(checkpoint)) == "qwen3vl"
        assert resolve_model_backend("Qwen/Qwen3-VL-4B-Instruct") == "qwen3vl"
        assert resolve_model_backend("Qwen/Qwen3-VL-2B-Instruct") == "qwen3vl"


def check_task_token_constants() -> None:
    assert DYNEVAL_TASK_TOKENS == ["<|T2IA|>", "<|IQA|>", "<|EVALUATION|>"]
    assert DYNEVAL_TASK_TOKEN_IDS == {
        "<|T2IA|>": 151669,
        "<|IQA|>": 151670,
        "<|EVALUATION|>": 151671,
    }
    assert DYNEVAL_ADDITIONAL_SPECIAL_TOKENS == "<|T2IA|>,<|IQA|>,<|EVALUATION|>"


def check_argument_parsing() -> None:
    try:
        from transformers import HfArgumentParser
    except ModuleNotFoundError:
        print("Skipping argument parsing check: transformers not installed in this environment.")
        return

    from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, _ = parser.parse_args_into_dataclasses(
        [
            "--model_name_or_path",
            "Qwen/Qwen3-VL-4B-Instruct",
            "--dataset_use",
            "dynevalinstruct_t2ia",
            "--output_dir",
            "./output/test",
        ]
    )
    assert model_args.additional_special_tokens == DYNEVAL_ADDITIONAL_SPECIAL_TOKENS
    assert data_args.dataset_use == "dynevalinstruct_t2ia"


def check_tokenizer_registration() -> None:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        print("Skipping tokenizer registration check: transformers not installed in this environment.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct",
        trust_remote_code=True,
        use_fast=False,
    )
    num_new_tokens = tokenizer.add_special_tokens(
        {"additional_special_tokens": DYNEVAL_TASK_TOKENS}
    )
    assert num_new_tokens == len(DYNEVAL_TASK_TOKENS), (
        f"Expected {len(DYNEVAL_TASK_TOKENS)} new tokens, got {num_new_tokens}. "
        "The base tokenizer may already include DynEval tokens."
    )
    validate_task_token_ids(tokenizer)


def check_dataset_env_override() -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        tmp.write("[]")
        annotation_path = tmp.name

    os.environ["DYNEVALINSTRUCT_T2IA_ANNOTATION"] = annotation_path
    os.environ["DYNEVALINSTRUCT_T2IA_DATA"] = "/tmp/images"

    from importlib import reload
    import qwenvl.data as data_module

    reload(data_module)
    config = data_module.data_dict["dynevalinstruct_t2ia"]
    assert config["annotation_path"] == annotation_path
    assert config["data_path"] == "/tmp/images"

    os.environ.pop("DYNEVALINSTRUCT_T2IA_ANNOTATION", None)
    os.environ.pop("DYNEVALINSTRUCT_T2IA_DATA", None)
    Path(annotation_path).unlink(missing_ok=True)


def main() -> None:
    check_model_backend_resolution()
    check_task_token_constants()
    check_argument_parsing()
    check_dataset_env_override()
    check_tokenizer_registration()
    print("DynEval training preflight checks passed.")
    print(f"Task tokens: {DYNEVAL_TASK_TOKEN_IDS}")


if __name__ == "__main__":
    main()
