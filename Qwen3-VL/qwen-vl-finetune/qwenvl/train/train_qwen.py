# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


from qwenvl.train.model_utils import resolve_model_backend
from qwenvl.train.dyneval_tokens import (
    DYNEVAL_TASK_TOKENS,
    parse_task_tokens,
    validate_task_token_ids,
)


def load_model(model_args, training_args, attn_implementation="flash_attention_2"):
    model_path = model_args.model_name_or_path
    backend = resolve_model_backend(model_path)
    dtype = torch.bfloat16 if training_args.bf16 else None
    common_kwargs = {
        "cache_dir": training_args.cache_dir,
        "attn_implementation": attn_implementation,
        "dtype": dtype,
    }

    if backend == "qwen3_moe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_path, **common_kwargs
        )
        return model, "qwen3vl"
    if backend == "qwen3vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, **common_kwargs
        )
        return model, "qwen3vl"
    if backend == "qwen2.5vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, **common_kwargs
        )
        return model, "qwen2.5vl"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, **common_kwargs
    )
    return model, "qwen2vl"


def validate_dataset_paths(data_args):
    from qwenvl.data import data_list

    dataset_names = [name for name in data_args.dataset_use.split(",") if name.strip()]
    if not dataset_names:
        raise ValueError("--dataset_use must specify at least one registered dataset.")

    missing_paths = []
    for config in data_list(dataset_names):
        annotation_path = Path(config["annotation_path"])
        if not annotation_path.exists():
            missing_paths.append(str(annotation_path))

    if missing_paths:
        raise FileNotFoundError(
            "Dataset annotation file(s) not found:\n  - "
            + "\n  - ".join(missing_paths)
            + "\nUpdate qwenvl/data/__init__.py or set DYNEVALINSTRUCT_* env vars."
        )


def add_task_specific_tokens(model, processor, tokenizer, additional_special_tokens):
    requested_tokens = parse_task_tokens(additional_special_tokens)
    if not requested_tokens:
        return

    existing_tokens = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    special_tokens = [token for token in requested_tokens if token not in existing_tokens]

    if special_tokens:
        num_new_tokens = tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        if num_new_tokens > 0:
            current_size = model.get_input_embeddings().weight.shape[0]
            target_size = len(tokenizer)
            if target_size > current_size:
                model.resize_token_embeddings(target_size)
            else:
                rank0_print(
                    f"Tokenizer length {target_size} <= embedding rows {current_size}; "
                    "keeping existing embedding size."
                )
    else:
        num_new_tokens = 0

    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    if set(requested_tokens) == set(DYNEVAL_TASK_TOKENS):
        token_ids = validate_task_token_ids(tokenizer)
    else:
        token_ids = {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in requested_tokens
        }
    rank0_print(
        f"Task-specific tokens ready ({num_new_tokens} newly added): {token_ids}"
    )


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    validate_dataset_paths(data_args)

    model, model_type = load_model(model_args, training_args, attn_implementation)
    data_args.model_type = model_type

    print(
        f"Initialized model from {model_args.model_name_or_path} "
        f"({model.__class__.__name__}, backend={model_type})"
    )
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    add_task_specific_tokens(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        additional_special_tokens=model_args.additional_special_tokens,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    processor.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
