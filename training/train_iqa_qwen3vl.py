from __future__ import annotations
import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from PIL import Image


TASK_TOKENS = ["<T2IA>", "<IQA>", "<EVALUATION>"]
IQA_TOKEN = "<IQA>"

DEFAULT_MODEL_BY_SIZE = {
    "2b": "Qwen/Qwen3-VL-2B-Instruct",
    "4b": "Qwen/Qwen3-VL-4B-Instruct",
}
DEFAULT_DATA_DIR = Path("data/sft/qwen3vl_iqa_dyneval_sft_data")
DEFAULT_OUTPUT_DIR = Path("checkpoints/final/qwen3vl-iqa-dyneval")


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


def require_training_deps():
    # Import optional training dependencies only when training.
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments
    except Exception as exc:
        raise SystemExit(
            "Training dependencies are not importable. Install compatible versions, for example:\n"
            "  pip install 'transformers>=4.57.0' peft accelerate\n"
            f"Import error: {exc}"
        )
    return AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, LoraConfig, get_peft_model


def distributed_env() -> tuple[int, int]:
    # Return DDP world size/local rank from torchrun environment variables.
    return int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "-1"))


def json_dumps(value: Any) -> str:
    # Serialize assistant targets in a stable, readable JSON format.
    return json.dumps(value, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_image_path(image_ref: str, images_root: Path | None) -> Path | None:
    # Resolve relative/absolute image references against an optional image root.
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


def build_answer_index(answers_dir: Path | None, pair_id: str) -> tuple[list[dict[str, Any]], Any]:
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

        teacher_answers, overall_score = build_answer_index(answers_dir, pair_id)
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


def prepare_iqa_sft(
    scene_graph_dir: Path,
    answers_dir: Path | None,
    images_root: Path | None,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    score_mode: str,
    limit: int | None = None,
) -> dict:
    # Build IQA SFT JSONL train/val files plus a manifest.
    rows, counts = build_iqa_rows(scene_graph_dir, answers_dir, images_root, score_mode)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No IQA rows were built. Check the teacher output folders and image paths.")

    rng = random.Random(seed)
    rng.shuffle(rows)
    val_count = max(1, int(len(rows) * val_ratio)) if len(rows) > 1 else 0
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_rows, output_dir / "train.jsonl")
    write_jsonl(val_rows, output_dir / "val.jsonl")

    manifest = {
        "task": "IQA",
        "scene_graph_dir": str(scene_graph_dir),
        "answers_dir": str(answers_dir) if answers_dir is not None else None,
        "images_root": str(images_root) if images_root is not None else None,
        "score_mode": score_mode,
        "registered_task_tokens": TASK_TOKENS,
        "rows": {
            "iqa_scene_graph": counts["scene_graph"],
            "iqa_question_generation": counts["question_generation"],
            "iqa_evaluation": counts["evaluation"],
            "skipped": counts["skipped"],
            "total": len(rows),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "val_ratio": val_ratio,
        "seed": seed,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class JsonlConversationDataset(Dataset):
    # Tiny JSONL dataset wrapper used by Hugging Face Trainer.

    def __init__(self, path: Path, max_samples: int | None = None):
        self.rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    self.rows.append(json.loads(line))
                if max_samples is not None and len(self.rows) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def tensor_without_batch_dim(key: str, value):
    # Remove processor batch dimension from tensors before custom padding.
    if key in {"input_ids", "attention_mask", "assistant_masks", "mm_token_type_ids"}:
        return value.squeeze(0)
    return value


def mask_assistant_spans(input_ids: torch.Tensor, messages: list[dict], tokenizer) -> torch.Tensor:
    # Find assistant answer spans so loss is only applied to target text.
    labels_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    cursor = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
        if not isinstance(content, str) or not content:
            continue
        answer_ids = tokenizer(content, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        if answer_ids.numel() == 0 or answer_ids.numel() > input_ids.numel():
            continue
        for start in range(cursor, input_ids.numel() - answer_ids.numel() + 1):
            if torch.equal(input_ids[start : start + answer_ids.numel()].cpu(), answer_ids.cpu()):
                labels_mask[start : start + answer_ids.numel()] = True
                cursor = start + answer_ids.numel()
                break
    return labels_mask


@dataclass
class TextConversationCollator:
    # Tokenize image conversations and build labels for SFT.
    processor: object
    max_length: int
    train_all_tokens_if_no_mask: bool = False

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch_inputs = []
        for row in features:
            if row.get("image_path"):
                text = self.processor.apply_chat_template(
                    row["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                image = Image.open(row["image_path"]).convert("RGB")
                encoded = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length,
                )
            else:
                encoded = self.processor.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                    return_assistant_tokens_mask=True,
                    processor_kwargs={
                        "padding": False,
                        "truncation": True,
                        "max_length": self.max_length,
                    },
                )
            item = {key: tensor_without_batch_dim(key, value) for key, value in encoded.items() if torch.is_tensor(value)}
            item["_manual_assistant_mask"] = mask_assistant_spans(item["input_ids"], row["messages"], self.processor.tokenizer)
            batch_inputs.append(item)

        tokenizer = self.processor.tokenizer
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in batch_inputs],
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"] for item in batch_inputs],
            batch_first=True,
            padding_value=0,
        )
        labels = input_ids.clone()

        if "assistant_masks" in batch_inputs[0] and any(item["assistant_masks"].sum().item() > 0 for item in batch_inputs):
            assistant_mask = torch.nn.utils.rnn.pad_sequence(
                [item["assistant_masks"] for item in batch_inputs],
                batch_first=True,
                padding_value=0,
            )
            labels[assistant_mask == 0] = -100
        elif any(item["_manual_assistant_mask"].sum().item() > 0 for item in batch_inputs):
            assistant_mask = torch.nn.utils.rnn.pad_sequence(
                [item["_manual_assistant_mask"] for item in batch_inputs],
                batch_first=True,
                padding_value=0,
            )
            labels[assistant_mask == 0] = -100
        elif not self.train_all_tokens_if_no_mask:
            raise RuntimeError("Could not identify assistant tokens. Use --train-all-tokens-if-no-mask to continue.")

        labels[attention_mask == 0] = -100
        output = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if all("mm_token_type_ids" in item for item in batch_inputs):
            output["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                [item["mm_token_type_ids"] for item in batch_inputs],
                batch_first=True,
                padding_value=0,
            )
        vision_tensor_keys = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        extra_tensor_keys = sorted(
            key
            for item in batch_inputs
            for key in item
            if key in vision_tensor_keys and torch.is_tensor(item[key])
        )
        for key in extra_tensor_keys:
            values = [item[key] for item in batch_inputs if key in item]
            if len(values) == len(batch_inputs):
                output[key] = torch.cat(values, dim=0) if values[0].dim() > 0 else torch.stack(values)
        return output


def register_task_tokens(processor, model) -> int:
    # Register DynEval task tokens and resize embeddings when needed.
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    existing_special_tokens = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged_special_tokens = existing_special_tokens + [
        token for token in TASK_TOKENS if token not in existing_special_tokens
    ]
    old_vocab_size = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": merged_special_tokens})
    added = len(tokenizer) - old_vocab_size
    if added:
        model.resize_token_embeddings(len(tokenizer))
    return added


def count_trainable_parameters(model) -> tuple[int, int]:
    # Return trainable and total parameter counts for logging.
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def train_iqa(args: argparse.Namespace) -> None:
    # Load model/data, configure full or LoRA tuning, and run Trainer.
    AutoProcessor, Qwen3VLForConditionalGeneration, Trainer, TrainingArguments, LoraConfig, get_peft_model = (
        require_training_deps()
    )

    world_size, local_rank = distributed_env()
    is_distributed = world_size > 1
    device_map = args.device_map or "none"
    if is_distributed and device_map != "none":
        raise ValueError("Use --device-map none for torchrun/accelerate multi-GPU DDP training.")
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    if device_map == "auto":
        model_kwargs["device_map"] = "auto"
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)

    added = register_task_tokens(processor, model)
    print(f"Registered task tokens: {TASK_TOKENS}; newly added={added}; vocab_size={len(processor.tokenizer)}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.finetune_mode == "lora":
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        for param in model.parameters():
            param.requires_grad_(True)
        trainable, total = count_trainable_parameters(model)
        print(
            f"Full fine-tuning enabled: trainable params={trainable:,} || "
            f"all params={total:,} || trainable%={100 * trainable / total:.4f}"
        )

    train_ds = JsonlConversationDataset(args.train_file, args.max_train_samples)
    eval_ds = JsonlConversationDataset(args.val_file, args.max_val_samples) if args.val_file.exists() else None
    collator = TextConversationCollator(
        processor=processor,
        max_length=args.max_length,
        train_all_tokens_if_no_mask=args.train_all_tokens_if_no_mask,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_ds else "no",
        save_strategy="steps",
        optim=args.optim,
        remove_unused_columns=False,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters if is_distributed else None,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


def build_parser() -> argparse.ArgumentParser:
    # Create CLI arguments for IQA SFT data preparation and training.
    parser = argparse.ArgumentParser(description="Train Qwen3-VL for the DynEval <IQA> task token.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")

    # Data preparation inputs (teacher outputs).
    parser.add_argument("--scene-graph-dir", type=Path, default=None, help="Directory of teacher IQA scene-graph/question JSON files (iqa_outputs).")
    parser.add_argument("--answers-dir", type=Path, default=None, help="Directory of teacher IQA answer JSON files (iqa_answers). Optional; enables scoring rows.")
    parser.add_argument("--images-root", type=Path, default=None, help="Root folder used to resolve image paths.")
    parser.add_argument(
        "--score-mode",
        choices=["correct", "overall"],
        default="correct",
        help="Per-question 1-5 score source: 'correct' maps correct/incorrect to 5/1; 'overall' uses the teacher's overall score.",
    )

    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--val-file", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--model-size", choices=["2b", "4b"], default="4b", help="Original Qwen3-VL checkpoint size to use when --model-path is not provided.")
    parser.add_argument("--model-path", type=str, default=None, help="Model path or Hugging Face repo id. Overrides --model-size. Use a T2IA/EVALUATION checkpoint to add IQA on top.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--finetune-mode",
        choices=["lora", "full"],
        default="full",
        help="Use 'lora' for adapter training or 'full' to update and save all model weights.",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler-type", type=str, default="cosine", help="Learning-rate scheduler type passed to Trainer, e.g. linear or cosine.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--optim", type=str, default="adamw_torch", help="Trainer optimizer. For full fine-tuning on limited VRAM, try 'adafactor'.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--report-to", type=str, default="none", help="Trainer reporting backend. Use 'wandb' to log to Weights & Biases.")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action="store_true",
        help="Enable DDP unused-parameter detection for mixed batch shapes.",
    )
    parser.add_argument("--train-all-tokens-if-no-mask", action="store_true")
    parser.add_argument(
        "--device-map",
        choices=["auto", "none"],
        default=None,
        help="Use 'none' for training. 'auto' may offload modules before token embeddings are resized and can fail after registering task tokens.",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.prepare_only and args.train_only:
        raise SystemExit("Use only one of --prepare-only or --train-only.")

    if not args.train_only and args.scene_graph_dir is None:
        raise SystemExit(
            "Missing --scene-graph-dir for IQA data preparation. "
            "Pass the teacher IQA scene-graph/question folder, or use --train-only with a prepared --data-dir."
        )

    if args.train_file is None:
        args.train_file = args.data_dir / "train.jsonl"
    if args.val_file is None:
        args.val_file = args.data_dir / "val.jsonl"
    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_BY_SIZE[args.model_size]

    if not args.train_only:
        manifest = prepare_iqa_sft(
            scene_graph_dir=args.scene_graph_dir,
            answers_dir=args.answers_dir,
            images_root=args.images_root,
            output_dir=args.data_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            score_mode=args.score_mode,
            limit=args.limit,
        )
        print(json.dumps(manifest, indent=2))

    if args.prepare_only:
        return

    if not args.train_file.exists():
        raise SystemExit(f"Missing train file: {args.train_file}. Run without --train-only first.")

    train_iqa(args)


if __name__ == "__main__":
    main()
