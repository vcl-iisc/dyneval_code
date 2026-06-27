"""DynEval task tokens and human-turn templates for DynEvalInstruct."""

from __future__ import annotations

TOKEN_T2IA = "<|T2IA|>"
TOKEN_IQA = "<|IQA|>"
TOKEN_EVALUATION = "<|EVALUATION|>"

T2IA_INSTRUCTION = (
    "Generate atomic yes/no verification questions for text-to-image alignment, "
    "including distortion checks."
)
IQA_INSTRUCTION = (
    "Parse the image into a scene graph and generate image-quality questions for "
    "shape consistency, distortions, texture fidelity, and spatial cues."
)
EVALUATION_INSTRUCTION = "Answer each question and assign a score from 1 to 5."


def format_t2ia_human(prompt: str) -> str:
    return f"{TOKEN_T2IA}\nPrompt: {prompt.strip()}\n{T2IA_INSTRUCTION}"


def format_iqa_human() -> str:
    return f"<image>\n{TOKEN_IQA}\n{IQA_INSTRUCTION}"


def format_evaluation_human(prompt: str, questions: list[dict]) -> str:
    lines = []
    for index, item in enumerate(questions, start=1):
        question = str(item.get("question", "")).strip()
        if question:
            lines.append(f"{index}. {question}")
    question_block = "\n".join(lines)
    return (
        f"<image>\n{TOKEN_EVALUATION}\n"
        f"Prompt: {prompt.strip()}\n"
        f"Questions:\n{question_block}\n"
        f"{EVALUATION_INSTRUCTION}"
    )
