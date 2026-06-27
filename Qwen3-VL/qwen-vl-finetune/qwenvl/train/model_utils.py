import json
from pathlib import Path


def resolve_model_backend(model_name_or_path: str) -> str:
    """Resolve the Qwen-VL model family from a HF id or a local checkpoint path."""
    path_lower = model_name_or_path.lower()
    path_name = Path(model_name_or_path.rstrip("/")).name.lower()

    if "qwen3" in path_lower and "a" in path_name:
        return "qwen3_moe"
    if "qwen3" in path_lower:
        return "qwen3vl"
    if "qwen2.5" in path_lower:
        return "qwen2.5vl"

    config_path = Path(model_name_or_path) / "config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as config_file:
            config = json.load(config_file)
        model_type = str(config.get("model_type", "")).lower()
        architectures = " ".join(config.get("architectures", [])).lower()

        if "qwen3_vl_moe" in model_type or "qwen3vlmoe" in architectures:
            return "qwen3_moe"
        if "qwen3_vl" in model_type or "qwen3vl" in architectures:
            return "qwen3vl"
        if "qwen2_5_vl" in model_type or "qwen2_5_vl" in architectures:
            return "qwen2.5vl"
        if "qwen2_vl" in model_type or "qwen2vl" in architectures:
            return "qwen2vl"

    return "qwen2vl"
