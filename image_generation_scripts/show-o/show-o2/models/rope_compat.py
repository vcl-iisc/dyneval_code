import math

import torch

try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
except ModuleNotFoundError:
    def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
        if config is not None:
            base = config.rope_theta
            dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        else:
            base = rope_kwargs["base"]
            dim = rope_kwargs["dim"]

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
        )
        return inv_freq, 1.0

    def _compute_llama3_rope_parameters(config, device, seq_len=None, **rope_kwargs):
        inv_freq, attention_factor = _compute_default_rope_parameters(config, device, seq_len, **rope_kwargs)

        scaling = config.rope_scaling
        factor = scaling["factor"]
        low_freq_factor = scaling["low_freq_factor"]
        high_freq_factor = scaling["high_freq_factor"]
        old_context_len = scaling["original_max_position_embeddings"]

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor
        wavelen = 2 * math.pi / inv_freq

        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq / factor + smooth_factor * inv_freq
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        return inv_freq_llama, attention_factor

    ROPE_INIT_FUNCTIONS = {
        "default": _compute_default_rope_parameters,
        "llama3": _compute_llama3_rope_parameters,
    }
