
import argparse
import os
import time

DEFAULT_OUT = "outputs"
DEFAULT_MODEL = "Shitao/OmniGen-v1"
RECOMMENDED_TRANSFORMERS = "4.45.2"


def _version_tuple(version_str):
    parts = []
    for piece in version_str.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _patch_omnigen_phi3_transformer():
    """OmniGen's Phi3Transformer skips position_embeddings required since transformers 4.46."""
    import torch
    import transformers
    from OmniGen import transformer as omnigen_transformer
    from transformers.cache_utils import DynamicCache
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if _version_tuple(transformers.__version__) < (4, 46, 0):
        return False
    if getattr(omnigen_transformer.Phi3Transformer.forward, "_dyneval_patched", False):
        return True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        offload_model=False,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, DynamicCache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if attention_mask is not None and attention_mask.dim() == 3:
            dtype = inputs_embeds.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = attention_mask.unsqueeze(1).to(inputs_embeds.dtype)
        else:
            raise Exception("attention_mask parameter was unavailable or invalid")

        hidden_states = inputs_embeds
        position_embeddings = None
        if hasattr(self, "rotary_emb"):
            if cache_position is None:
                past_seen_tokens = (
                    past_key_values.get_seq_length() if past_key_values is not None else 0
                )
                cache_position = torch.arange(
                    past_seen_tokens,
                    past_seen_tokens + inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        layer_idx = -1
        for decoder_layer in self.layers:
            layer_idx += 1

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                if offload_model and not self.training:
                    self.get_offlaod_layer(layer_idx, device=inputs_embeds.device)
                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
                if position_embeddings is not None:
                    layer_kwargs["position_embeddings"] = position_embeddings
                layer_outputs = decoder_layer(**layer_kwargs)

            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if output_attentions and len(layer_outputs) > 1:
                    all_self_attns += (layer_outputs[1],)
            else:
                hidden_states = layer_outputs

            # DynamicCache is updated in-place; newer transformers no longer
            # return past_key_value inside layer_outputs.
            if use_cache:
                next_decoder_cache = past_key_values

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache and next_cache is not None:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    forward._dyneval_patched = True
    omnigen_transformer.Phi3Transformer.forward = forward
    return True


def load_pipeline(model_name_or_path):
    import transformers
    from OmniGen import OmniGenPipeline

    patched = _patch_omnigen_phi3_transformer()
    if patched:
        print(
            f"Applied OmniGen Phi3 patch for transformers {transformers.__version__}. "
            f"For a clean env you can also use transformers=={RECOMMENDED_TRANSFORMERS}."
        )
    elif _version_tuple(transformers.__version__) >= (4, 46, 0):
        print(
            f"WARNING: transformers {transformers.__version__} may break OmniGen. "
            f"Try: pip install transformers=={RECOMMENDED_TRANSFORMERS}"
        )

    return OmniGenPipeline.from_pretrained(model_name_or_path)


def generate_one(pipe, prompt, height, width, guidance_scale, seed):
    images = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    return images[0]


def main():
    ap = argparse.ArgumentParser(description="Generate one OmniGen image from a single prompt.")
    ap.add_argument("prompt", help="text prompt to generate")
    ap.add_argument("--output", default=None, help="full output image path")
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--filename", default="output.png")
    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance_scale", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=None, help="GPU id to expose as cuda:0")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output = args.output or os.path.join(args.output_dir, args.filename)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print(f"  model={args.model_name_or_path}")
    print(f"  size={args.width}x{args.height}  guidance={args.guidance_scale}  seed={args.seed}")
    print(f"  output -> {output}")

    print(f"Loading {args.model_name_or_path} ...")
    pipe = load_pipeline(args.model_name_or_path)
    start = time.time()
    image = generate_one(
        pipe,
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    image.save(output)
    print(f"Saved {output} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
