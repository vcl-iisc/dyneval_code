#!/usr/bin/env python3
"""Generate missing OmniGen images for the DYNEVAL-1K set.

Reads OmniGen-MISSING-PROMPTS.json (534 prompts: 129 GenEval + 405 EvalMuse).
Already have: 466 DPG-Bench. Missing: 534.

Saves:
    <output_dir>/<gid:04d>.png   e.g. 0001.png, 0026.png

Copy to main machine:
    DYNEVAL-1K-IMAGES/OmniGen/

Usage (from DYNEVAL project root; paths below are relative to cwd):

    python dyneval_code/generator_scripts/omnigen/image-infer.py --dry-run
    python dyneval_code/generator_scripts/omnigen/image-infer.py --limit 5
    python dyneval_code/generator_scripts/omnigen/image-infer.py --gpus 0,1,2,3

Remote server:
    python dyneval_code/generator_scripts/omnigen/image-infer.py \\
        --batch_json OmniGen-MISSING-PROMPTS.json \\
        --output_dir ./omnigen_outputs \\
        --model_name_or_path Shitao/OmniGen-v1 \\
        --gpus 0,1,2,3

If every prompt fails with "cannot unpack non-iterable NoneType object", your
transformers build is too new for OmniGen's Phi3 fork. Either:
    pip install transformers==4.45.2 peft==0.17.1 diffusers==0.30.3
or rerun with this script (it auto-patches transformers>=4.46).

"tuple index out of range" was a follow-on bug in the first patch (fixed now):
newer Phi3DecoderLayer no longer returns past_key_value in layer_outputs.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback

DEFAULT_JSON = "OmniGen-MISSING-PROMPTS.json"
DEFAULT_OUT = os.path.join("DYNEVAL-1K-IMAGES", "OmniGen")
DEFAULT_MODEL = "./OMNIGEN-MODEL"
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


def load_prompts(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("prompts", data)
    return sorted(prompts, key=lambda p: int(p["gid"]))


def filter_pending(prompts, output_dir, overwrite):
    if overwrite:
        return prompts
    return [
        p for p in prompts
        if not os.path.exists(os.path.join(output_dir, p["filename"]))
    ]


def shard(items, n_shards, shard_id):
    return items[shard_id::n_shards]


def worker_main(worker_id, gpu_id, pending, cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    tag = f"gpu{gpu_id}"
    manifest = os.path.join(cfg["output_dir"], f"generated_manifest.{tag}.jsonl")
    fail_log = os.path.join(cfg["output_dir"], f"generate_fail.{tag}.log")

    print(f"[{tag}] worker {worker_id}: {len(pending)} prompts")
    if not pending:
        return 0, 0

    t0 = time.time()
    print(f"[{tag}] loading {cfg['model_name_or_path']} ...")
    pipe = load_pipeline(cfg["model_name_or_path"])
    print(f"[{tag}] model ready in {time.time() - t0:.1f}s")

    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        t1 = time.time()
        try:
            generate_one(
                pipe,
                prompt=prompt,
                height=cfg["height"],
                width=cfg["width"],
                guidance_scale=cfg["guidance_scale"],
                seed=cfg["seed"],
            ).save(dst)
            ok += 1
            elapsed = time.time() - t1
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({
                    "gid": entry["gid"],
                    "filename": filename,
                    "benchmark": entry["benchmark"],
                    "prompt": prompt,
                    "seconds": round(elapsed, 2),
                    "gpu": gpu_id,
                }, ensure_ascii=False) + "\n")
            print(f"[{tag} {i}/{len(pending)}] {filename} OK"
                  f"  gid={entry['gid']}  [{entry['benchmark']}]  {elapsed:.1f}s")
        except Exception as exc:
            bad += 1
            with open(fail_log, "a", encoding="utf-8") as lf:
                lf.write(f"{entry['gid']}\t{filename}\t{prompt}\t{exc}\n")
                lf.write(traceback.format_exc() + "\n")
            print(f"[{tag} {i}/{len(pending)}] {filename} FAILED: {exc}")

    print(f"[{tag}] done: generated={ok} failed={bad}")
    return ok, bad


def run_single_gpu(pending, cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    manifest = os.path.join(cfg["output_dir"], "generated_manifest.jsonl")
    fail_log = os.path.join(cfg["output_dir"], "generate_fail.log")

    if cfg.get("gpu") is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])

    print(f"Loading {cfg['model_name_or_path']} ...")
    pipe = load_pipeline(cfg["model_name_or_path"])

    ok = bad = 0
    for i, entry in enumerate(pending, 1):
        filename = entry["filename"]
        prompt = entry["prompt"]
        dst = os.path.join(cfg["output_dir"], filename)
        t1 = time.time()
        try:
            generate_one(
                pipe,
                prompt=prompt,
                height=cfg["height"],
                width=cfg["width"],
                guidance_scale=cfg["guidance_scale"],
                seed=cfg["seed"],
            ).save(dst)
            ok += 1
            elapsed = time.time() - t1
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({
                    "gid": entry["gid"],
                    "filename": filename,
                    "benchmark": entry["benchmark"],
                    "prompt": prompt,
                    "seconds": round(elapsed, 2),
                }, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(pending)}] {filename} OK"
                  f"  gid={entry['gid']}  [{entry['benchmark']}]  {elapsed:.1f}s")
        except Exception as exc:
            bad += 1
            with open(fail_log, "a", encoding="utf-8") as lf:
                lf.write(f"{entry['gid']}\t{filename}\t{prompt}\t{exc}\n")
                lf.write(traceback.format_exc() + "\n")
            print(f"[{i}/{len(pending)}] {filename} FAILED: {exc}")

    return ok, bad


def parse_gpus(gpu_str):
    if not gpu_str:
        return None
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip() != ""]


def build_cfg(args, gpu=None):
    return {
        "model_name_or_path": args.model_name_or_path,
        "output_dir": args.output_dir,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "gpu": gpu,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_json", default=DEFAULT_JSON)
    ap.add_argument("--output_dir", default=DEFAULT_OUT)
    ap.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--guidance_scale", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids, e.g. 0,1,2,3")
    args = ap.parse_args()

    prompts = load_prompts(args.batch_json)
    pending = filter_pending(prompts, args.output_dir, args.overwrite)

    print(f"[OmniGen] json={len(prompts)}  already done={len(prompts) - len(pending)}"
          f"  to generate={len(pending)}")
    print(f"  model={args.model_name_or_path}")
    print(f"  size={args.width}x{args.height}  guidance={args.guidance_scale}  seed={args.seed}")
    print(f"  output -> {args.output_dir}")

    if args.limit:
        pending = pending[: args.limit]
        print(f"  limited to {len(pending)} this run")

    if not pending:
        print("Nothing to do.")
        return

    if args.dry_run:
        gpus = parse_gpus(args.gpus) or [0]
        for rank, gpu in enumerate(gpus):
            part = shard(pending, len(gpus), rank)
            print(f"  GPU {gpu}: {len(part)} prompts")
        for p in pending[:10]:
            print(f"  gid={p['gid']}  {p['filename']}  [{p['benchmark']}]  {p['prompt'][:80]}")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return

    gpus = parse_gpus(args.gpus)
    if gpus and len(gpus) > 1:
        cfg = build_cfg(args)
        ctx = mp.get_context("spawn")
        jobs = [
            (rank, gpu, shard(pending, len(gpus), rank), cfg)
            for rank, gpu in enumerate(gpus)
            if shard(pending, len(gpus), rank)
        ]
        print(f"Launching {len(jobs)} GPU workers on {gpus} ...")
        with ctx.Pool(len(jobs)) as pool:
            results = pool.starmap(worker_main, jobs)
        ok = sum(r[0] for r in results)
        bad = sum(r[1] for r in results)
    else:
        gpu = gpus[0] if gpus else None
        ok, bad = run_single_gpu(pending, build_cfg(args, gpu=gpu))

    print(f"\nDone OmniGen: generated={ok}  failed={bad}  -> {args.output_dir}")
    if bad:
        print("Check generate_fail*.log in output_dir")


if __name__ == "__main__":
    main()
