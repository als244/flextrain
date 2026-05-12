"""Gemma 4 HF <-> flextrain mapping (text-only path).

Covers `Gemma4ForConditionalGeneration` (the only public 31B / 26B-A4B
shape today) and `Gemma4ForCausalLM` (HF declares it for inheritance).

Deltas vs Gemma 3:

* HF safetensor prefix is ``model.language_model.layers.{i}.*`` (vs
  Gemma 3 4B/12B's ``language_model.model.layers.{i}.*`` — different
  wrapping order). Primary names use the Gemma-4 prefix; alternates
  cover the bare ``model.*`` form that a hypothetical
  ``Gemma4ForCausalLM`` export would produce.
* No ``w_v`` on full-attention layers (``attention_k_eq_v=true``). The
  entry is ``optional=True``; the param spec for those layers does not
  declare ``w_v`` either, so the loader's "hf name absent" early-skip
  takes care of both sides.
* No ``w_v_norm`` — Gemma 4's V-RMSNorm has ``with_scale=False`` (no γ).
* ``layer_scalar`` per-layer is loaded via ``post_load_permute`` directly
  from the safetensors (it's a buffer, not in ParamSpec) and stored on
  the ``Gemma4Block`` instance via ``set_layer_scalar``.
* ``post_load_permute`` does the halved → pair-interleave permute on
  only the rotated channels of Q/K (proportional partial rope on global
  layers; full rope on sliding layers).
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _gemma4_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Gemma 4 RMSNorm convention deltas vs Gemma 2 / 3.

    Gemma 4 changed the RMSNorm forward to multiply by ``weight`` directly
    (no ``1 + weight`` trick) with ``weight`` init to **ones** instead of
    zeros (``modeling_gemma4.Gemma4RMSNorm``). The safetensor therefore
    stores canonical γ directly. **No +1 shift is needed at load time** —
    flextrain's RMSNorm kernel already multiplies by γ.

    This hook is intentionally a no-op for Gemma 4. It's kept for
    symmetry with the Gemma 2 / 3 arch loaders and to leave a clear
    landing site for any future γ-side corrections.
    """
    del hf_path, dest, num_layers


# Multi-prefix loader fallback: the 31B / 26B-A4B safetensors live
# under ``model.language_model.*``. A bare ``Gemma4ForCausalLM`` export
# (if it ever existed) would use ``model.*`` directly. The loader tries
# the primary first, falls back to the alternate.
def _alt(name: str) -> tuple[str, ...]:
    return (name.replace("model.language_model.", "model.", 1),)


GEMMA4_ARCH = ArchSpec(
    hf_arch_ids=("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"),
    embed=(
        WeightMapEntry(
            flextrain_name="w_tok_embeddings",
            hf_name="model.language_model.embed_tokens.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.language_model.embed_tokens.weight"),
        ),
    ),
    head=(
        WeightMapEntry(
            flextrain_name="w_final_norm",
            hf_name="model.language_model.norm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.language_model.norm.weight"),
        ),
    ),
    layer=(
        WeightMapEntry(
            flextrain_name="w_pre_attn_norm",
            hf_name="model.language_model.layers.{i}.input_layernorm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.input_layernorm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_post_attn_norm",
            hf_name="model.language_model.layers.{i}.post_attention_layernorm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.post_attention_layernorm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_q_norm",
            hf_name="model.language_model.layers.{i}.self_attn.q_norm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.self_attn.q_norm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_k_norm",
            hf_name="model.language_model.layers.{i}.self_attn.k_norm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.self_attn.k_norm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_q",
            hf_name="model.language_model.layers.{i}.self_attn.q_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.self_attn.q_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_k",
            hf_name="model.language_model.layers.{i}.self_attn.k_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.self_attn.k_proj.weight"),
        ),
        # ``w_v`` is OPTIONAL: global (full_attention) layers have
        # ``attention_k_eq_v=True`` ⇒ no v_proj in the safetensor and
        # no ``w_v`` slot in the layer's param spec. Sliding layers
        # have w_v as usual.
        WeightMapEntry(
            flextrain_name="w_v",
            hf_name="model.language_model.layers.{i}.self_attn.v_proj.weight",
            transform=Transform.TRANSPOSE,
            optional=True,
            hf_name_alternates=_alt("model.language_model.layers.{i}.self_attn.v_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_o",
            hf_name="model.language_model.layers.{i}.self_attn.o_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.self_attn.o_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_pre_ffn_norm",
            hf_name="model.language_model.layers.{i}.pre_feedforward_layernorm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.pre_feedforward_layernorm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_post_ffn_norm",
            hf_name="model.language_model.layers.{i}.post_feedforward_layernorm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.post_feedforward_layernorm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_1",
            hf_name="model.language_model.layers.{i}.mlp.gate_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.mlp.gate_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_2",
            hf_name="model.language_model.layers.{i}.mlp.down_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.mlp.down_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_3",
            hf_name="model.language_model.layers.{i}.mlp.up_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.language_model.layers.{i}.mlp.up_proj.weight"),
        ),
    ),
    post_load_hook=_gemma4_post_load_hook,
)

register_arch(GEMMA4_ARCH)


ARCH_NAME = "gemma4"


_REQUIRED_DIMS = (
    "vocab_size", "n_layers", "d_model", "n_heads", "head_dim", "expert_dim",
)
_DEFAULT_DATATYPES = {
    "embed": "bfloat16", "head_proj": "bfloat16", "attn_proj": "bfloat16",
    "expert_proj": "bfloat16", "router": "bfloat16", "norm": "bfloat16",
    "residual": "bfloat16",
}


def expand_dims(dims) -> dict:
    """Gemma 4 dims schema = Gemma 3 + global_head_dim + n_global_kv_heads.

    Per-layer-type head shapes live in hyperparams; the dims dict
    carries the SLIDING values (most-common layer type) so the
    backbone factory has a base to ``dataclasses.replace`` from.
    """
    out = dict(dims)
    missing = [k for k in _REQUIRED_DIMS if k not in out]
    if missing:
        raise KeyError(
            f"gemma4 dims missing required keys: {missing}. "
            f"Got keys: {sorted(out)}"
        )
    out.setdefault("n_kv_heads", out["n_heads"])
    out.setdefault("num_shared_experts", 1)
    out.setdefault("num_routed_experts", 0)
    out.setdefault("top_k", 0)
    out.setdefault("is_causal", True)
    out.setdefault("datatypes", dict(_DEFAULT_DATATYPES))
    out["attn_dim"] = int(out["n_heads"]) * int(out["head_dim"])
    out["kv_dim"] = int(out["n_kv_heads"]) * int(out["head_dim"])
    return out


def default_hyperparams() -> dict:
    """Gemma 4 defaults: eps=1e-6, sliding theta=10k / global theta=1e6,
    proportional partial-rope on global with prf=0.25, K=V on global.

    ``layer_types`` and ``sliding_window`` come from the HF config
    (no default here — the backbone factory needs the per-layer schedule).
    """
    return {
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "rope_local_base": 10_000.0,
        "attn_logit_softcap": None,
        "final_logit_softcap": 30.0,
        "query_pre_attn_scalar": None,
        "sliding_window": None,
        "layer_types": None,
        "global_head_dim": None,
        "num_global_key_value_heads": None,
        "attention_k_eq_v": True,
        "global_partial_rotary_factor": 0.25,
    }


def _resolve_text_config(hf_config: Any):
    """Return a ``Gemma4TextConfig`` instance regardless of whether
    ``hf_config`` was loaded as a raw JSON dict or as an HF
    ``PretrainedConfig``. Mirrors ``flextrain.io.arch.gemma3._resolve_text_config``.
    """
    from transformers import Gemma4TextConfig

    if isinstance(hf_config, dict):
        text_dict = hf_config.get("text_config", hf_config)
        return Gemma4TextConfig(**text_dict)
    text_cfg = getattr(hf_config, "text_config", None) or hf_config
    if isinstance(text_cfg, Gemma4TextConfig):
        return text_cfg
    return Gemma4TextConfig(**text_cfg.to_dict())


def hf_config_to_flextrain(hf_config: Any) -> dict:
    """Top-level dims dict consumed by the engine constructor.

    The dims dict carries the SLIDING-layer head shapes — the
    backbone factory uses ``layer_types`` from hyperparams to switch
    to global head shapes on full-attention layers.
    """
    cfg = _resolve_text_config(hf_config)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(cfg.num_key_value_heads)
    hidden = int(cfg.hidden_size)
    head_dim = int(cfg.head_dim)
    return {
        "vocab_size": int(cfg.vocab_size),
        "n_layers": int(cfg.num_hidden_layers),
        "d_model": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "expert_dim": int(cfg.intermediate_size),
        "num_shared_experts": 1,
        "num_routed_experts": 0,
        "top_k": 0,
        "is_causal": True,
        "datatypes": dict(_DEFAULT_DATATYPES),
    }


def hf_config_to_hyperparams(hf_config: Any) -> dict:
    text_cfg = _resolve_text_config(hf_config)
    get = lambda k, default=None: getattr(text_cfg, k, default)

    rope_params = get("rope_parameters") or {}
    full_rope = rope_params.get("full_attention", {}) or {}
    sliding_rope = rope_params.get("sliding_attention", {}) or {}

    return {
        "rms_norm_eps": float(get("rms_norm_eps", 1e-6)),
        # Gemma scales token embeddings by sqrt(hidden_size) before the
        # first decoder layer (same as Gemma 2/3).
        "embed_scale": int(get("hidden_size")) ** 0.5,
        # Per-layer-type rope bases.
        "rope_theta": float(
            full_rope.get("rope_theta")
            or get("rope_theta", 1_000_000.0)
        ),
        "rope_local_base": float(
            sliding_rope.get("rope_theta")
            or get("rope_local_base_freq", 10_000.0)
        ),
        "rope_parameters": dict(rope_params) if rope_params else {},
        "attn_logit_softcap": get("attn_logit_softcapping"),
        "final_logit_softcap": get("final_logit_softcapping"),
        "query_pre_attn_scalar": get("query_pre_attn_scalar"),
        "sliding_window": int(get("sliding_window")),
        "tie_word_embeddings": bool(get("tie_word_embeddings", True)),
        "layer_types": list(get("layer_types")),
        # Gemma 4 globals: doubled head_dim, fewer KV heads, k_eq_v, partial rope.
        "global_head_dim": int(get("global_head_dim")),
        "num_global_key_value_heads": int(get("num_global_key_value_heads")),
        "attention_k_eq_v": bool(get("attention_k_eq_v", True)),
        "global_partial_rotary_factor": float(
            full_rope.get("partial_rotary_factor", 0.25)
        ),
    }


def _gemma4_block_builder(layer_idx: int, ctx) -> object:
    """Build one :class:`Gemma4Block` with per-layer head_dim /
    n_kv_heads / rope_base / partial_rotary_factor / k_eq_v selected
    by the ``layer_types`` schedule.

    Per-layer choices:
    * ``sliding_attention``: head_dim=cfg.head_dim, n_kv_heads=cfg.n_kv_heads,
      rope_base=rope_local_base, prf=1.0, k_eq_v=False, window_size_left=sliding_window.
    * ``full_attention``: head_dim=global_head_dim, n_kv_heads=num_global_key_value_heads,
      rope_base=rope_theta (=1e6), prf=global_partial_rotary_factor (=0.25),
      k_eq_v=attention_k_eq_v (=True), window_size_left=-1.
    """
    from flextrain.nn.layers.gemma4 import Gemma4Block, Gemma4BlockConfig

    dims = ctx.dims
    hp = ctx.hyperparams

    layer_types = hp.get("layer_types") or []
    if not layer_types:
        raise ValueError(
            "gemma4 block builder requires layer_types in hyperparams"
        )
    lt = layer_types[layer_idx]
    is_sliding = (lt == "sliding_attention")

    if is_sliding:
        head_dim = int(dims["head_dim"])
        n_kv_heads = int(dims["n_kv_heads"])
        rope_base = float(hp.get("rope_local_base", 10_000.0))
        rope_scaling = None
        prf = 1.0
        k_eq_v = False
        window = int(hp.get("sliding_window") or 0)
    elif lt == "full_attention":
        head_dim = int(hp["global_head_dim"])
        n_kv_heads = int(hp["num_global_key_value_heads"])
        rope_base = float(hp.get("rope_theta", 1_000_000.0))
        rope_scaling = {"rope_type": "proportional"}
        prf = float(hp.get("global_partial_rotary_factor", 0.25))
        k_eq_v = bool(hp.get("attention_k_eq_v", True))
        window = -1
    else:
        raise ValueError(f"gemma4: unknown layer_type {lt!r} at layer {layer_idx}")

    block_cfg = Gemma4BlockConfig(
        d_model=int(dims["d_model"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        expert_dim=int(dims["expert_dim"]),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-6)),
        rope_base=rope_base,
        rope_scaling=rope_scaling,
        is_causal=True,
        attn_logit_softcap=hp.get("attn_logit_softcap"),
        final_logit_softcap=hp.get("final_logit_softcap"),
        query_pre_attn_scalar=hp.get("query_pre_attn_scalar"),
        window_size_left=window,
        v_norm=True,
        k_eq_v=k_eq_v,
        partial_rotary_factor=prf,
        layer_scalar_init=1.0,
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )
    base = Gemma4Block(layer_id=layer_idx, cfg=block_cfg)
    if not ctx.lora_targets:
        return base
    from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
    return LoRAWrapperLayer(
        base, lora_targets=ctx.lora_targets,
        rank=ctx.lora_rank, alpha=ctx.lora_alpha,
        dims=dict(dims, attn_dim=int(dims["n_heads"]) * head_dim,
                  kv_dim=n_kv_heads * head_dim),
        adapter_compute_dtype=ctx.lora_adapter_compute_dtype,
        adapter_master_dtype=ctx.lora_adapter_master_dtype,
        adapter_grad_dtype=ctx.lora_adapter_grad_dtype,
        adapter_opt_state_dtype=ctx.lora_adapter_opt_state_dtype,
    )


# ---------------------------------------------------------------------------
# post_load_permute
# ---------------------------------------------------------------------------


def _partial_halved_to_pair_perm(
    head_dim: int, rope_angles: int,
) -> torch.Tensor:
    """Per-head permutation from HF halved-rope layout to flextrain's
    pair-interleave layout, rotating only the first ``2*rope_angles``
    channels. The remaining (head_dim - 2*rope_angles) channels are
    placed after the rotated prefix in their natural HF order.

    For full rope (rope_angles = head_dim/2) this reduces exactly to
    :func:`flextrain.io.arch.gemma3._halved_to_pair_perm` per-head.
    """
    half = head_dim // 2
    out = torch.empty(head_dim, dtype=torch.int64)
    # Rotated prefix: FT[2i] = HF[i], FT[2i+1] = HF[half + i].
    for i in range(rope_angles):
        out[2 * i] = i
        out[2 * i + 1] = half + i
    # Non-rotated suffix: HF positions [rope_angles, half) ∪
    # [half + rope_angles, head_dim) in natural order.
    suffix = 2 * rope_angles
    for i in range(rope_angles, half):
        out[suffix] = i
        suffix += 1
    for i in range(half + rope_angles, head_dim):
        out[suffix] = i
        suffix += 1
    assert suffix == head_dim
    return out


def _multi_head_perm(
    dim: int, head_dim: int, rope_angles: int,
) -> torch.Tensor:
    """Apply the per-head permute across an out-dim spanning multiple heads
    (Q's attn_dim or K's kv_dim)."""
    head_perm = _partial_halved_to_pair_perm(head_dim, rope_angles)
    out = torch.empty(dim, dtype=torch.int64)
    for h in range(dim // head_dim):
        base = h * head_dim
        out[base : base + head_dim] = head_perm + base
    return out


def post_load_permute(am, hf_config, dims, hyperparams):
    """Gemma 4 post-load fixups:

    1. Per-layer Q/K halved → pair-interleave permute, rotating only
       the channels covered by RoPE on that layer-type:
       * Sliding layers: full head_dim (rope_angles = head_dim/2).
       * Global layers: partial (rope_angles = int(prf × global_head_dim // 2)).
    2. Same per-head permute on the ``w_q_norm`` / ``w_k_norm`` γ vectors
       (length head_dim, shared across heads).
    3. Load ``layer_scalar`` (per-layer scalar buffer) from the
       safetensors into each ``Gemma4Block.set_layer_scalar``. The
       Instruct checkpoint has non-trivial values (e.g. 0.55, 0.68);
       defaulting to 1.0 would silently miscompute.
    4. Tied LM head: mirror ``w_tok_embeddings.t()`` into ``w_head_proj``
       when the head was loaded as zeros.
    """
    import os
    import json

    layer_types = hyperparams.get("layer_types") or []
    n_layers = int(dims["n_layers"])
    if not layer_types or len(layer_types) != n_layers:
        raise ValueError(
            "gemma4 post_load_permute requires hyperparams['layer_types'] "
            f"of length {n_layers}; got {len(layer_types)}"
        )
    n_heads = int(dims["n_heads"])
    sliding_head_dim = int(dims["head_dim"])
    sliding_n_kv = int(dims["n_kv_heads"])
    global_head_dim = int(hyperparams["global_head_dim"])
    global_n_kv = int(hyperparams["num_global_key_value_heads"])
    global_prf = float(hyperparams.get("global_partial_rotary_factor", 0.25))

    # Precompute permutations.
    sliding_rope_angles = sliding_head_dim // 2  # full rope
    global_rope_angles = int(global_prf * global_head_dim // 2)

    sliding_q_perm = _multi_head_perm(
        n_heads * sliding_head_dim, sliding_head_dim, sliding_rope_angles,
    )
    sliding_k_perm = _multi_head_perm(
        sliding_n_kv * sliding_head_dim, sliding_head_dim, sliding_rope_angles,
    )
    sliding_head_perm = _partial_halved_to_pair_perm(
        sliding_head_dim, sliding_rope_angles,
    )

    global_q_perm = _multi_head_perm(
        n_heads * global_head_dim, global_head_dim, global_rope_angles,
    )
    global_k_perm = _multi_head_perm(
        global_n_kv * global_head_dim, global_head_dim, global_rope_angles,
    )
    global_head_perm = _partial_halved_to_pair_perm(
        global_head_dim, global_rope_angles,
    )

    for i in range(n_layers):
        host = am.buffers.host_params[i]
        is_sliding = layer_types[i] == "sliding_attention"
        q_perm = sliding_q_perm if is_sliding else global_q_perm
        k_perm = sliding_k_perm if is_sliding else global_k_perm
        head_perm = sliding_head_perm if is_sliding else global_head_perm

        if "w_q" in host:
            host["w_q"].copy_(host["w_q"][:, q_perm])
        if "w_k" in host:
            host["w_k"].copy_(host["w_k"][:, k_perm])
        if "w_q_norm" in host:
            host["w_q_norm"].copy_(host["w_q_norm"][head_perm])
        if "w_k_norm" in host:
            host["w_k_norm"].copy_(host["w_k_norm"][head_perm])
        # LoRA-B mirrors (when LoRA is wrapping the block).
        for name, perm in (("w_q_lora_b", q_perm), ("w_k_lora_b", k_perm)):
            if name in host and host[name].dim() == 2:
                host[name].copy_(host[name][:, perm])

    # --- Load layer_scalar from the safetensors (it's a per-layer buffer,
    # not in ParamSpec, so the standard loader doesn't see it).
    hf_source = getattr(am, "_hf_source_path", None)
    if hf_source is not None and os.path.isdir(hf_source):
        from safetensors import safe_open

        # Build index of which shard each layer_scalar lives in.
        index_path = os.path.join(hf_source, "model.safetensors.index.json")
        weight_map: dict[str, str] = {}
        if os.path.isfile(index_path):
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {}) or {}

        def _scalar_key(L: int) -> str | None:
            for candidate in (
                f"model.language_model.layers.{L}.layer_scalar",
                f"model.layers.{L}.layer_scalar",
            ):
                if weight_map and candidate in weight_map:
                    return candidate
                if not weight_map:
                    # Without an index we'll need to probe each shard.
                    return candidate
            return None

        # If no index, scan every shard to find layer_scalar entries.
        if not weight_map:
            import glob
            shards = sorted(glob.glob(os.path.join(hf_source, "*.safetensors")))
            shard_keys: dict[str, set[str]] = {}
            for sp in shards:
                with safe_open(sp, framework="pt") as f:
                    shard_keys[sp] = set(f.keys())
        else:
            shards = None
            shard_keys = None

        for L in range(n_layers):
            block = am.backbone[L]
            inner = getattr(block, "base", block)  # unwrap LoRA wrapper
            if not hasattr(inner, "set_layer_scalar"):
                continue

            candidates = (
                f"model.language_model.layers.{L}.layer_scalar",
                f"model.layers.{L}.layer_scalar",
            )
            scalar_value: float | None = None
            for key in candidates:
                shard_path: str | None = None
                if weight_map and key in weight_map:
                    shard_path = os.path.join(hf_source, weight_map[key])
                elif shard_keys is not None:
                    for sp, keys in shard_keys.items():
                        if key in keys:
                            shard_path = sp
                            break
                if shard_path is None:
                    continue
                with safe_open(shard_path, framework="pt") as f:
                    t = f.get_tensor(key)
                    scalar_value = float(t.reshape(-1)[0].item())
                break
            if scalar_value is not None:
                inner.set_layer_scalar(scalar_value)

    # --- Tied LM head: mirror embed.t() into w_head_proj when the
    # head was loaded as zeros (no separate lm_head.weight in the
    # safetensor, which is the case for Gemma 4 since tie_word_embeddings=True).
    tied = hyperparams.get("tie_word_embeddings", True)
    if tied:
        head_w = am.buffers.host_head_params.get("w_head_proj")
        embed_w = am.buffers.host_embed_params.get("w_tok_embeddings")
        if (
            head_w is not None and embed_w is not None
            and float(head_w.abs().sum().item()) == 0.0
        ):
            head_w.copy_(embed_w.t())

    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        if name in am.buffers.host_head_params:
            dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(
        ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"),
        _gemma4_block_builder,
    )


BLOCK_BUILDER = _gemma4_block_builder

_register_builder()
