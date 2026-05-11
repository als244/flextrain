"""Gemma 3 HF <-> FlexTrain mapping.

Gemma 3 = Gemma 2 + per-head QK-norm. The post-load hook still does the
+1 shift on RMSNorm γ values (Gemma 3 inherits the convention). The
QK-norm weights are saved as ``γ - 1`` too.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _gemma3_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Shift every Gemma RMSNorm weight by +1 (Gemma's γ convention)."""
    for L in range(num_layers):
        for n in (
            "w_pre_attn_norm", "w_post_attn_norm",
            "w_pre_ffn_norm", "w_post_ffn_norm",
            "w_q_norm", "w_k_norm",
        ):
            t = dest.get((f"layer_{L}", n))
            if t is not None:
                t.add_(1.0)
    final = dest.get(("head", "w_final_norm"))
    if final is not None:
        final.add_(1.0)


# 4B / 12B (Gemma3ForConditionalGeneration) save text weights under
# ``language_model.model.*`` instead of plain ``model.*``. One ArchSpec
# handles both via ``hf_name_alternates``; the loader picks whichever
# prefix is present in the shards.
def _alt(name: str) -> tuple[str, ...]:
    return (f"language_model.{name}",)


GEMMA3_ARCH = ArchSpec(
    hf_arch_ids=("Gemma3ForCausalLM", "Gemma3ForConditionalGeneration"),
    embed=(
        WeightMapEntry(
            flextrain_name="w_tok_embeddings",
            hf_name="model.embed_tokens.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.embed_tokens.weight"),
        ),
    ),
    head=(
        WeightMapEntry(
            flextrain_name="w_final_norm",
            hf_name="model.norm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.norm.weight"),
        ),
    ),
    layer=(
        WeightMapEntry(
            flextrain_name="w_pre_attn_norm",
            hf_name="model.layers.{i}.input_layernorm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.layers.{i}.input_layernorm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_post_attn_norm",
            hf_name="model.layers.{i}.post_attention_layernorm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.layers.{i}.post_attention_layernorm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_q_norm",
            hf_name="model.layers.{i}.self_attn.q_norm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.layers.{i}.self_attn.q_norm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_k_norm",
            hf_name="model.layers.{i}.self_attn.k_norm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.layers.{i}.self_attn.k_norm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_q",
            hf_name="model.layers.{i}.self_attn.q_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.layers.{i}.self_attn.q_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_k",
            hf_name="model.layers.{i}.self_attn.k_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.layers.{i}.self_attn.k_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_v",
            hf_name="model.layers.{i}.self_attn.v_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.layers.{i}.self_attn.v_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_o",
            hf_name="model.layers.{i}.self_attn.o_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.layers.{i}.self_attn.o_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_pre_ffn_norm",
            hf_name="model.layers.{i}.pre_feedforward_layernorm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.layers.{i}.pre_feedforward_layernorm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_post_ffn_norm",
            hf_name="model.layers.{i}.post_feedforward_layernorm.weight",
            transform=Transform.NONE,
            hf_name_alternates=_alt("model.layers.{i}.post_feedforward_layernorm.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_1",
            hf_name="model.layers.{i}.mlp.gate_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.layers.{i}.mlp.gate_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_2",
            hf_name="model.layers.{i}.mlp.down_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.layers.{i}.mlp.down_proj.weight"),
        ),
        WeightMapEntry(
            flextrain_name="w_3",
            hf_name="model.layers.{i}.mlp.up_proj.weight",
            transform=Transform.TRANSPOSE,
            hf_name_alternates=_alt("model.layers.{i}.mlp.up_proj.weight"),
        ),
    ),
    post_load_hook=_gemma3_post_load_hook,
)

register_arch(GEMMA3_ARCH)


ARCH_NAME = "gemma3"


_REQUIRED_DIMS = (
    "vocab_size", "n_layers", "d_model", "n_heads", "head_dim", "expert_dim",
)
_DEFAULT_DATATYPES = {
    "embed": "bfloat16", "head_proj": "bfloat16", "attn_proj": "bfloat16",
    "expert_proj": "bfloat16", "router": "bfloat16", "norm": "bfloat16",
    "residual": "bfloat16",
}


def expand_dims(dims) -> dict:
    """Gemma3 dims schema = Llama. Local/global pattern lives in
    hyperparams."""
    out = dict(dims)
    missing = [k for k in _REQUIRED_DIMS if k not in out]
    if missing:
        raise KeyError(
            f"gemma3 dims missing required keys: {missing}. "
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
    """Gemma3 defaults: eps=1e-6, rope=1e6 (full) / 10k (local).
    ``sliding_window`` and ``layer_types`` left as ``None`` so the
    block builder falls back to its own alternation default."""
    return {
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "rope_local_base": 10_000.0,
        "attn_logit_softcap": None,
        "final_logit_softcap": None,
        "query_pre_attn_scalar": None,
        "sliding_window": None,
        "layer_types": None,
    }


def _resolve_text_config(hf_config: Any):
    """Return a ``Gemma3TextConfig`` instance regardless of whether
    ``hf_config`` was loaded as a raw JSON dict or as an HF
    ``PretrainedConfig``. For 4B/12B the raw dict's ``text_config`` is
    minimal — HF's class fills in n_heads / n_kv_heads / head_dim /
    rms_norm_eps / rope_parameters / layer_types / etc. defaults that
    ``from_pretrained`` relies on. Round-tripping through the class
    gives us those defaults without hardcoding them here.
    """
    from transformers import Gemma3TextConfig

    if isinstance(hf_config, dict):
        text_dict = hf_config.get("text_config", hf_config)
        return Gemma3TextConfig(**text_dict)
    text_cfg = getattr(hf_config, "text_config", None) or hf_config
    if isinstance(text_cfg, Gemma3TextConfig):
        return text_cfg
    # Some HF configs return a plain ``PretrainedConfig`` for the text
    # branch when the parent isn't the wrapper class. Round-trip through
    # the dict form to normalize.
    return Gemma3TextConfig(**text_cfg.to_dict())


def hf_config_to_flextrain(hf_config: Any) -> dict:
    cfg = _resolve_text_config(hf_config)
    n_heads = cfg.num_attention_heads
    n_kv = cfg.num_key_value_heads
    hidden = cfg.hidden_size
    head_dim = cfg.head_dim
    return {
        "vocab_size": cfg.vocab_size,
        "n_layers": cfg.num_hidden_layers,
        "d_model": hidden,
        "n_heads": n_heads,
        "n_kv_heads": n_kv,
        "head_dim": head_dim,
        "expert_dim": cfg.intermediate_size,
        "num_shared_experts": 1,
        "num_routed_experts": 0,
        "top_k": 0,
        "is_causal": True,
        "datatypes": {
            "embed": "bfloat16", "head_proj": "bfloat16",
            "attn_proj": "bfloat16", "expert_proj": "bfloat16",
            "router": "bfloat16", "norm": "bfloat16", "residual": "bfloat16",
        },
    }


def hf_config_to_hyperparams(hf_config: Any) -> dict:
    text_cfg = _resolve_text_config(hf_config)
    get = lambda k, default=None: getattr(text_cfg, k, default)
    rope_params = get("rope_parameters") or {}
    hidden = int(text_cfg.hidden_size)
    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-6),
        # Gemma multiplies input embeddings by sqrt(hidden_size) before
        # the first decoder layer. HF: ``inputs_embeds * d**0.5``.
        "embed_scale": hidden ** 0.5,
        "rope_theta": (
            rope_params.get("full_attention", {}).get("rope_theta")
            or get("rope_theta", 1_000_000.0)
        ),
        "rope_local_base": (
            rope_params.get("sliding_attention", {}).get("rope_theta")
            or get("rope_local_base_freq", 10_000.0)
        ),
        # Raw HF ``rope_parameters`` dict (one entry per ``layer_type``
        # with ``rope_type``, ``rope_theta``, and ``factor`` for linear
        # scaling). The block builder consumes this to derive the
        # per-layer ``rope_scaling`` kwarg — 4B/12B set ``rope_type:
        # linear, factor: 8.0`` on full-attention layers only.
        "rope_parameters": dict(rope_params) if rope_params else {},
        "attn_logit_softcap": get("attn_logit_softcapping"),
        "final_logit_softcap": get("final_logit_softcapping"),
        "query_pre_attn_scalar": get("query_pre_attn_scalar"),
        "sliding_window": get("sliding_window"),
        "tie_word_embeddings": get("tie_word_embeddings", True),
        "layer_types": get("layer_types"),
    }


def _gemma3_block_builder(layer_idx: int, ctx) -> object:
    """Build one :class:`Gemma3Block` with per-layer rope base /
    rope_scaling derived from ``layer_types`` + ``rope_parameters``.

    Sliding layers use ``rope_local_base`` (vanilla scaling); full
    layers use ``rope_theta`` and pick up the optional linear scaling
    (4B/12B factor=8.0). Layer type list comes from HF's
    ``text_config.layer_types`` (when set); otherwise we synthesize the
    "5 sliding then 1 full" pattern that Gemma 3 uses by default.
    """
    from flextrain.nn.layers.gemma3 import Gemma3Block, Gemma3BlockConfig

    dims = ctx.dims
    hp = ctx.hyperparams

    layer_types = hp.get("layer_types") or []
    if layer_idx < len(layer_types):
        lt = layer_types[layer_idx]
    else:
        # Default Gemma 3 pattern: every ``sliding_window_pattern``-th
        # layer is full attention. 5 sliding → 1 full repeating.
        period = int(hp.get("sliding_window_pattern", 6))
        lt = "full_attention" if ((layer_idx + 1) % period == 0) else "sliding_attention"
    is_sliding = (lt == "sliding_attention")

    rope_params = hp.get("rope_parameters") or {}
    layer_rope = (
        rope_params.get("sliding_attention", {})
        if is_sliding
        else rope_params.get("full_attention", {})
    )
    if is_sliding:
        rope_base = float(layer_rope.get("rope_theta", hp.get("rope_local_base", 10_000.0)))
    else:
        rope_base = float(layer_rope.get("rope_theta", hp.get("rope_theta", 1_000_000.0)))

    rope_scaling = None
    rtype = layer_rope.get("rope_type", "default")
    if rtype == "linear":
        rope_scaling = {
            "rope_type": "linear",
            "factor": float(layer_rope.get("factor", 1.0)),
        }
    elif rtype not in ("default", None):
        # llama3 / yarn etc. — flow through to rope.py which will warn
        # and fall back to vanilla. Keep the field so the layer config
        # carries it for diagnostics.
        rope_scaling = dict(layer_rope)

    block_cfg = Gemma3BlockConfig(
        d_model=int(dims["d_model"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=int(dims["n_kv_heads"]),
        head_dim=int(dims["head_dim"]),
        expert_dim=int(dims["expert_dim"]),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-6)),
        rope_base=rope_base,
        rope_scaling=rope_scaling,
        is_causal=True,
        attn_logit_softcap=hp.get("attn_logit_softcap"),
        final_logit_softcap=hp.get("final_logit_softcap"),
        query_pre_attn_scalar=hp.get("query_pre_attn_scalar"),
        window_size_left=int(hp.get("sliding_window", 0)) if is_sliding else -1,
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )
    base = Gemma3Block(layer_id=layer_idx, cfg=block_cfg)
    if not ctx.lora_targets:
        return base
    from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
    return LoRAWrapperLayer(
        base, lora_targets=ctx.lora_targets,
        rank=ctx.lora_rank, alpha=ctx.lora_alpha,
        dims=dict(dims, attn_dim=int(dims["n_heads"]) * int(dims["head_dim"]),
                  kv_dim=int(dims["n_kv_heads"]) * int(dims["head_dim"])),
        adapter_compute_dtype=ctx.lora_adapter_compute_dtype,
        adapter_master_dtype=ctx.lora_adapter_master_dtype,
        adapter_grad_dtype=ctx.lora_adapter_grad_dtype,
        adapter_opt_state_dtype=ctx.lora_adapter_opt_state_dtype,
    )


def post_load_permute(am, hf_config, dims, hyperparams):
    """Gemma 3 post-load fixups:

    1. Halved → pair-interleave permute on the OUT dim of ``w_q`` and
       ``w_k`` (the flextrain RoPE kernel expects pair-interleave).
    2. Same permute on the (head_dim,)-vector ``w_q_norm`` /
       ``w_k_norm`` since QK-norm γ indexes the post-permute channel.
    3. Tied LM head: Gemma 3 ties to the embedding (no separate
       ``lm_head.weight`` in the safetensor). Mirror ``embed.t()`` into
       ``w_head_proj`` when the head was loaded zeros.
    """
    import torch

    head_dim = int(dims["head_dim"])
    n_heads = int(dims["n_heads"])
    n_kv = int(dims["n_kv_heads"])
    n_layers = int(dims["n_layers"])
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    def _halved_to_pair_perm(dim: int, head_dim: int) -> torch.Tensor:
        half = head_dim // 2
        out = torch.empty(dim, dtype=torch.int64)
        for h in range(dim // head_dim):
            base = h * head_dim
            for i in range(half):
                out[base + 2 * i] = base + i
                out[base + 2 * i + 1] = base + half + i
        return out

    q_perm = _halved_to_pair_perm(attn_dim, head_dim)
    k_perm = _halved_to_pair_perm(kv_dim, head_dim)
    # Per-head γ vector for QK-norm: length head_dim, shared across
    # heads. Same halved → pair permute as above.
    head_perm = _halved_to_pair_perm(head_dim, head_dim)

    for i in range(n_layers):
        host = am.buffers.host_params[i]
        if "w_q" in host:
            host["w_q"].copy_(host["w_q"][:, q_perm])
        if "w_k" in host:
            host["w_k"].copy_(host["w_k"][:, k_perm])
        if "w_q_norm" in host:
            host["w_q_norm"].copy_(host["w_q_norm"][head_perm])
        if "w_k_norm" in host:
            host["w_k_norm"].copy_(host["w_k_norm"][head_perm])
        # LoRA-B mirrors if present.
        for name, perm in (("w_q_lora_b", q_perm), ("w_k_lora_b", k_perm)):
            if name in host and host[name].dim() == 2:
                host[name].copy_(host[name][:, perm])

    # Tied LM head (Gemma 3 always ties; check hyperparam too).
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
        ("Gemma3ForCausalLM", "Gemma3ForConditionalGeneration"),
        _gemma3_block_builder,
    )


BLOCK_BUILDER = _gemma3_block_builder

_register_builder()
