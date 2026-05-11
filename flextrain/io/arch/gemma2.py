"""Gemma 2 HF <-> FlexTrain mapping.

Differences from Llama:
* Four norms per layer (pre+post for attn, pre+post for FFN).
* RMSNorm weight stored as ``γ - 1`` (so untrained values center at 0,
  not 1). Post-load hook shifts every loaded norm weight by +1 to match
  the standard ``γ`` convention used by our :class:`RMSNormBlock`.
* Attention logit softcap (``attn_logit_softcapping`` in HF config).
* Final logit softcap on LM head (``final_logit_softcapping``).
* Per-layer alternating sliding-window via ``layer_types``.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _gemma2_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Shift every Gemma RMSNorm weight by +1 to match Llama convention.

    Gemma 2 stores ``γ_HF = γ_canonical - 1``; our :class:`RMSNormBlock`
    expects ``γ_canonical``. Apply the shift in-place at load time.
    """
    for L in range(num_layers):
        for n in (
            "w_pre_attn_norm", "w_post_attn_norm",
            "w_pre_ffn_norm", "w_post_ffn_norm",
        ):
            t = dest.get((f"layer_{L}", n))
            if t is not None:
                t.add_(1.0)
    # Final norm.
    final = dest.get(("head", "w_final_norm"))
    if final is not None:
        final.add_(1.0)


GEMMA2_ARCH = ArchSpec(
    hf_arch_ids=("Gemma2ForCausalLM",),
    embed=(
        WeightMapEntry(
            flextrain_name="w_tok_embeddings",
            hf_name="model.embed_tokens.weight",
            transform=Transform.NONE,
        ),
    ),
    head=(
        WeightMapEntry(
            flextrain_name="w_final_norm",
            hf_name="model.norm.weight",
            transform=Transform.NONE,
        ),
        # Gemma 2 ties LM head to the embedding (no separate w_head_proj).
    ),
    layer=(
        WeightMapEntry(
            flextrain_name="w_pre_attn_norm",
            hf_name="model.layers.{i}.input_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_post_attn_norm",
            hf_name="model.layers.{i}.post_attention_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_q",
            hf_name="model.layers.{i}.self_attn.q_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_k",
            hf_name="model.layers.{i}.self_attn.k_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_v",
            hf_name="model.layers.{i}.self_attn.v_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_o",
            hf_name="model.layers.{i}.self_attn.o_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_pre_ffn_norm",
            hf_name="model.layers.{i}.pre_feedforward_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_post_ffn_norm",
            hf_name="model.layers.{i}.post_feedforward_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_1",
            hf_name="model.layers.{i}.mlp.gate_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_2",
            hf_name="model.layers.{i}.mlp.down_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_3",
            hf_name="model.layers.{i}.mlp.up_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
    ),
    post_load_hook=_gemma2_post_load_hook,
)

register_arch(GEMMA2_ARCH)


ARCH_NAME = "gemma2"


_REQUIRED_DIMS = (
    "vocab_size", "n_layers", "d_model", "n_heads", "head_dim", "expert_dim",
)
_DEFAULT_DATATYPES = {
    "embed": "bfloat16", "head_proj": "bfloat16", "attn_proj": "bfloat16",
    "expert_proj": "bfloat16", "router": "bfloat16", "norm": "bfloat16",
    "residual": "bfloat16",
}


def expand_dims(dims) -> dict:
    """Gemma2 dims schema = Llama. The sliding-window-vs-global
    schedule lives in hyperparams (``layer_types``)."""
    out = dict(dims)
    missing = [k for k in _REQUIRED_DIMS if k not in out]
    if missing:
        raise KeyError(
            f"gemma2 dims missing required keys: {missing}. "
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
    """Gemma2 defaults: eps=1e-6, rope=10k, sliding_window=4096, attn
    soft-cap at 50, final logit soft-cap at 30. ``layer_types=None``
    falls back to the alternating-default in the block builder
    (every odd layer sliding)."""
    return {
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "attn_logit_softcap": 50.0,
        "final_logit_softcap": 30.0,
        "query_pre_attn_scalar": None,
        "sliding_window": 4096,
        "layer_types": None,
    }


def flextrain_to_hf_config(dims, hyperparams=None) -> dict:
    """Inverse mapping for Gemma2. Emits ``layer_types`` when set, soft-cap
    fields, and ``query_pre_attn_scalar``."""
    hp = dict(default_hyperparams())
    if hyperparams:
        hp.update(hyperparams)
    d = expand_dims(dims)
    cfg = {
        "architectures": ["Gemma2ForCausalLM"],
        "model_type": "gemma2",
        "vocab_size": int(d["vocab_size"]),
        "num_hidden_layers": int(d["n_layers"]),
        "hidden_size": int(d["d_model"]),
        "num_attention_heads": int(d["n_heads"]),
        "num_key_value_heads": int(d["n_kv_heads"]),
        "head_dim": int(d["head_dim"]),
        "intermediate_size": int(d["expert_dim"]),
        "rms_norm_eps": float(hp["rms_norm_eps"]),
        "rope_theta": float(hp["rope_theta"]),
        "rope_scaling": hp.get("rope_scaling"),
        "max_position_embeddings": 8192,
        "hidden_act": "gelu_pytorch_tanh",
        "tie_word_embeddings": True,
        "initializer_range": 0.02,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "sliding_window": int(hp.get("sliding_window", 4096)),
        "attn_logit_softcapping": (
            float(hp["attn_logit_softcap"])
            if hp.get("attn_logit_softcap") is not None else None
        ),
        "final_logit_softcapping": (
            float(hp["final_logit_softcap"])
            if hp.get("final_logit_softcap") is not None else None
        ),
        "query_pre_attn_scalar": hp.get("query_pre_attn_scalar"),
    }
    if hp.get("layer_types"):
        cfg["layer_types"] = list(hp["layer_types"])
    return cfg


def hf_config_to_flextrain(hf_config: Any) -> dict:
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    n_heads = get("num_attention_heads")
    hidden = get("hidden_size")
    head_dim = get("head_dim") or (hidden // n_heads)
    return {
        "vocab_size": get("vocab_size"),
        "n_layers": get("num_hidden_layers"),
        "d_model": hidden,
        "n_heads": n_heads,
        "n_kv_heads": get("num_key_value_heads") or n_heads,
        "head_dim": head_dim,
        "expert_dim": get("intermediate_size"),
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
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    hidden = int(get("hidden_size"))
    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-6),
        "rope_theta": get("rope_theta", 10_000.0),
        "rope_scaling": get("rope_scaling"),
        "attn_logit_softcap": get("attn_logit_softcapping", 50.0),
        "final_logit_softcap": get("final_logit_softcapping", 30.0),
        "query_pre_attn_scalar": get("query_pre_attn_scalar"),
        "sliding_window": get("sliding_window") or 4096,
        # Gemma multiplies input embeddings by ``sqrt(hidden_size)``
        # before the first decoder layer.
        "embed_scale": hidden ** 0.5,
        # ``layer_types`` is a list of "global_attention" / "sliding_attention"
        # one per layer. Default: alternate (every odd layer is sliding).
        "layer_types": get("layer_types"),
    }


def _gemma2_block_builder(layer_idx: int, ctx) -> object:
    from flextrain.nn.layers.gemma2 import Gemma2Block, Gemma2BlockConfig

    dims = ctx.dims
    hp = ctx.hyperparams

    # Layer-type classification: full (global) vs sliding.
    layer_types = hp.get("layer_types") or []
    if layer_idx < len(layer_types):
        is_sliding = layer_types[layer_idx] == "sliding_attention"
    else:
        # Gemma 2 default: alternate (odd layers are sliding).
        is_sliding = (layer_idx % 2) == 1
    window_size_left = int(hp.get("sliding_window", 4096)) if is_sliding else -1

    block_cfg = Gemma2BlockConfig(
        d_model=int(dims["d_model"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=int(dims["n_kv_heads"]),
        head_dim=int(dims["head_dim"]),
        expert_dim=int(dims["expert_dim"]),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-6)),
        rope_base=float(hp.get("rope_theta", 10_000.0)),
        is_causal=True,
        attn_logit_softcap=float(hp.get("attn_logit_softcap", 50.0)),
        final_logit_softcap=float(hp.get("final_logit_softcap", 30.0)),
        query_pre_attn_scalar=hp.get("query_pre_attn_scalar"),
        window_size_left=window_size_left,
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )
    base = Gemma2Block(layer_id=layer_idx, cfg=block_cfg)
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
    """Gemma 2: same Q/K halved→pair perm as Llama. Tied LM head:
    Gemma 2 ties to the embedding by convention (no separate
    ``lm_head.weight`` in the safetensors), so mirror embed.t() into
    head."""
    import torch

    head_dim = int(dims["head_dim"])
    n_heads = int(dims["n_heads"])
    n_kv = int(dims["n_kv_heads"])
    n_layers = int(dims["n_layers"])
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    def _halved_to_pair(dim: int, head_dim: int) -> torch.Tensor:
        half = head_dim // 2
        out = torch.empty(dim, dtype=torch.int64)
        for h in range(dim // head_dim):
            base = h * head_dim
            for i in range(half):
                out[base + 2 * i] = base + i
                out[base + 2 * i + 1] = base + half + i
        return out

    q_perm = _halved_to_pair(attn_dim, head_dim)
    k_perm = _halved_to_pair(kv_dim, head_dim)

    for i in range(n_layers):
        host = am.buffers.host_params[i]
        for name, perm in (("w_q", q_perm), ("w_k", k_perm)):
            if name in host:
                host[name].copy_(host[name][:, perm])
        for name, perm in (("w_q_lora_b", q_perm), ("w_k_lora_b", k_perm)):
            if name in host and host[name].dim() == 2:
                host[name].copy_(host[name][:, perm])

    # Gemma 2 ties LM head to embed.
    head_w = am.buffers.host_head_params.get("w_head_proj")
    embed_w = am.buffers.host_embed_params.get("w_tok_embeddings")
    if (head_w is not None and embed_w is not None
            and float(head_w.abs().sum().item()) == 0.0):
        head_w.copy_(embed_w.t())

    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        if name in am.buffers.host_head_params:
            dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(("Gemma2ForCausalLM",), _gemma2_block_builder)


BLOCK_BUILDER = _gemma2_block_builder


_register_builder()
