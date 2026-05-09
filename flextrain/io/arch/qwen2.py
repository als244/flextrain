"""Qwen2 / Qwen2.5 HF <-> FlexTrain mapping.

Covers the Qwen2 family (Qwen2-0.5B through Qwen2.5-72B, dense
variants). Key differences from Llama:

* **Q/K/V biases** on the attention projections
  (``attention_bias=True`` in HF config). Our
  :class:`GQAAttentionBlock` with ``cfg.qkv_bias=True`` reads
  ``b_q`` / ``b_k`` / ``b_v`` bias vectors (shape ``(attn_dim,)`` /
  ``(kv_dim,)`` / ``(kv_dim,)``).
* **No QK-norm** (unlike Qwen3).
* **RoPE base** is 1e6 (like Qwen3; Llama3 uses 5e5).
* **Default rms_norm_eps** 1e-6 (like Qwen3).
* **Some small variants have tied embeddings** (``lm_head.weight``
  absent from safetensors).
"""

from __future__ import annotations

from typing import Any

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


QWEN2_ARCH = ArchSpec(
    hf_arch_ids=("Qwen2ForCausalLM",),
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
        WeightMapEntry(
            flextrain_name="w_head_proj",
            hf_name="lm_head.weight",
            transform=Transform.TRANSPOSE,
        ),
    ),
    layer=(
        WeightMapEntry(
            flextrain_name="w_attn_norm",
            hf_name="model.layers.{i}.input_layernorm.weight",
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
        # Qwen2-specific: Q/K/V biases. 1-D, no transpose.
        WeightMapEntry(
            flextrain_name="b_q",
            hf_name="model.layers.{i}.self_attn.q_proj.bias",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="b_k",
            hf_name="model.layers.{i}.self_attn.k_proj.bias",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="b_v",
            hf_name="model.layers.{i}.self_attn.v_proj.bias",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_o",
            hf_name="model.layers.{i}.self_attn.o_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            flextrain_name="w_ffn_norm",
            hf_name="model.layers.{i}.post_attention_layernorm.weight",
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
)

register_arch(QWEN2_ARCH)


ARCH_NAME = "qwen2"


_REQUIRED_DIMS = (
    "vocab_size", "n_layers", "d_model", "n_heads", "head_dim", "expert_dim",
)
_DEFAULT_DATATYPES = {
    "embed": "bfloat16", "head_proj": "bfloat16", "attn_proj": "bfloat16",
    "expert_proj": "bfloat16", "router": "bfloat16", "norm": "bfloat16",
    "residual": "bfloat16",
}


def expand_dims(dims) -> dict:
    """Qwen2 dims schema = Llama + ``qkv_bias`` (defaults to True —
    Qwen2 uses biased Q/K/V projections)."""
    out = dict(dims)
    missing = [k for k in _REQUIRED_DIMS if k not in out]
    if missing:
        raise KeyError(
            f"qwen2 dims missing required keys: {missing}. "
            f"Got keys: {sorted(out)}"
        )
    out.setdefault("n_kv_heads", out["n_heads"])
    out.setdefault("num_shared_experts", 1)
    out.setdefault("num_routed_experts", 0)
    out.setdefault("top_k", 0)
    out.setdefault("is_causal", True)
    out.setdefault("qkv_bias", True)
    out.setdefault("datatypes", dict(_DEFAULT_DATATYPES))
    out["attn_dim"] = int(out["n_heads"]) * int(out["head_dim"])
    out["kv_dim"] = int(out["n_kv_heads"]) * int(out["head_dim"])
    return out


def default_hyperparams() -> dict:
    """Qwen2 defaults: eps=1e-6, rope=1e6, full attention."""
    return {
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "rope_scaling": None,
        "window_size_left": -1,
        "window_size_right": 0,
    }


def hf_config_to_flextrain(hf_config: Any) -> dict:
    """Qwen2 ``config.json`` → FlexTrain dims dict."""
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    return {
        "vocab_size": get("vocab_size"),
        "n_layers": get("num_hidden_layers"),
        "d_model": get("hidden_size"),
        "n_heads": get("num_attention_heads"),
        "n_kv_heads": get("num_key_value_heads") or get("num_attention_heads"),
        "head_dim": get("head_dim")
        or (get("hidden_size") // get("num_attention_heads")),
        "expert_dim": get("intermediate_size"),
        "num_shared_experts": 1,
        "num_routed_experts": 0,
        "top_k": 0,
        "is_causal": True,
        "qkv_bias": bool(get("attention_bias", True)),
        "datatypes": {
            "embed": "bfloat16",
            "head_proj": "bfloat16",
            "attn_proj": "bfloat16",
            "expert_proj": "bfloat16",
            "router": "bfloat16",
            "norm": "bfloat16",
            "residual": "bfloat16",
        },
    }


def hf_config_to_hyperparams(hf_config: Any) -> dict:
    """Per-layer hyperparams. Qwen2 defaults: eps=1e-6, rope_theta=1e6."""
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-6),
        "rope_theta": get("rope_theta", 1_000_000.0),
        "rope_scaling": get("rope_scaling"),
        "window_size_left": -1,
        "window_size_right": 0,
    }


# ---------------------------------------------------------------------------
# Block builder for ``flextrain.from_pretrained``.
# ---------------------------------------------------------------------------


def _qwen2_block_builder(layer_idx: int, ctx) -> object:
    from flextrain.nn.layers.qwen2 import Qwen2Block, Qwen2BlockConfig

    dims = ctx.dims
    hp = ctx.hyperparams
    block_cfg = Qwen2BlockConfig(
        d_model=int(dims["d_model"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=int(dims["n_kv_heads"]),
        head_dim=int(dims["head_dim"]),
        expert_dim=int(dims["expert_dim"]),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-6)),
        rope_base=float(hp.get("rope_theta", 1_000_000.0)),
        rope_scaling=hp.get("rope_scaling"),
        is_causal=True,
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )
    base = Qwen2Block(layer_id=layer_idx, cfg=block_cfg)
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


# Qwen2 needs the Llama Q/K halved→pair perm AND the same perm applied
# to the Q/K bias vectors (since each bias entry is added to a specific
# head_dim slot post-projection — the bias must follow the head_dim
# permutation).
def post_load_permute(am, hf_config, dims, hyperparams):
    import torch
    from flextrain.io.arch.llama import post_load_permute as _llama_perm

    # Run the Llama-style perm first (handles w_q, w_k, w_q_lora_b,
    # w_k_lora_b, tied embeddings, GPU refresh).
    _llama_perm(am, hf_config, dims, hyperparams)

    # Then permute Q/K bias vectors along their single axis.
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
        if "b_q" in host:
            host["b_q"].copy_(host["b_q"][q_perm])
        if "b_k" in host:
            host["b_k"].copy_(host["b_k"][k_perm])

    # Re-mirror bias values to GPU slots if any are resident.
    am._refresh_gpu_residents()
    torch.cuda.synchronize()


def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(("Qwen2ForCausalLM",), _qwen2_block_builder)


BLOCK_BUILDER = _qwen2_block_builder


_register_builder()
