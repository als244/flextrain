"""Qwen3 (dense) family HF <-> FlexTrain mapping.

Covers the dense Qwen3 variants (Qwen3-0.6B, -1.7B, -4B, -8B, -14B,
-32B). The MoE variants (Qwen3-MoE, Qwen3-30B-A3B) have a different
FFN layout and need a separate arch spec once ``MoETransformerBlock``
lands.

Differences from Llama:
* **QK-norm**: every attention block has ``self_attn.q_norm.weight``
  and ``self_attn.k_norm.weight`` (RMSNorm applied per-head to Q and
  K *after* their projections, *before* RoPE). Our
  :class:`Qwen3DenseBlock` handles this via the extra
  ``w_q_norm`` / ``w_k_norm`` params.
* **No attention biases** (``attention_bias=False``).
* **Different default rope base** (1e6 vs Llama's 5e5).
* **Different rms_norm_eps** default (1e-6 vs Llama's 1e-5).
* **Some configs have tied embeddings** (small variants); the
  ``w_head_proj`` weight will be missing from the safetensors, same as
  Llama-3.2.
"""

from __future__ import annotations

from typing import Any

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _qwen3_pre_export_hook(am, dst, num_layers: int) -> None:
    """Drop ``lm_head.weight`` from the export when the source had
    ``tie_word_embeddings: True`` (Qwen3-1.7B / 4B). FT mirrors the
    embedding into ``w_head_proj`` at load time; HF re-mirrors at
    the receiving end."""
    from flextrain.export._pre_export_helpers import read_tie_word_embeddings
    if read_tie_word_embeddings(am):
        dst.pop("lm_head.weight", None)


QWEN3_ARCH = ArchSpec(
    hf_arch_ids=("Qwen3ForCausalLM",),
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
            optional=True,
        ),
    ),
    layer=(
        WeightMapEntry(
            flextrain_name="w_attn_norm",
            hf_name="model.layers.{i}.input_layernorm.weight",
            transform=Transform.NONE,
        ),
        # QK-norm weights (Qwen3-specific). No transpose, 1-D.
        WeightMapEntry(
            flextrain_name="w_q_norm",
            hf_name="model.layers.{i}.self_attn.q_norm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_k_norm",
            hf_name="model.layers.{i}.self_attn.k_norm.weight",
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
    pre_export_hook=_qwen3_pre_export_hook,
)

register_arch(QWEN3_ARCH)


ARCH_NAME = "qwen3"


_REQUIRED_DIMS = (
    "vocab_size", "n_layers", "d_model", "n_heads", "head_dim", "expert_dim",
)
_DEFAULT_DATATYPES = {
    "embed": "bfloat16", "head_proj": "bfloat16", "attn_proj": "bfloat16",
    "expert_proj": "bfloat16", "router": "bfloat16", "norm": "bfloat16",
    "residual": "bfloat16",
}


def expand_dims(dims) -> dict:
    """Qwen3 dense dims schema = Llama. QK-norm lives in the block, not
    in dims."""
    out = dict(dims)
    missing = [k for k in _REQUIRED_DIMS if k not in out]
    if missing:
        raise KeyError(
            f"qwen3 dims missing required keys: {missing}. "
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
    """Qwen3 defaults: eps=1e-6, rope=1e6, full attention.
    ``max_window_layers=0`` together with ``window_size_left=-1`` means
    every layer is full-context (the no-sliding-window default).
    """
    return {
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "rope_scaling": None,
        "window_size_left": -1,
        "window_size_right": 0,
        "max_window_layers": 0,
    }


def hf_config_to_flextrain(hf_config: Any) -> dict:
    """Qwen3 dense ``config.json`` → FlexTrain dims dict."""
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
    """Per-layer hyperparams (norm eps, RoPE base). Qwen3 defaults:
    eps=1e-6, rope_theta=1e6."""
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    # Qwen3's config.json nests RoPE under ``rope_parameters`` in
    # newer transformers releases; fall back to top-level
    # ``rope_theta`` for older configs.
    rope_params = get("rope_parameters") or {}
    rope_theta = rope_params.get("rope_theta") or get("rope_theta", 1_000_000.0)
    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-6),
        "rope_theta": rope_theta,
        "rope_scaling": rope_params.get("rope_scaling") or get("rope_scaling"),
        "window_size_left": int(get("sliding_window") or -1),
        "window_size_right": 0,
        # Layers in [0, max_window_layers) use full-context attention;
        # layers >= max_window_layers use sliding window (when enabled).
        # Some Qwen3 configs leave this off, in which case all layers
        # are full-context.
        "max_window_layers": int(get("max_window_layers") or get("num_hidden_layers", 0)),
    }


# ---------------------------------------------------------------------------
# Block builder + post-load permutation hook for ``flextrain.from_pretrained``.
# ---------------------------------------------------------------------------


def _qwen3_block_builder(layer_idx: int, ctx) -> object:
    import torch
    from flextrain.nn.layers.qwen3 import (
        Qwen3DenseBlock, Qwen3DenseBlockConfig,
        Qwen3DenseSWABlock, Qwen3DenseSWABlockConfig,
    )

    dims = ctx.dims
    hp = ctx.hyperparams
    max_full = int(hp.get("max_window_layers") or dims["n_layers"])
    use_swa = layer_idx >= max_full and int(hp.get("window_size_left") or -1) > 0

    if use_swa:
        block_cfg = Qwen3DenseSWABlockConfig(
            d_model=int(dims["d_model"]),
            n_heads=int(dims["n_heads"]),
            n_kv_heads=int(dims["n_kv_heads"]),
            head_dim=int(dims["head_dim"]),
            expert_dim=int(dims["expert_dim"]),
            rms_norm_eps=float(hp.get("rms_norm_eps", 1e-6)),
            rope_base=float(hp.get("rope_theta", 1_000_000.0)),
            window_size_left=int(hp["window_size_left"]),
            is_causal=True,
            compute_dtype=ctx.compute_dtype,
            master_dtype=ctx.master_dtype,
            grad_dtype=ctx.grad_dtype,
            norm_grad_dtype=ctx.norm_grad_dtype,
        )
        base = Qwen3DenseSWABlock(layer_id=layer_idx, cfg=block_cfg)
    else:
        block_cfg = Qwen3DenseBlockConfig(
            d_model=int(dims["d_model"]),
            n_heads=int(dims["n_heads"]),
            n_kv_heads=int(dims["n_kv_heads"]),
            head_dim=int(dims["head_dim"]),
            expert_dim=int(dims["expert_dim"]),
            rms_norm_eps=float(hp.get("rms_norm_eps", 1e-6)),
            rope_base=float(hp.get("rope_theta", 1_000_000.0)),
            is_causal=True,
            compute_dtype=ctx.compute_dtype,
            master_dtype=ctx.master_dtype,
            grad_dtype=ctx.grad_dtype,
            norm_grad_dtype=ctx.norm_grad_dtype,
        )
        base = Qwen3DenseBlock(layer_id=layer_idx, cfg=block_cfg)

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
    """Qwen3 post-load: Q/K halved→pair permutation + per-head-dim
    QK-norm permutation. Also handles tied embeddings."""
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
    norm_perm = _halved_to_pair(head_dim, head_dim)  # per-head_dim norm vec

    for i in range(n_layers):
        host = am.buffers.host_params[i]
        # Q/K weight matrices.
        for name, perm in (("w_q", q_perm), ("w_k", k_perm)):
            if name in host:
                host[name].copy_(host[name][:, perm])
        # QK-norm weight vectors (Qwen3 has per-head_dim QK norm).
        for nm in ("w_q_norm", "w_k_norm"):
            if nm in host:
                w = host[nm]
                # 1-D head_dim vector.
                host[nm].copy_(w[norm_perm])
        # LoRA B mirrors permutation along its column dim if present.
        for name, perm in (("w_q_lora_b", q_perm), ("w_k_lora_b", k_perm)):
            if name in host and host[name].dim() == 2:
                host[name].copy_(host[name][:, perm])

    # Tied embeddings (Qwen3 0.6B / 1.7B sometimes tie).
    tied = (
        getattr(hf_config, "tie_word_embeddings", None)
        if not isinstance(hf_config, dict)
        else hf_config.get("tie_word_embeddings")
    )
    if tied:
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
    register_block_builder(("Qwen3ForCausalLM",), _qwen3_block_builder)


BLOCK_BUILDER = _qwen3_block_builder


_register_builder()
