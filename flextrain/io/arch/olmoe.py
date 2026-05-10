"""OLMoE-1B-7B HF <-> FlexTrain mapping.

OLMoE (AllenAI) is a dense-attention + MoE-FFN architecture:
* GQA attention (but n_heads == n_kv_heads in OLMoE-1B-7B, so effectively MHA).
* No QK-norm, no attention biases.
* MoE SwiGLU FFN: 64 experts, top-k=8, intermediate_size=1024.
* RoPE base 10_000 (Llama-2 style, not Llama-3), no rope_scaling.
* No tied embeddings (the small 1B variant has distinct embed and lm_head).
* Load-balance aux loss coef = 0.01.

HF MoE weight layout vs FlexTrain's stacked ``w_up (E, 2*F, d)`` /
``w_down (E, d, F)`` (post option-B migration):

* HF:   ``model.layers.{L}.mlp.gate.weight``           — (E, d)    router
        ``model.layers.{L}.mlp.experts.{E}.gate_proj`` — (F, d)    per-expert gate (x1)
        ``model.layers.{L}.mlp.experts.{E}.up_proj``   — (F, d)    per-expert value (x3)
        ``model.layers.{L}.mlp.experts.{E}.down_proj`` — (d, F)    per-expert down
* FT:   ``w_router (d, E)``  — HF gate transposed.
        ``w_up (E, 2F, d)``  — per expert the packed layout along dim=0 of
                              the (2F, d) slice is ``[x3, x1]`` (value
                              first, gate second). Same orientation as HF
                              gate_proj/up_proj — just stacked.
        ``w_down (E, d, F)`` — per expert: same as HF down_proj.

The ``post_load_hook`` iterates experts, reads the four per-expert HF
tensors, transposes+stacks them into FT's layout, and writes into
``dest[("layer_{L}", "w_up")]`` / ``dest[("layer_{L}", "w_down")]``.
The ArchSpec proper handles the non-MoE parts via regular
``WeightMapEntry`` declarations.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import torch
from safetensors import safe_open

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _olmoe_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Stack per-expert HF tensors into FT's ``w_up``/``w_down``."""
    # Determine num_experts from the shape of w_up in dest (first layer).
    sample_w_up = dest.get(("layer_0", "w_up"))
    if sample_w_up is None:
        # Model has no MoE layers declared — nothing to do.
        return
    # FT layout (post option-B migration): w_up is (E, 2F, D), w_down (E, D, F).
    E, TwoF, D = sample_w_up.shape
    F = TwoF // 2
    sample_w_down = dest[("layer_0", "w_down")]
    assert sample_w_down.shape == (E, D, F), (
        f"w_down shape mismatch: expected ({E}, {D}, {F}), "
        f"got {tuple(sample_w_down.shape)}"
    )

    # Build index of shards (single or multi-file).
    idx_path = os.path.join(hf_path, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        import json
        with open(idx_path) as f:
            file_index = json.load(f)["weight_map"]
    else:
        # Single shard: every tensor is in model.safetensors.
        single = os.path.join(hf_path, "model.safetensors")
        if not os.path.isfile(single):
            raise FileNotFoundError(
                f"No safetensors index at {idx_path} and no single file"
            )
        file_index = None

    def _shard_for(name: str) -> str:
        if file_index is not None:
            return os.path.join(hf_path, file_index[name])
        return os.path.join(hf_path, "model.safetensors")

    # Open each shard once and pull every expert tensor that lives in
    # it. The naive (L × E × 3) loop with a fresh ``safe_open`` per
    # tensor sends ~18k metadata opens through the filesystem; fine on
    # local NVMe, pathological on GPFS / NFS / Lustre.
    wanted_by_shard: dict[str, list[tuple[int, int, str, str]]] = {}
    for L in range(num_layers):
        for e in range(E):
            for kind in ("gate", "up", "down"):
                hf_name = f"model.layers.{L}.mlp.experts.{e}.{kind}_proj.weight"
                wanted_by_shard.setdefault(_shard_for(hf_name), []).append(
                    (L, e, kind, hf_name)
                )

    pending: dict[tuple[int, int], dict[str, torch.Tensor]] = {}

    def _flush_if_ready(L: int, e: int) -> None:
        slot = pending.get((L, e))
        if slot is None or not all(k in slot for k in ("gate", "up", "down")):
            return
        w_up_ft = dest[(f"layer_{L}", "w_up")]
        w_down_ft = dest[(f"layer_{L}", "w_down")]
        dtype = w_up_ft.dtype
        # HF: gate/up are (F, D), down is (D, F).
        # FT w_up[e]: (2F, D) packed [x3, x1] = [up; gate] cat along dim=0.
        # FT w_down[e]: (D, F) — same orientation as HF.
        w_up_ft[e, :, :].copy_(
            torch.cat([slot["up"].to(dtype), slot["gate"].to(dtype)], dim=0)
        )
        w_down_ft[e, :, :].copy_(slot["down"].to(dtype))
        del pending[(L, e)]

    for shard_path, wanted in wanted_by_shard.items():
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for L, e, kind, hf_name in wanted:
                pending.setdefault((L, e), {})[kind] = f.get_tensor(hf_name)
                _flush_if_ready(L, e)

    if pending:
        sample_keys = list(pending.keys())[:3]
        raise RuntimeError(
            f"olmoe loader: {len(pending)} (layer, expert) entries "
            f"never received all of (gate, up, down). Sample: {sample_keys}"
        )


def _olmoe_pre_export_hook(am, dst, num_layers: int) -> None:
    """Inverse of ``_olmoe_post_load_hook``: emit per-expert HF tensors
    from FT's stacked ``w_up`` / ``w_down``. OLMoE has only the
    per-expert format (no fused gate_up_proj).
    """
    from flextrain.export._pre_export_helpers import (
        emit_routed_experts,
        read_tie_word_embeddings,
    )
    for L in range(num_layers):
        host = am.buffers.host_params[L]
        w_up = host.get("w_up")
        w_down = host.get("w_down")
        if w_up is None or w_down is None:
            continue
        emit_routed_experts(
            dst,
            layer_prefix=f"model.layers.{L}",
            w_up=w_up,
            w_down=w_down,
            fused=False,
        )
    if read_tie_word_embeddings(am):
        dst.pop("lm_head.weight", None)


OLMOE_ARCH = ArchSpec(
    hf_arch_ids=("OlmoeForCausalLM",),
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
        WeightMapEntry(
            flextrain_name="w_o",
            hf_name="model.layers.{i}.self_attn.o_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        # OLMoE: full-dim RMSNorm on Q and K between projection and RoPE.
        # Weight shapes: (attn_dim,) and (kv_dim,) respectively.
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
            flextrain_name="w_ffn_norm",
            hf_name="model.layers.{i}.post_attention_layernorm.weight",
            transform=Transform.NONE,
        ),
        # MoE router (gate). HF stores as (E, d); FT uses (d, E).
        WeightMapEntry(
            flextrain_name="w_router",
            hf_name="model.layers.{i}.mlp.gate.weight",
            transform=Transform.TRANSPOSE,
        ),
        # NOTE: w_up and w_down are stacked per-expert by the
        # post_load_hook below. No ArchSpec entry — the engine
        # allocates empty tensors in dest; the hook populates them.
    ),
    post_load_hook=_olmoe_post_load_hook,
    pre_export_hook=_olmoe_pre_export_hook,
)

register_arch(OLMOE_ARCH)


ARCH_NAME = "olmoe"


_REQUIRED_DIMS = (
    "vocab_size", "n_layers", "d_model", "n_heads", "head_dim", "expert_dim",
    "num_routed_experts", "top_k",
)
_DEFAULT_DATATYPES = {
    "embed": "bfloat16", "head_proj": "bfloat16", "attn_proj": "bfloat16",
    "expert_proj": "bfloat16", "router": "bfloat16", "norm": "bfloat16",
    "residual": "bfloat16",
}


def expand_dims(dims) -> dict:
    """OLMoE dims schema = Llama + ``num_routed_experts`` + ``top_k``,
    no shared expert. ``expert_dim`` is the per-expert FFN intermediate
    size (HF ``intermediate_size``)."""
    out = dict(dims)
    missing = [k for k in _REQUIRED_DIMS if k not in out]
    if missing:
        raise KeyError(
            f"olmoe dims missing required keys: {missing}. "
            f"Got keys: {sorted(out)}"
        )
    out.setdefault("n_kv_heads", out["n_heads"])
    out.setdefault("num_shared_experts", 0)
    out.setdefault("is_causal", True)
    out.setdefault("datatypes", dict(_DEFAULT_DATATYPES))
    out["attn_dim"] = int(out["n_heads"]) * int(out["head_dim"])
    out["kv_dim"] = int(out["n_kv_heads"]) * int(out["head_dim"])
    return out


def default_hyperparams() -> dict:
    """OLMoE defaults: eps=1e-5, rope=10k, full attention. Aux-loss
    coef 0.01 and ``softmax_then_topk`` routing match the published
    OLMoE-7B-A1B checkpoint (``norm_topk_prob=False``)."""
    return {
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "rope_scaling": None,
        "window_size_left": -1,
        "window_size_right": 0,
        "load_balance_coef": 0.01,
        "routing_mode": "softmax_then_topk",
    }


def flextrain_to_hf_config(dims, hyperparams=None) -> dict:
    """Inverse mapping for OLMoE. ``intermediate_size`` is the per-expert
    FFN dim (HF stores it under ``intermediate_size`` for OLMoE — same
    field name as dense models, no separate ``moe_intermediate_size``)."""
    hp = dict(default_hyperparams())
    if hyperparams:
        hp.update(hyperparams)
    d = expand_dims(dims)
    norm_topk = (hp.get("routing_mode") == "topk_then_softmax")
    return {
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "vocab_size": int(d["vocab_size"]),
        "num_hidden_layers": int(d["n_layers"]),
        "hidden_size": int(d["d_model"]),
        "num_attention_heads": int(d["n_heads"]),
        "num_key_value_heads": int(d["n_kv_heads"]),
        "head_dim": int(d["head_dim"]),
        "intermediate_size": int(d["expert_dim"]),
        "num_experts": int(d["num_routed_experts"]),
        "num_experts_per_tok": int(d["top_k"]),
        "norm_topk_prob": norm_topk,
        "router_aux_loss_coef": float(hp.get("load_balance_coef", 0.01)),
        "rms_norm_eps": float(hp["rms_norm_eps"]),
        "rope_theta": float(hp["rope_theta"]),
        "rope_scaling": hp.get("rope_scaling"),
        "max_position_embeddings": 4096,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "initializer_range": 0.02,
        "torch_dtype": "bfloat16",
        "use_cache": True,
    }


def hf_config_to_flextrain(hf_config: Any) -> dict:
    """OLMoE ``config.json`` → FlexTrain dims dict."""
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
        "num_shared_experts": 0,
        "num_routed_experts": get("num_experts"),
        "top_k": get("num_experts_per_tok"),
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
    """Per-layer hyperparams. OLMoE defaults: eps=1e-5, rope=10000."""
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    norm_topk = bool(get("norm_topk_prob", False))
    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-5),
        "rope_theta": get("rope_theta", 10_000.0),
        "rope_scaling": get("rope_scaling"),
        "window_size_left": -1,
        "window_size_right": 0,
        "load_balance_coef": get("router_aux_loss_coef", 0.01),
        "routing_mode": "topk_then_softmax" if norm_topk else "softmax_then_topk",
    }


# ---------------------------------------------------------------------------
# Block builder + post-load permutation hook for ``flextrain.from_pretrained``.
# ---------------------------------------------------------------------------


def _olmoe_block_builder(layer_idx: int, ctx) -> object:
    from flextrain.nn.layers.olmoe import OLMoEBlock, OLMoEBlockConfig

    dims = ctx.dims
    hp = ctx.hyperparams
    block_cfg = OLMoEBlockConfig(
        d_model=int(dims["d_model"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=int(dims["n_kv_heads"]),
        head_dim=int(dims["head_dim"]),
        expert_dim=int(dims["expert_dim"]),
        num_experts=int(dims["num_routed_experts"]),
        top_k=int(dims["top_k"]),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-5)),
        rope_base=float(hp.get("rope_theta", 10_000.0)),
        load_balance_coef=float(hp.get("load_balance_coef", 0.01)),
        routing_mode=str(hp.get("routing_mode", "softmax_then_topk")),
        is_causal=True,
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )
    base = OLMoEBlock(
        layer_id=layer_idx, cfg=block_cfg,
        expert_compute=getattr(ctx, "moe_backend", None),
    )
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
    """OLMoE post-load: same Q/K halved→pair perm as Llama, plus the
    full-row QK-norm 1-D vectors get the same perm along their single axis."""
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
        # Full-row QK-norm vectors share the perm along their 1-D axis.
        if "w_q_norm" in host:
            host["w_q_norm"].copy_(host["w_q_norm"][q_perm])
        if "w_k_norm" in host:
            host["w_k_norm"].copy_(host["w_k_norm"][k_perm])
        # LoRA on Q/K mirror to column dim.
        for name, perm in (("w_q_lora_b", q_perm), ("w_k_lora_b", k_perm)):
            if name in host and host[name].dim() == 2:
                host[name].copy_(host[name][:, perm])

    # OLMoE has no tied embeddings (separate lm_head).
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        if name in am.buffers.host_head_params:
            dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(("OlmoeForCausalLM",), _olmoe_block_builder)


BLOCK_BUILDER = _olmoe_block_builder


_register_builder()
