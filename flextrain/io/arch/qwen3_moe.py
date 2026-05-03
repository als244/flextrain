"""Qwen3-MoE HF <-> FlexTrain mapping.

Covers the MoE variants (Qwen3-30B-A3B, Qwen3-235B-A22B, etc.). Shares
attention with Qwen3-dense (GQA + per-head QK-norm) but replaces the
dense SwiGLU FFN with a stacked-expert MoE FFN.

HF expert layout: ``model.layers.{L}.mlp.experts.{e}.{gate,up,down}_proj.weight``
with shapes (F, d), (F, d), (d, F) — identical to OLMoE. FlexTrain
stacks these into ``w_up (E, 2F, d)`` (packed ``[up; gate]`` = ``[x3, x1]``
along the 2F axis) and ``w_down (E, d, F)``. Router is ``w_router (d, E)``
(HF's ``mlp.gate.weight`` transposed).
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

import torch
from safetensors import safe_open

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _qwen3_moe_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Stack per-expert HF tensors into FT's ``w_up``/``w_down``. The
    packing order is ``[up, gate]`` so the orig swiglu_moe kernel
    reads ``x3`` (value) from the first half and ``x1`` (gate) from
    the second half.

    Open each shard exactly once and pull every expert tensor that
    lives in it. The naive (L × E × 3) loop with a fresh ``safe_open``
    per tensor sends ~18k metadata opens through the filesystem, which
    is fine on local NVMe but pathological on GPFS / NFS / Lustre
    where every open is an RPC.
    """
    sample_w_up = dest.get(("layer_0", "w_up"))
    if sample_w_up is None:
        return
    # FT layout (post option-B migration): w_up is (E, 2F, D), w_down (E, D, F).
    E, TwoF, D = sample_w_up.shape
    F = TwoF // 2
    sample_w_down = dest[("layer_0", "w_down")]
    assert sample_w_down.shape == (E, D, F), (
        f"w_down shape mismatch: expected ({E}, {D}, {F}), "
        f"got {tuple(sample_w_down.shape)}"
    )

    idx_path = os.path.join(hf_path, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            file_index = json.load(f)["weight_map"]
    else:
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
        # FT w_up[e]: (2F, D) with [up; gate] cat along dim=0 = [x3, x1].
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
            f"qwen3_moe loader: {len(pending)} (layer, expert) entries "
            f"never received all of (gate, up, down). Sample: {sample_keys}"
        )


def _qwen3_moe_pre_export_hook(am, dst, num_layers: int) -> None:
    """Inverse of ``_qwen3_moe_post_load_hook``: emit per-expert HF
    tensors from FT's stacked ``w_up`` / ``w_down``.

    Qwen3-MoE has only the per-expert format on disk (no fused
    ``gate_up_proj``); we always emit per-expert
    ``mlp.experts.{e}.{gate,up,down}_proj.weight``. The ArchSpec walk
    didn't emit ``w_up`` / ``w_down`` (no entries in ``layer``), so
    nothing to drop.

    Tied embeddings: when the source config has
    ``tie_word_embeddings: True``, drop ``lm_head.weight`` from the
    export — HF will re-mirror at load time.
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
            continue  # not a MoE layer (defensive — qwen3_moe always is)
        emit_routed_experts(
            dst,
            layer_prefix=f"model.layers.{L}",
            w_up=w_up,
            w_down=w_down,
            fused=False,
        )
    if read_tie_word_embeddings(am):
        dst.pop("lm_head.weight", None)


QWEN3_MOE_ARCH = ArchSpec(
    hf_arch_ids=("Qwen3MoeForCausalLM",),
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
            flextrain_name="w_router",
            hf_name="model.layers.{i}.mlp.gate.weight",
            transform=Transform.TRANSPOSE,
        ),
        # w_up / w_down populated by post_load_hook.
    ),
    post_load_hook=_qwen3_moe_post_load_hook,
    pre_export_hook=_qwen3_moe_pre_export_hook,
)

register_arch(QWEN3_MOE_ARCH)


def hf_config_to_flextrain(hf_config: Any) -> dict:
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    n_heads = get("num_attention_heads")
    hidden = get("hidden_size")
    head_dim = get("head_dim") or (hidden // n_heads)
    # Qwen3-MoE uses ``moe_intermediate_size`` for the per-expert dim.
    expert_dim = get("moe_intermediate_size") or get("intermediate_size")
    return {
        "vocab_size": get("vocab_size"),
        "n_layers": get("num_hidden_layers"),
        "d_model": hidden,
        "n_heads": n_heads,
        "n_kv_heads": get("num_key_value_heads") or n_heads,
        "head_dim": head_dim,
        "expert_dim": expert_dim,
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
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    rope_params = get("rope_parameters") or {}
    rope_theta = rope_params.get("rope_theta") or get("rope_theta", 1_000_000.0)
    norm_topk = bool(get("norm_topk_prob", True))
    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-6),
        "rope_theta": rope_theta,
        "rope_scaling": rope_params.get("rope_scaling") or get("rope_scaling"),
        "window_size_left": -1,
        "window_size_right": 0,
        "load_balance_coef": get("router_aux_loss_coef", 0.001),
        "routing_mode": "topk_then_softmax" if norm_topk else "softmax_then_topk",
    }


def _qwen3_moe_block_builder(layer_idx: int, ctx) -> object:
    from flextrain.nn.layers.qwen3_moe import Qwen3MoEBlock, Qwen3MoEBlockConfig

    dims = ctx.dims
    hp = ctx.hyperparams
    block_cfg = Qwen3MoEBlockConfig(
        d_model=int(dims["d_model"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=int(dims["n_kv_heads"]),
        head_dim=int(dims["head_dim"]),
        expert_dim=int(dims["expert_dim"]),
        num_experts=int(dims["num_routed_experts"]),
        top_k=int(dims["top_k"]),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-6)),
        rope_base=float(hp.get("rope_theta", 1_000_000.0)),
        load_balance_coef=float(hp.get("load_balance_coef", 0.001)),
        routing_mode=str(hp.get("routing_mode", "topk_then_softmax")),
        is_causal=True,
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )
    base = Qwen3MoEBlock(
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
    """Qwen3-MoE: same as Qwen3-dense (Q/K halved→pair + per-head_dim QK-norm)."""
    from flextrain.io.arch.qwen3 import post_load_permute as _qwen3_perm
    _qwen3_perm(am, hf_config, dims, hyperparams)


def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(("Qwen3MoeForCausalLM",), _qwen3_moe_block_builder)


_register_builder()
