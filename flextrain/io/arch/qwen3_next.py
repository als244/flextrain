"""Qwen3-Next HF <-> FlexTrain mapping.

Qwen3-Next has alternating linear-attention and full-attention layers
plus a Qwen3-MoE-style sparse FFN. The HF tensor names are different
per layer-type:

* linear-attention layers: ``model.layers.{L}.linear_attn.*``
  (in_proj_qkvz, in_proj_ba, conv1d, A_log, dt_bias, norm, out_proj)
* full-attention layers: ``model.layers.{L}.self_attn.*``
  (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm)

Both layer types share:
* ``model.layers.{L}.input_layernorm.weight`` (pre-attn norm)
* ``model.layers.{L}.post_attention_layernorm.weight`` (pre-ffn norm)
* MoE FFN at ``model.layers.{L}.mlp.*`` (gate router + per-expert
  gate_proj/up_proj/down_proj — stacked by the post_load_hook).

The arch spec declares the union of all weight-map entries; the loader
runs with ``strict=False`` so per-layer-type entries that don't exist
in HF (linear-attn entries on full-attn layers, etc.) are silently
skipped. The destination ``dest`` mapping only has slots for tensors
the layer's ParamSpec actually declared, so misses are benign.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

import torch
from safetensors import safe_open

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _qwen3_next_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Stack per-expert HF tensors into FT's ``w_up`` / ``w_down``,
    AND apply Qwen3-Next's ``(1 + weight)`` RMSNorm convention by
    shifting every loaded RMSNorm γ by +1.

    HF Qwen3NextRMSNorm stores ``γ_canonical - 1`` (init.zeros_) and
    forwards as ``x.normed * (1 + weight)``. FT's RMSNormBlock expects
    ``γ_canonical`` directly, so we apply the +1 shift in this hook.

    Same convention as Gemma 2 — see ``_gemma2_post_load_hook``.
    """
    # ----- (1) RMSNorm γ shift: HF stores γ_canonical - 1 for the
    # NON-GATED norms (Qwen3NextRMSNorm uses ``init.zeros_`` + forward
    # ``output * (1 + weight)``). The GATED norm (Qwen3NextRMSNormGated,
    # used as ``linear_attn.norm`` aka ``w_lin_norm``) uses
    # ``init.ones_`` + plain ``weight * x`` — store γ_canonical directly
    # and must NOT be shifted.
    _norm_field_names = (
        "w_attn_norm",          # input_layernorm  (Qwen3NextRMSNorm)
        "w_ffn_norm",           # post_attention_layernorm
        "w_q_norm",             # full-attn per-head q_norm
        "w_k_norm",             # full-attn per-head k_norm
    )
    for L in range(num_layers):
        for n in _norm_field_names:
            t = dest.get((f"layer_{L}", n))
            if t is not None:
                t.add_(1.0)
    final_norm = dest.get(("head", "w_final_norm"))
    if final_norm is not None:
        final_norm.add_(1.0)

    # ----- (2) Stack per-expert MoE weights. -----
    # Two HF formats observed:
    #   * fused (newer):    experts.gate_up_proj  (E, 2F, d_model)
    #                       experts.down_proj      (E, d_model, F)
    #   * per-expert (older): experts.{e}.{gate,up,down}_proj.weight
    #
    # The fused format is what real Qwen3-Next / Qwen3.5-MoE / Qwen3.6-MoE
    # ship. The per-expert format is kept as fallback for older saves.
    #
    # ALSO supports HF's optional ``model.language_model.layers.{L}.*``
    # prefix (Qwen3.5/3.6 multimodal saves) — try both top-level
    # ``model.layers.*`` and nested ``model.language_model.layers.*``.

    sample_w_up = dest.get(("layer_0", "w_up"))
    if sample_w_up is None:
        return  # No MoE in this model (dense only).
    E, D, TwoF = sample_w_up.shape
    Fmid = TwoF // 2  # avoid shadowing torch.nn.functional alias as ``F``

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

    def _open_for(name: str):
        if file_index is not None:
            shard = file_index.get(name)
            if shard is None:
                return None
            return os.path.join(hf_path, shard)
        return os.path.join(hf_path, "model.safetensors")

    def _resolve_layer_prefix(layer_idx: int) -> str | None:
        """HF uses ``model.layers.{L}.*`` for plain text-only saves and
        ``model.language_model.layers.{L}.*`` for multimodal saves
        (Qwen3.5/3.6). Try both; return whichever has a matching key."""
        for prefix in (
            f"model.language_model.layers.{layer_idx}",
            f"model.layers.{layer_idx}",
        ):
            # Probe for any expected MLP key under this prefix.
            for probe in (
                f"{prefix}.mlp.experts.gate_up_proj",
                f"{prefix}.mlp.experts.0.gate_proj.weight",
            ):
                if (file_index and probe in file_index) or (
                    file_index is None and _open_for(probe) is not None
                ):
                    return prefix
        return None

    for L in range(num_layers):
        if dest.get((f"layer_{L}", "w_up")) is None:
            continue
        w_up_ft = dest[(f"layer_{L}", "w_up")]
        w_down_ft = dest[(f"layer_{L}", "w_down")]
        dtype = w_up_ft.dtype

        prefix = _resolve_layer_prefix(L)
        if prefix is None:
            continue  # Skip layers with no MoE weights.

        # Fused format first (real Qwen3-Next/3.5/3.6).
        fused_gate_up_name = f"{prefix}.mlp.experts.gate_up_proj"
        fused_down_name    = f"{prefix}.mlp.experts.down_proj"
        if (file_index and fused_gate_up_name in file_index) or (
            file_index is None and _open_for(fused_gate_up_name) is not None
        ):
            with safe_open(_open_for(fused_gate_up_name), framework="pt", device="cpu") as f:
                hf_gate_up = f.get_tensor(fused_gate_up_name)   # (E, 2F, D)
            with safe_open(_open_for(fused_down_name), framework="pt", device="cpu") as f:
                hf_down = f.get_tensor(fused_down_name)         # (E, D, F)
            # HF: gate_up_proj[e] is (2F, D). After F.linear(x, gate_up_proj[e]):
            #   y = x @ gate_up_proj[e].T   → (T, 2F),   chunk → (gate, up).
            # So along dim=1 (the 2F dim), gate is FIRST half, up SECOND half.
            gate_part = hf_gate_up[:, :Fmid, :]                  # (E, F, D)
            up_part   = hf_gate_up[:, Fmid:, :]                  # (E, F, D)
            # FT layout: w_up[e] = (d_model, 2F) with [up, gate] concat
            # along the last dim. Transpose each part: (E, F, D) -> (E, D, F).
            up_T   = up_part.transpose(-2, -1).contiguous().to(dtype)    # (E, D, F)
            gate_T = gate_part.transpose(-2, -1).contiguous().to(dtype)  # (E, D, F)
            w_up_ft.copy_(torch.cat([up_T, gate_T], dim=-1))             # (E, D, 2F)
            # FT down: (E, F, D). HF: (E, D, F). Single transpose.
            w_down_ft.copy_(hf_down.transpose(-2, -1).contiguous().to(dtype))
            continue

        # Per-expert fallback (older saves).
        per_expert_present = False
        for e in range(E):
            gate_name = f"{prefix}.mlp.experts.{e}.gate_proj.weight"
            up_name   = f"{prefix}.mlp.experts.{e}.up_proj.weight"
            down_name = f"{prefix}.mlp.experts.{e}.down_proj.weight"
            shard_g = _open_for(gate_name)
            if shard_g is None:
                continue
            per_expert_present = True
            with safe_open(shard_g, framework="pt", device="cpu") as f:
                gate = f.get_tensor(gate_name)
            with safe_open(_open_for(up_name), framework="pt", device="cpu") as f:
                up = f.get_tensor(up_name)
            with safe_open(_open_for(down_name), framework="pt", device="cpu") as f:
                down = f.get_tensor(down_name)
            gate_t = gate.T.contiguous().to(dtype)
            up_t = up.T.contiguous().to(dtype)
            w_up_ft[e, :, :].copy_(torch.cat([up_t, gate_t], dim=1))
            w_down_ft[e, :, :].copy_(down.T.contiguous().to(dtype))
        if not per_expert_present:
            # Layer expected MoE weights but found none in either format.
            # Don't silently zero out — the engine will run with zero
            # weights which trains to nonsense. Raise loudly.
            raise FileNotFoundError(
                f"Qwen3-Next loader: no expert weights found at "
                f"{prefix}.mlp.experts.* (neither fused nor per-expert) "
                f"for layer {L}. Did the safetensors index get truncated?"
            )

    # ----- (3) Shared-expert weights. -----
    # Qwen3-Next / 3.5 / 3.6 have a single shared expert per MoE layer:
    #   shared_expert.gate_proj.weight   (F_s, d_model)
    #   shared_expert.up_proj.weight     (F_s, d_model)
    #   shared_expert.down_proj.weight   (d_model, F_s)
    #   shared_expert_gate.weight        (1, d_model)
    #
    # FT expects these as (S, d_model, 2F_s), (S, F_s, d_model),
    # (d_model, S) tensors (S = num_shared_experts; 1 for these arches).
    sample_w_shared_up = dest.get(("layer_0", "w_shared_up"))
    if sample_w_shared_up is None:
        return  # No shared-expert weights declared in this model spec.
    S, D_sh, TwoFs = sample_w_shared_up.shape
    Fs = TwoFs // 2
    assert S == 1, (
        f"Qwen3-Next shared-expert loader currently supports S=1; "
        f"got S={S}. (DeepSeek-V3-style S>1 path TBD.)"
    )

    for L in range(num_layers):
        if dest.get((f"layer_{L}", "w_shared_up")) is None:
            continue
        w_shared_up_ft = dest[(f"layer_{L}", "w_shared_up")]      # (1, D, 2F_s)
        w_shared_down_ft = dest[(f"layer_{L}", "w_shared_down")]  # (1, F_s, D)
        w_shared_gate_ft = dest.get((f"layer_{L}", "w_shared_expert_gate"))  # (D, 1)
        dtype = w_shared_up_ft.dtype

        prefix = _resolve_layer_prefix(L)
        if prefix is None:
            continue
        gate_name = f"{prefix}.mlp.shared_expert.gate_proj.weight"
        up_name   = f"{prefix}.mlp.shared_expert.up_proj.weight"
        down_name = f"{prefix}.mlp.shared_expert.down_proj.weight"
        sh_gate_name = f"{prefix}.mlp.shared_expert_gate.weight"

        if _open_for(gate_name) is None:
            continue  # Layer has no shared-expert weights.

        with safe_open(_open_for(gate_name), framework="pt", device="cpu") as f:
            gate = f.get_tensor(gate_name)            # (F_s, D)
        with safe_open(_open_for(up_name), framework="pt", device="cpu") as f:
            up = f.get_tensor(up_name)                # (F_s, D)
        with safe_open(_open_for(down_name), framework="pt", device="cpu") as f:
            down = f.get_tensor(down_name)            # (D, F_s)

        # FT w_shared_up: (1, D, 2F_s) with [up, gate] concat along last dim.
        up_T = up.T.contiguous().to(dtype)            # (D, F_s)
        gate_T = gate.T.contiguous().to(dtype)        # (D, F_s)
        w_shared_up_ft[0].copy_(torch.cat([up_T, gate_T], dim=-1))

        # FT w_shared_down: (1, F_s, D). HF: (D, F_s). Transpose.
        w_shared_down_ft[0].copy_(down.T.contiguous().to(dtype))

        # Shared-expert gate: HF stores (1, D); FT expects (D, S=1).
        if w_shared_gate_ft is not None and _open_for(sh_gate_name) is not None:
            with safe_open(_open_for(sh_gate_name), framework="pt", device="cpu") as f:
                sh_gate = f.get_tensor(sh_gate_name)  # (1, D)
            # Squeeze the leading 1, then unsqueeze trailing for FT (D, 1).
            w_shared_gate_ft.copy_(sh_gate.squeeze(0).unsqueeze(-1).contiguous().to(dtype))


# Linear-attention layer entries (present only at HF layer positions
# where ``layer_types[i] == "linear_attention"``). Marked ``optional``
# so the strict loader doesn't complain on full-attn layer indices.
_LINEAR_ATTN_ENTRIES = (
    WeightMapEntry(
        flextrain_name="w_lin_qkvz",
        hf_name="model.layers.{i}.linear_attn.in_proj_qkvz.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_ba",
        hf_name="model.layers.{i}.linear_attn.in_proj_ba.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_out",
        hf_name="model.layers.{i}.linear_attn.out_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_conv",
        hf_name="model.layers.{i}.linear_attn.conv1d.weight",
        transform=Transform.NONE,        # already (conv_dim, 1, K)
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_dt_bias",
        hf_name="model.layers.{i}.linear_attn.dt_bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_A_log",
        hf_name="model.layers.{i}.linear_attn.A_log",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_norm",
        hf_name="model.layers.{i}.linear_attn.norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
)

# Full-attention layer entries (present only at HF layer positions
# where ``layer_types[i] == "full_attention"``). Same optional treatment
# as linear-attn entries.
_FULL_ATTN_ENTRIES = (
    WeightMapEntry(
        flextrain_name="w_q_norm",
        hf_name="model.layers.{i}.self_attn.q_norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_k_norm",
        hf_name="model.layers.{i}.self_attn.k_norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_q",
        hf_name="model.layers.{i}.self_attn.q_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_k",
        hf_name="model.layers.{i}.self_attn.k_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_v",
        hf_name="model.layers.{i}.self_attn.v_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_o",
        hf_name="model.layers.{i}.self_attn.o_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
)

# Common entries (all layer types).
_COMMON_ENTRIES = (
    WeightMapEntry(
        flextrain_name="w_attn_norm",
        hf_name="model.layers.{i}.input_layernorm.weight",
        transform=Transform.NONE,
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
)


QWEN3_NEXT_ARCH = ArchSpec(
    hf_arch_ids=("Qwen3NextForCausalLM",),
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
    layer=_COMMON_ENTRIES + _LINEAR_ATTN_ENTRIES + _FULL_ATTN_ENTRIES,
    post_load_hook=_qwen3_next_post_load_hook,
)

register_arch(QWEN3_NEXT_ARCH)


def hf_config_to_flextrain(hf_config: Any) -> dict:
    get = (
        (lambda k, default=None: getattr(hf_config, k, default))
        if not isinstance(hf_config, dict)
        else hf_config.get
    )
    n_heads = get("num_attention_heads")
    hidden = get("hidden_size")
    head_dim = get("head_dim") or (hidden // n_heads)
    expert_dim = get("moe_intermediate_size") or get("intermediate_size")
    shared_expert_dim = get("shared_expert_intermediate_size", 0)
    # Qwen3-Next: 1 shared expert per layer when shared_expert_dim > 0.
    num_shared_experts = 1 if shared_expert_dim and shared_expert_dim > 0 else 0
    return {
        "vocab_size": get("vocab_size"),
        "n_layers": get("num_hidden_layers"),
        "d_model": hidden,
        "n_heads": n_heads,
        "n_kv_heads": get("num_key_value_heads") or n_heads,
        "head_dim": head_dim,
        "expert_dim": expert_dim,
        "num_shared_experts": num_shared_experts,
        "shared_expert_dim": shared_expert_dim,
        "num_routed_experts": get("num_experts"),
        "top_k": get("num_experts_per_tok"),
        "is_causal": True,
        "partial_rotary_factor": get("partial_rotary_factor", 0.25),
        # Linear-attention dims.
        "linear_num_v_heads": get("linear_num_value_heads", 32),
        "linear_num_k_heads": get("linear_num_key_heads", 16),
        "linear_head_k_dim": get("linear_key_head_dim", 128),
        "linear_head_v_dim": get("linear_value_head_dim", 128),
        "linear_conv_kernel": get("linear_conv_kernel_dim", 4),
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
    rope_params = get("rope_parameters") or {}
    rope_theta = (
        rope_params.get("rope_theta") if isinstance(rope_params, dict)
        else None
    ) or get("rope_theta", 10_000_000.0)
    norm_topk = bool(get("norm_topk_prob", True))
    return {
        "rms_norm_eps": get("rms_norm_eps", 1e-6),
        "rope_theta": rope_theta,
        "load_balance_coef": get("router_aux_loss_coef", 0.001),
        "routing_mode": "topk_then_softmax" if norm_topk else "softmax_then_topk",
        "layer_types": get("layer_types"),
        "decoder_sparse_step": get("decoder_sparse_step", 1),
    }
