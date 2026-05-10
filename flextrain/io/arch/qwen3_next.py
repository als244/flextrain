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

    # ----- (1.5) Linear-attn column permutation. -----
    # The HF in_proj_qkvz weight has columns interleaved per-K-head:
    # for each K-head h, the row of columns is
    # ``[Q[h] | K[h] | V_grp[h] | Z_grp[h]]``. FT wants the column-block-
    # major layout ``[Q | K | V | Z]`` (head-major within each block),
    # which lets the fwd pass q/k/v/z as zero-copy contiguous slices of
    # the matmul output instead of cat-ing per-K-head.
    # (Same idea for in_proj_ba: HF is ``[B_grp | A_grp]`` per-K-head;
    # FT is ``[B | A]`` flat.)
    # The exporter applies the inverse permutation when serializing.
    cfg_path = os.path.join(hf_path, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            hf_cfg = json.load(f)
        from flextrain.nn.blocks.linear_attn import (
            GatedDeltaNetConfig,
            build_qkvz_perm, build_ba_perm,
        )
        # Read the linear-attn dims via the same getters used in
        # ``hf_config_to_flextrain``.
        _LA_CFG = GatedDeltaNetConfig(
            d_model=hf_cfg.get("hidden_size", 0),
            num_v_heads=hf_cfg.get("linear_num_value_heads", 32),
            num_k_heads=hf_cfg.get("linear_num_key_heads", 16),
            head_k_dim=hf_cfg.get("linear_key_head_dim", 128),
            head_v_dim=hf_cfg.get("linear_value_head_dim", 128),
            conv_kernel_size=hf_cfg.get("linear_conv_kernel_dim", 4),
        )
        perm_qkvz = build_qkvz_perm(_LA_CFG)
        perm_ba = build_ba_perm(_LA_CFG)
        for L in range(num_layers):
            t_qkvz = dest.get((f"layer_{L}", "w_lin_qkvz"))
            if t_qkvz is not None:
                # In-place column permute: t_qkvz is (hidden, proj_qkvz_dim).
                t_qkvz.copy_(t_qkvz[:, perm_qkvz].contiguous())
            t_ba = dest.get((f"layer_{L}", "w_lin_ba"))
            if t_ba is not None:
                t_ba.copy_(t_ba[:, perm_ba].contiguous())

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
    # FT layout (post option-B migration): w_up is (E, 2F, D), w_down (E, D, F).
    E, TwoF, D = sample_w_up.shape
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
            # FT: (E, 2F, D) with [up_first_F, gate_second_F] along dim=1.
            # Same axis order as HF; just swap halves.
            gate_part = hf_gate_up[:, :Fmid, :]                  # (E, F, D)
            up_part   = hf_gate_up[:, Fmid:, :]                  # (E, F, D)
            w_up_ft.copy_(torch.cat([up_part, gate_part], dim=1).to(dtype))  # (E, 2F, D)
            # FT down: (E, D, F). HF: (E, D, F). Direct copy.
            w_down_ft.copy_(hf_down.to(dtype))
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
                gate = f.get_tensor(gate_name)            # (F, D)
            with safe_open(_open_for(up_name), framework="pt", device="cpu") as f:
                up = f.get_tensor(up_name)                # (F, D)
            with safe_open(_open_for(down_name), framework="pt", device="cpu") as f:
                down = f.get_tensor(down_name)            # (D, F)
            # FT w_up[e]: (2F, D) with [up; gate] cat along dim=0.
            # FT w_down[e]: (D, F) — same orientation as HF.
            w_up_ft[e, :, :].copy_(torch.cat([up.to(dtype), gate.to(dtype)], dim=0))
            w_down_ft[e, :, :].copy_(down.to(dtype))
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


ARCH_NAME = "qwen3_next"


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
    """Qwen3-Next dims schema: like Qwen3.5-MoE but with the
    Qwen3-Next-specific 47-1 hybrid pattern by convention. Defaults
    target the published Qwen3-Next-A30B reference shape.
    """
    out = dict(dims)
    missing = [k for k in _REQUIRED_DIMS if k not in out]
    if missing:
        raise KeyError(
            f"qwen3_next dims missing required keys: {missing}. "
            f"Got keys: {sorted(out)}"
        )
    out.setdefault("n_kv_heads", out["n_heads"])
    out.setdefault("is_causal", True)
    out.setdefault("datatypes", dict(_DEFAULT_DATATYPES))
    sed = int(out.get("shared_expert_dim", 512))
    out.setdefault("shared_expert_dim", sed)
    out.setdefault("num_shared_experts", 1 if sed > 0 else 0)
    out.setdefault("partial_rotary_factor", 0.25)
    out.setdefault("linear_num_v_heads", 32)
    out.setdefault("linear_num_k_heads", 16)
    out.setdefault("linear_head_k_dim", 128)
    out.setdefault("linear_head_v_dim", 128)
    out.setdefault("linear_conv_kernel", 4)
    out["attn_dim"] = int(out["n_heads"]) * int(out["head_dim"])
    out["kv_dim"] = int(out["n_kv_heads"]) * int(out["head_dim"])
    nv = int(out["linear_num_v_heads"])
    nk = int(out["linear_num_k_heads"])
    hk = int(out["linear_head_k_dim"])
    hv = int(out["linear_head_v_dim"])
    ck = int(out["linear_conv_kernel"])
    key_dim = nk * hk
    value_dim = nv * hv
    out.setdefault("num_v_heads", nv)
    out.setdefault("num_k_heads", nk)
    out.setdefault("head_k_dim", hk)
    out.setdefault("head_v_dim", hv)
    out.setdefault("key_dim", key_dim)
    out.setdefault("value_dim", value_dim)
    out.setdefault("conv_dim", 2 * key_dim + value_dim)
    out.setdefault("proj_qkvz_dim", 2 * key_dim + 2 * value_dim)
    out.setdefault("proj_ba_dim", 2 * nv)
    out.setdefault("conv_kernel_size", ck)
    return out


def default_hyperparams() -> dict:
    """Qwen3-Next defaults: eps=1e-6, rope=1e7, aux-loss 0.001,
    ``topk_then_softmax``. ``layer_types=None`` ⇒ caller supplies via
    ``hyperparams`` or shorthand ``dims["layer_pattern"]``."""
    return {
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000_000.0,
        "rope_scaling": None,
        "partial_rotary_factor": 0.25,
        "load_balance_coef": 0.001,
        "routing_mode": "topk_then_softmax",
        "layer_types": None,
        "decoder_sparse_step": 1,
    }


def flextrain_to_hf_config(dims, hyperparams=None) -> dict:
    """Inverse mapping for Qwen3-Next."""
    hp = dict(default_hyperparams())
    if hyperparams:
        hp.update(hyperparams)
    d = expand_dims(dims)
    norm_topk = (hp.get("routing_mode") == "topk_then_softmax")
    cfg = {
        "architectures": ["Qwen3NextForCausalLM"],
        "model_type": "qwen3_next",
        "vocab_size": int(d["vocab_size"]),
        "num_hidden_layers": int(d["n_layers"]),
        "hidden_size": int(d["d_model"]),
        "num_attention_heads": int(d["n_heads"]),
        "num_key_value_heads": int(d["n_kv_heads"]),
        "head_dim": int(d["head_dim"]),
        "moe_intermediate_size": int(d["expert_dim"]),
        "num_experts": int(d["num_routed_experts"]),
        "num_experts_per_tok": int(d["top_k"]),
        "norm_topk_prob": norm_topk,
        "router_aux_loss_coef": float(hp.get("load_balance_coef", 0.001)),
        "linear_num_value_heads": int(d["linear_num_v_heads"]),
        "linear_num_key_heads": int(d["linear_num_k_heads"]),
        "linear_key_head_dim": int(d["linear_head_k_dim"]),
        "linear_value_head_dim": int(d["linear_head_v_dim"]),
        "linear_conv_kernel_dim": int(d["linear_conv_kernel"]),
        "rms_norm_eps": float(hp["rms_norm_eps"]),
        "rope_theta": float(hp["rope_theta"]),
        "rope_scaling": hp.get("rope_scaling"),
        "partial_rotary_factor": float(hp["partial_rotary_factor"]),
        "max_position_embeddings": 32768,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "initializer_range": 0.02,
        "torch_dtype": "bfloat16",
        "use_cache": True,
        "decoder_sparse_step": int(hp.get("decoder_sparse_step", 1)),
    }
    sed = int(d.get("shared_expert_dim", 0))
    if sed > 0:
        cfg["shared_expert_intermediate_size"] = sed
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


# ---------------------------------------------------------------------------
# Block builder. Reuses Qwen3NextLinearLayer / Qwen3NextFullLayer
# (the same classes Qwen3.5-MoE uses — math is identical across both
# arches per the existing comment in qwen3_5_moe.py:689-691).
# ---------------------------------------------------------------------------


def _qwen3_next_block_builder(layer_idx: int, ctx) -> object:
    from flextrain.nn.layers.qwen3_next import (
        Qwen3NextLayerConfig, Qwen3NextLinearLayer, Qwen3NextFullLayer,
    )

    dims = ctx.dims
    hp = ctx.hyperparams
    layer_types = hp.get("layer_types") or []
    if layer_idx >= len(layer_types):
        raise ValueError(
            f"qwen3_next layer {layer_idx}: layer_types list has only "
            f"{len(layer_types)} entries"
        )
    is_full = (layer_types[layer_idx] == "full_attention")

    block_cfg = Qwen3NextLayerConfig(
        d_model=int(dims["d_model"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=int(dims["n_kv_heads"]),
        head_dim=int(dims["head_dim"]),
        expert_dim=int(dims["expert_dim"]),
        num_experts=int(dims.get("num_routed_experts", 256)),
        top_k=int(dims.get("top_k", 8)),
        linear_num_v_heads=int(dims.get("linear_num_v_heads", 32)),
        linear_num_k_heads=int(dims.get("linear_num_k_heads", 16)),
        linear_head_k_dim=int(dims.get("linear_head_k_dim", 128)),
        linear_head_v_dim=int(dims.get("linear_head_v_dim", 128)),
        linear_conv_kernel=int(dims.get("linear_conv_kernel", 4)),
        partial_rotary_factor=float(hp.get("partial_rotary_factor", 0.25)),
        num_shared_experts=int(dims.get("num_shared_experts", 1)),
        shared_expert_dim=int(dims.get("shared_expert_dim", 512)),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-6)),
        rope_base=float(hp.get("rope_theta", 10_000_000.0)),
        is_causal=True,
        load_balance_coef=float(hp.get("load_balance_coef", 0.001)),
        routing_mode=hp.get("routing_mode", "topk_then_softmax"),
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )

    moe_backend = getattr(ctx, "moe_backend", None)
    if is_full:
        base = Qwen3NextFullLayer(
            layer_id=layer_idx, cfg=block_cfg, expert_compute=moe_backend,
        )
    else:
        base = Qwen3NextLinearLayer(
            layer_id=layer_idx, cfg=block_cfg, expert_compute=moe_backend,
        )

    if not ctx.lora_targets:
        return base
    # Same per-layer LoRA-target filtering as Qwen3.5-MoE (linear layers
    # have no w_q/w_k/w_v/w_o, so an attn-only target list would error
    # there). Mirror HF PEFT's per-layer auto-skip semantics.
    from flextrain.nn.layers.lora_wrapper import (
        LoRAWrapperLayer, _discover_lora_eligible_names,
    )
    layer_dims = dict(
        dims,
        attn_dim=int(dims["n_heads"]) * int(dims["head_dim"]),
        kv_dim=int(dims["n_kv_heads"]) * int(dims["head_dim"]),
    )
    if ctx.lora_targets != "all" and ctx.lora_targets is not None:
        eligible = set(_discover_lora_eligible_names(base.param_spec, layer_dims))
        kept = tuple(t for t in ctx.lora_targets if t in eligible)
        if not kept:
            from dataclasses import replace as _replace
            from flextrain.core.layer import ParamSpec
            base.param_spec = ParamSpec(tensors=tuple(
                _replace(t, frozen=True) for t in base.param_spec.tensors
            ))
            return base
        targets_for_layer: object = kept
    else:
        targets_for_layer = ctx.lora_targets
    return LoRAWrapperLayer(
        base, lora_targets=targets_for_layer,
        rank=ctx.lora_rank, alpha=ctx.lora_alpha,
        adapter_compute_dtype=ctx.lora_adapter_compute_dtype,
        adapter_master_dtype=ctx.lora_adapter_master_dtype,
        adapter_grad_dtype=ctx.lora_adapter_grad_dtype,
        adapter_opt_state_dtype=ctx.lora_adapter_opt_state_dtype,
        layer_dims=layer_dims,
    )


def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(("Qwen3NextForCausalLM",), _qwen3_next_block_builder)


_register_builder()


BLOCK_BUILDER = _qwen3_next_block_builder
