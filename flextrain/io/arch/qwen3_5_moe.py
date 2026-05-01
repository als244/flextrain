"""Qwen3.5-MoE (e.g. Qwen/Qwen3.5-35B-A3B) HF -> FT weight loader.

Architectural recap (verified by reading the HF safetensors index of
``Qwen3.5-35B-A3B`` and the HF ``modeling_qwen3_5_moe`` source):

* **Hybrid backbone** — ``layer_types[L]`` is one of
  ``"linear_attention"`` (Qwen3_5MoeGatedDeltaNet) or
  ``"full_attention"`` (Qwen3_5MoeAttention with sigmoid output gate +
  per-head QK-norm + partial-rotary RoPE, factor 0.25).
* **MoE FFN every layer** — Qwen3_5MoeSparseMoeBlock with 256 routed
  experts (top-K=8 by config), 1 shared expert with intermediate dim
  ``shared_expert_intermediate_size`` and a sigmoid gate
  ``Linear(hidden, 1)``.
* **Multimodal wrapper** — text weights live under
  ``model.language_model.layers.{L}.*`` (the model also ships a vision
  tower under ``model.visual.*`` which we ignore).
* **(1+w) RMSNorm convention** for the NON-gated norms (input_layernorm,
  post_attention_layernorm, q_norm, k_norm). Stored γ is
  ``γ_canonical - 1``; we shift by +1 at load. The gated norm
  ``linear_attn.norm`` uses ``init.ones_`` + plain ``weight * x``, so
  must NOT be shifted (same convention as Qwen3-Next gated norm).
* **q_proj per-head ``[q | gate]`` interleaving** — HF stores
  ``q_proj.weight: (n_heads * head_dim * 2, hidden)`` with each head's
  output chunk split as ``[q_h(head_dim) | gate_h(head_dim)]``. FT
  expects flat ``[Q_all(attn_dim) | gate_all(attn_dim)]``. We permute
  in :func:`post_load_permute` and additionally apply the standard
  halved->pair RoPE permutation to the Q half.
* **Linear-attn projections split-vs-bundled** — HF stores
  ``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` / ``in_proj_a`` as
  4 separate tensors; FT consumes them bundled per-K-head into
  ``w_lin_qkvz`` and ``w_lin_ba``. We do the bundling at load time.
* **Batched expert weights** — HF stores
  ``mlp.experts.gate_up_proj: (E, 2F, D)`` and
  ``mlp.experts.down_proj: (E, D, F)`` in the fused (newer) format.
  FT expects ``w_up: (E, D, 2F)`` ([up | gate] concat) and
  ``w_down: (E, F, D)``.
* **Tied word embeddings**: NOT TIED for the 35B (config:
  ``tie_word_embeddings: False``); ``lm_head.weight`` ships as a
  separate tensor.

Layer-side modeling code: we reuse :class:`Qwen3NextLinearLayer` /
:class:`Qwen3NextFullLayer` directly — verified by HF source-read,
their math is identical to Qwen3.5-MoE's decoder layer (gated GQA with
QK-norm, partial-rotary RoPE; gated DeltaNet linear attention; MoE
FFN with top-K routed experts and shared expert with sigmoid gate).
The only differences for this checkpoint are config values (40 layers,
256 experts, top_k=8, head_dim=256, shared_expert_dim=512) and the
loader specifics handled in this file.
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

import torch
from safetensors import safe_open

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


# ---------------------------------------------------------------------------
# Layer-prefix probing.
# ---------------------------------------------------------------------------


def _resolve_layer_prefix(file_index, hf_path, layer_idx: int) -> str | None:
    """Multimodal saves nest text weights under ``model.language_model.``;
    text-only saves use ``model.``. Probe both, return whichever has
    matching keys for this layer."""
    for prefix in (
        f"model.language_model.layers.{layer_idx}",
        f"model.layers.{layer_idx}",
    ):
        for probe in (
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
        ):
            if file_index is not None:
                if probe in file_index:
                    return prefix
            else:
                return prefix
    return None


def _open_for(file_index, hf_path, name):
    if file_index is not None:
        shard = file_index.get(name)
        if shard is None:
            return None
        return os.path.join(hf_path, shard)
    for fn in sorted(os.listdir(hf_path)):
        if fn.endswith(".safetensors"):
            full = os.path.join(hf_path, fn)
            try:
                with safe_open(full, framework="pt", device="cpu") as f:
                    if name in f.keys():
                        return full
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# post_load_hook: (1+w) RMSNorm shift, linear-attn bundling, expert/shared
# expert weight stacking. Mirrors qwen3_next + qwen3_5 hooks combined.
# ---------------------------------------------------------------------------


def _qwen3_5_moe_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Six operations, in order:

    1. ``(1 + γ_HF)`` shift on every NON-gated RMSNorm γ (matches
       Qwen3.5/Qwen3-Next/Gemma2 convention).
    2. Bundle per-K-head ``in_proj_qkv``/``in_proj_z``/``in_proj_b``/
       ``in_proj_a`` into FT ``w_lin_qkvz`` / ``w_lin_ba``.
    3. Stack per-expert HF ``mlp.experts.{gate_up_proj, down_proj}``
       into FT ``w_up`` / ``w_down`` (fused-format only — older
       per-expert format fallback included for completeness).
    4. Stack per-layer shared-expert ``mlp.shared_expert.*`` and
       ``mlp.shared_expert_gate.weight`` into FT
       ``w_shared_up`` / ``w_shared_down`` / ``w_shared_expert_gate``.
    """
    # Locate shard index.
    idx_path = os.path.join(hf_path, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            file_index = json.load(f)["weight_map"]
    else:
        file_index = None

    # Read text config for linear-attn dims (needed for bundling).
    cfg_path = os.path.join(hf_path, "config.json")
    with open(cfg_path) as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg.get("text_config", hf_cfg)
    cfg_num_k = int(text_cfg.get("linear_num_key_heads", 16))
    cfg_num_v = int(text_cfg.get("linear_num_value_heads", 32))

    # ----- (1) RMSNorm γ shift on the non-gated norms. -----
    _shift_norm_names = (
        "w_attn_norm",     # input_layernorm
        "w_ffn_norm",      # post_attention_layernorm
        "w_q_norm",        # per-head q_norm
        "w_k_norm",        # per-head k_norm
        # NOTE: w_lin_norm is the GATED norm inside DeltaNet; its
        # HF init is ones_ + plain ``weight * x`` -> store γ directly.
        # Must NOT be shifted.
    )
    for L in range(num_layers):
        for n in _shift_norm_names:
            t = dest.get((f"layer_{L}", n))
            if t is not None:
                t.add_(1.0)
    final_norm = dest.get(("head", "w_final_norm"))
    if final_norm is not None:
        final_norm.add_(1.0)

    # ----- (2) Linear-attn projection bundling. -----
    for L in range(num_layers):
        ft_qkvz = dest.get((f"layer_{L}", "w_lin_qkvz"))
        ft_ba = dest.get((f"layer_{L}", "w_lin_ba"))
        if ft_qkvz is None or ft_ba is None:
            continue  # not a linear-attn layer
        prefix = _resolve_layer_prefix(file_index, hf_path, L)
        if prefix is None:
            continue
        in_qkv_name = f"{prefix}.linear_attn.in_proj_qkv.weight"
        in_z_name = f"{prefix}.linear_attn.in_proj_z.weight"
        in_b_name = f"{prefix}.linear_attn.in_proj_b.weight"
        in_a_name = f"{prefix}.linear_attn.in_proj_a.weight"

        shard_qkv = _open_for(file_index, hf_path, in_qkv_name)
        if shard_qkv is None:
            # Layer not actually a linear-attn layer; skip silently.
            continue
        with safe_open(shard_qkv, framework="pt", device="cpu") as f:
            hf_qkv = f.get_tensor(in_qkv_name)            # (kd2+vd, hidden)
        with safe_open(_open_for(file_index, hf_path, in_z_name),
                       framework="pt", device="cpu") as f:
            hf_z = f.get_tensor(in_z_name)                # (vd, hidden)
        with safe_open(_open_for(file_index, hf_path, in_b_name),
                       framework="pt", device="cpu") as f:
            hf_b = f.get_tensor(in_b_name)                # (num_v, hidden)
        with safe_open(_open_for(file_index, hf_path, in_a_name),
                       framework="pt", device="cpu") as f:
            hf_a = f.get_tensor(in_a_name)                # (num_v, hidden)

        ft_dtype = ft_qkvz.dtype
        proj_qkvz_dim = ft_qkvz.shape[1]
        proj_ba_dim = ft_ba.shape[1]
        hidden = ft_qkvz.shape[0]
        num_v = proj_ba_dim // 2
        kd2_plus_vd = hf_qkv.shape[0]
        value_dim = hf_z.shape[0]
        key_dim_x2 = kd2_plus_vd - value_dim
        assert key_dim_x2 % 2 == 0
        key_dim = key_dim_x2 // 2

        assert cfg_num_v == num_v, (
            f"qwen3_5_moe loader: ba shape implies num_v={num_v} but "
            f"config says linear_num_value_heads={cfg_num_v}"
        )
        num_k = cfg_num_k
        assert num_v % num_k == 0, (
            f"linear_num_value_heads ({num_v}) must be divisible by "
            f"linear_num_key_heads ({num_k})"
        )
        grp = num_v // num_k
        hk = key_dim // num_k
        hv = value_dim // num_v
        head_block = 2 * hk + 2 * grp * hv
        assert num_k * head_block == proj_qkvz_dim, (
            f"qkvz bundling mismatch: num_k={num_k}, head_block="
            f"{head_block}, expected {proj_qkvz_dim}, got "
            f"{num_k*head_block}"
        )

        # FT block-major column layout: ``[Q | K | V | Z]`` over the
        # column axis; each block laid out as (head, dim) row-major.
        # See :func:`flextrain.nn.blocks.linear_attn.build_qkvz_perm` for
        # why and the inverse permutation used by the exporter.
        # HF input shapes are (out, in); we transpose to (in, out) at
        # the end. ``hf_qkv`` is the fused [Q-block | K-block | V-block]
        # along its first axis already (Qwen3.5 stored them split that
        # way per the in_proj_qkv tensor).
        q_block = hf_qkv[:key_dim, :]                           # (key_dim, hidden)
        k_block = hf_qkv[key_dim:2*key_dim, :]                  # (key_dim, hidden)
        v_block = hf_qkv[2*key_dim:, :]                         # (value_dim, hidden)
        z_block = hf_z                                          # (value_dim, hidden)
        # Stack along the out-axis: ``[Q | K | V | Z]`` then transpose
        # to (in, out) for FT.
        bundled = torch.cat([q_block, k_block, v_block, z_block], dim=0)
        bundled = bundled.T.contiguous()                        # (hidden, proj_qkvz_dim)
        ft_qkvz.copy_(bundled.to(ft_dtype))

        # FT ba layout: ``[B | A]`` along columns (no per-K-head structure).
        # HF stores in_proj_b and in_proj_a as separate (num_v, hidden) tensors.
        bundled_ba = torch.cat([hf_b, hf_a], dim=0)             # (2*num_v, hidden)
        bundled_ba = bundled_ba.T.contiguous()                  # (hidden, proj_ba_dim)
        ft_ba.copy_(bundled_ba.to(ft_dtype))

    # ----- (3) Stack routed expert weights (fused HF format). -----
    sample_w_up = dest.get(("layer_0", "w_up"))
    if sample_w_up is None:
        return  # No MoE at all.
    E, D, TwoF = sample_w_up.shape
    Fmid = TwoF // 2

    for L in range(num_layers):
        if dest.get((f"layer_{L}", "w_up")) is None:
            continue
        w_up_ft = dest[(f"layer_{L}", "w_up")]
        w_down_ft = dest[(f"layer_{L}", "w_down")]
        dtype = w_up_ft.dtype

        prefix = _resolve_layer_prefix(file_index, hf_path, L)
        if prefix is None:
            continue

        fused_gate_up_name = f"{prefix}.mlp.experts.gate_up_proj"
        fused_down_name = f"{prefix}.mlp.experts.down_proj"
        gu_shard = _open_for(file_index, hf_path, fused_gate_up_name)
        if gu_shard is not None:
            with safe_open(gu_shard, framework="pt", device="cpu") as f:
                hf_gate_up = f.get_tensor(fused_gate_up_name)  # (E, 2F, D)
            with safe_open(
                _open_for(file_index, hf_path, fused_down_name),
                framework="pt", device="cpu",
            ) as f:
                hf_down = f.get_tensor(fused_down_name)        # (E, D, F)
            # HF stores gate first, up second along the 2F axis.
            gate_part = hf_gate_up[:, :Fmid, :]                # (E, F, D)
            up_part = hf_gate_up[:, Fmid:, :]                  # (E, F, D)
            up_T = up_part.transpose(-2, -1).contiguous().to(dtype)
            gate_T = gate_part.transpose(-2, -1).contiguous().to(dtype)
            # FT layout: w_up[e] = (D, 2F) with [up | gate] concat.
            w_up_ft.copy_(torch.cat([up_T, gate_T], dim=-1))
            # FT down[e] = (F, D); HF is (D, F).
            w_down_ft.copy_(hf_down.transpose(-2, -1).contiguous().to(dtype))
            continue

        # Per-expert fallback (older saves).
        per_expert_present = False
        for e in range(E):
            gate_name = f"{prefix}.mlp.experts.{e}.gate_proj.weight"
            up_name = f"{prefix}.mlp.experts.{e}.up_proj.weight"
            down_name = f"{prefix}.mlp.experts.{e}.down_proj.weight"
            shard_g = _open_for(file_index, hf_path, gate_name)
            if shard_g is None:
                continue
            per_expert_present = True
            with safe_open(shard_g, framework="pt", device="cpu") as f:
                gate = f.get_tensor(gate_name)
            with safe_open(_open_for(file_index, hf_path, up_name),
                           framework="pt", device="cpu") as f:
                up = f.get_tensor(up_name)
            with safe_open(_open_for(file_index, hf_path, down_name),
                           framework="pt", device="cpu") as f:
                down = f.get_tensor(down_name)
            gate_t = gate.T.contiguous().to(dtype)
            up_t = up.T.contiguous().to(dtype)
            w_up_ft[e, :, :].copy_(torch.cat([up_t, gate_t], dim=1))
            w_down_ft[e, :, :].copy_(down.T.contiguous().to(dtype))
        if not per_expert_present:
            raise FileNotFoundError(
                f"qwen3_5_moe loader: no expert weights found for "
                f"layer {L} under {prefix}.mlp.experts.* (neither fused "
                f"nor per-expert format)."
            )

    # ----- (4) Shared-expert weights. -----
    sample_w_shared_up = dest.get(("layer_0", "w_shared_up"))
    if sample_w_shared_up is None:
        return
    S, D_sh, TwoFs = sample_w_shared_up.shape
    Fs = TwoFs // 2
    assert S == 1, (
        f"qwen3_5_moe loader supports S=1 shared expert; got S={S}."
    )

    for L in range(num_layers):
        if dest.get((f"layer_{L}", "w_shared_up")) is None:
            continue
        w_shared_up_ft = dest[(f"layer_{L}", "w_shared_up")]
        w_shared_down_ft = dest[(f"layer_{L}", "w_shared_down")]
        w_shared_gate_ft = dest.get((f"layer_{L}", "w_shared_expert_gate"))
        dtype = w_shared_up_ft.dtype

        prefix = _resolve_layer_prefix(file_index, hf_path, L)
        if prefix is None:
            continue
        gate_name = f"{prefix}.mlp.shared_expert.gate_proj.weight"
        up_name = f"{prefix}.mlp.shared_expert.up_proj.weight"
        down_name = f"{prefix}.mlp.shared_expert.down_proj.weight"
        sh_gate_name = f"{prefix}.mlp.shared_expert_gate.weight"

        if _open_for(file_index, hf_path, gate_name) is None:
            continue

        with safe_open(_open_for(file_index, hf_path, gate_name),
                       framework="pt", device="cpu") as f:
            gate = f.get_tensor(gate_name)            # (Fs, D)
        with safe_open(_open_for(file_index, hf_path, up_name),
                       framework="pt", device="cpu") as f:
            up = f.get_tensor(up_name)                # (Fs, D)
        with safe_open(_open_for(file_index, hf_path, down_name),
                       framework="pt", device="cpu") as f:
            down = f.get_tensor(down_name)            # (D, Fs)

        up_T = up.T.contiguous().to(dtype)
        gate_T = gate.T.contiguous().to(dtype)
        w_shared_up_ft[0].copy_(torch.cat([up_T, gate_T], dim=-1))
        w_shared_down_ft[0].copy_(down.T.contiguous().to(dtype))

        if w_shared_gate_ft is not None and \
                _open_for(file_index, hf_path, sh_gate_name) is not None:
            with safe_open(_open_for(file_index, hf_path, sh_gate_name),
                           framework="pt", device="cpu") as f:
                sh_gate = f.get_tensor(sh_gate_name)  # (1, D)
            w_shared_gate_ft.copy_(
                sh_gate.squeeze(0).unsqueeze(-1).contiguous().to(dtype)
            )


# ---------------------------------------------------------------------------
# Weight map entries.
# ---------------------------------------------------------------------------


_COMMON_ENTRIES = (
    WeightMapEntry(
        flextrain_name="w_attn_norm",
        hf_name="model.language_model.layers.{i}.input_layernorm.weight",
        transform=Transform.NONE,
    ),
    WeightMapEntry(
        flextrain_name="w_ffn_norm",
        hf_name="model.language_model.layers.{i}.post_attention_layernorm.weight",
        transform=Transform.NONE,
    ),
    WeightMapEntry(
        flextrain_name="w_router",
        hf_name="model.language_model.layers.{i}.mlp.gate.weight",
        transform=Transform.TRANSPOSE,
    ),
    # w_up, w_down, w_shared_up, w_shared_down, w_shared_expert_gate
    # filled by post_load_hook.
)

# Linear-attn layers.
_LINEAR_ATTN_ENTRIES = (
    WeightMapEntry(
        flextrain_name="w_lin_out",
        hf_name="model.language_model.layers.{i}.linear_attn.out_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_conv",
        hf_name="model.language_model.layers.{i}.linear_attn.conv1d.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_dt_bias",
        hf_name="model.language_model.layers.{i}.linear_attn.dt_bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_A_log",
        hf_name="model.language_model.layers.{i}.linear_attn.A_log",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_norm",
        hf_name="model.language_model.layers.{i}.linear_attn.norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    # w_lin_qkvz, w_lin_ba populated by post_load_hook (bundled).
)

# Full-attn layers. ``w_q`` is loaded as the raw HF tensor (per-head
# [q | gate]); post_load_permute will re-shape into FT [Q | gate] flat.
_FULL_ATTN_ENTRIES = (
    WeightMapEntry(
        flextrain_name="w_q_norm",
        hf_name="model.language_model.layers.{i}.self_attn.q_norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_k_norm",
        hf_name="model.language_model.layers.{i}.self_attn.k_norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_q",
        hf_name="model.language_model.layers.{i}.self_attn.q_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_k",
        hf_name="model.language_model.layers.{i}.self_attn.k_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_v",
        hf_name="model.language_model.layers.{i}.self_attn.v_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_o",
        hf_name="model.language_model.layers.{i}.self_attn.o_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
)


QWEN3_5_MOE_ARCH = ArchSpec(
    hf_arch_ids=("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"),
    embed=(
        WeightMapEntry(
            flextrain_name="w_tok_embeddings",
            hf_name="model.language_model.embed_tokens.weight",
            transform=Transform.NONE,
        ),
    ),
    head=(
        WeightMapEntry(
            flextrain_name="w_final_norm",
            hf_name="model.language_model.norm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_head_proj",
            hf_name="lm_head.weight",
            transform=Transform.TRANSPOSE,
        ),
    ),
    layer=_COMMON_ENTRIES + _LINEAR_ATTN_ENTRIES + _FULL_ATTN_ENTRIES,
    post_load_hook=_qwen3_5_moe_post_load_hook,
)

register_arch(QWEN3_5_MOE_ARCH)


# ---------------------------------------------------------------------------
# Config translation.
# ---------------------------------------------------------------------------


def _text_cfg(hf_config):
    if isinstance(hf_config, dict):
        return hf_config.get("text_config", hf_config)
    return getattr(hf_config, "text_config", hf_config)


def hf_config_to_flextrain(hf_config: Any) -> dict:
    tc = _text_cfg(hf_config)
    g = (tc.get if isinstance(tc, dict) else lambda k, default=None: getattr(tc, k, default))
    n_heads = g("num_attention_heads")
    hidden = g("hidden_size")
    head_dim = g("head_dim") or (hidden // n_heads)
    expert_dim = g("moe_intermediate_size") or g("intermediate_size")
    shared_expert_dim = g("shared_expert_intermediate_size", 0)
    num_shared_experts = 1 if shared_expert_dim and shared_expert_dim > 0 else 0
    num_v_heads = g("linear_num_value_heads", 32)
    num_k_heads = g("linear_num_key_heads", 16)
    head_k_dim = g("linear_key_head_dim", 128)
    head_v_dim = g("linear_value_head_dim", 128)
    conv_kernel = g("linear_conv_kernel_dim", 4)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    return {
        "vocab_size": g("vocab_size"),
        "n_layers": g("num_hidden_layers"),
        "d_model": hidden,
        "n_heads": n_heads,
        "n_kv_heads": g("num_key_value_heads") or n_heads,
        "head_dim": head_dim,
        "expert_dim": expert_dim,
        "num_shared_experts": num_shared_experts,
        "shared_expert_dim": shared_expert_dim,
        "num_routed_experts": g("num_experts"),
        "top_k": g("num_experts_per_tok"),
        "is_causal": True,
        "partial_rotary_factor": g("partial_rotary_factor", 0.25),
        # Linear-attn dims, both raw and "block param_spec" forms.
        "linear_num_v_heads": num_v_heads,
        "linear_num_k_heads": num_k_heads,
        "linear_head_k_dim": head_k_dim,
        "linear_head_v_dim": head_v_dim,
        "linear_conv_kernel": conv_kernel,
        "num_v_heads": num_v_heads,
        "num_k_heads": num_k_heads,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "conv_dim": 2 * key_dim + value_dim,
        "proj_qkvz_dim": 2 * key_dim + 2 * value_dim,
        "proj_ba_dim": 2 * num_v_heads,
        "conv_kernel_size": conv_kernel,
        "datatypes": {
            "embed": "bfloat16", "head_proj": "bfloat16",
            "attn_proj": "bfloat16", "expert_proj": "bfloat16",
            "router": "bfloat16", "norm": "bfloat16",
            "residual": "bfloat16",
        },
    }


def hf_config_to_hyperparams(hf_config: Any) -> dict:
    tc = _text_cfg(hf_config)
    g = (tc.get if isinstance(tc, dict) else lambda k, default=None: getattr(tc, k, default))
    rope_params = g("rope_parameters") or {}
    rope_theta = (
        rope_params.get("rope_theta") if isinstance(rope_params, dict)
        else None
    ) or g("rope_theta", 10_000_000.0)
    partial_rotary = (
        rope_params.get("partial_rotary_factor")
        if isinstance(rope_params, dict) else None
    ) or g("partial_rotary_factor", 0.25)
    norm_topk = bool(g("norm_topk_prob", True))
    return {
        "rms_norm_eps": g("rms_norm_eps", 1e-6),
        "rope_theta": rope_theta,
        "partial_rotary_factor": partial_rotary,
        "load_balance_coef": g("router_aux_loss_coef", 0.001),
        "routing_mode": "topk_then_softmax" if norm_topk else "softmax_then_topk",
        "layer_types": g("layer_types"),
        "full_attention_interval": g("full_attention_interval"),
        "attn_output_gate": g("attn_output_gate", True),
    }


# ---------------------------------------------------------------------------
# Block builder. Reuses Qwen3NextLinearLayer / Qwen3NextFullLayer
# (verified by reading HF Qwen3_5MoeAttention / Qwen3_5MoeGatedDeltaNet
# / Qwen3_5MoeSparseMoeBlock — math is identical).
# ---------------------------------------------------------------------------


def _qwen3_5_moe_block_builder(layer_idx: int, ctx) -> object:
    from flextrain.nn.layers.qwen3_next import (
        Qwen3NextLayerConfig, Qwen3NextLinearLayer, Qwen3NextFullLayer,
    )

    dims = ctx.dims
    hp = ctx.hyperparams
    layer_types = hp.get("layer_types") or []
    if layer_idx >= len(layer_types):
        raise ValueError(
            f"qwen3_5_moe layer {layer_idx}: layer_types list has only "
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
    from flextrain.nn.layers.lora_wrapper import (
        LoRAWrapperLayer, _discover_lora_eligible_names,
    )
    layer_dims = dict(
        dims,
        attn_dim=int(dims["n_heads"]) * int(dims["head_dim"]),
        kv_dim=int(dims["n_kv_heads"]) * int(dims["head_dim"]),
    )
    # Hybrid backbone: linear-attn layers have no w_q/w_k/w_v/w_o, so an
    # explicit target list like ("w_q","w_k","w_v","w_o") would error
    # there. Filter the user's targets against this layer's own
    # LoRA-eligible names, mirroring HF PEFT's per-layer auto-skipping.
    if ctx.lora_targets != "all" and ctx.lora_targets is not None:
        eligible = set(_discover_lora_eligible_names(base.param_spec, layer_dims))
        kept = tuple(t for t in ctx.lora_targets if t in eligible)
        if not kept:
            # No LoRA targets apply to this layer (e.g. attn-only LoRA on
            # a linear-attn layer). Still mark all base params frozen so
            # working_set + buffers skip grad / opt-state allocation —
            # matches the behavior of LoRAWrapperLayer when targets are
            # present (it freezes ALL base tensors, not just targets).
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
        dims=layer_dims,
        adapter_compute_dtype=ctx.lora_adapter_compute_dtype,
        adapter_master_dtype=ctx.lora_adapter_master_dtype,
        adapter_grad_dtype=ctx.lora_adapter_grad_dtype,
        adapter_opt_state_dtype=ctx.lora_adapter_opt_state_dtype,
    )


# ---------------------------------------------------------------------------
# post_load_permute: q_proj per-head [q|gate] -> flat [Q|gate], halved->pair
# RoPE permutation on Q and K. Mirrors qwen3_5.post_load_permute.
# Tied embeddings: not used here (Qwen3.5-MoE-35B has tie_word_embeddings=False).
# ---------------------------------------------------------------------------


def post_load_permute(am, hf_config, dims, hyperparams):
    """Two operations:

    1. Per-head w_q ``[q | gate]`` -> flat ``[Q | gate]``, applying
       halved->pair RoPE permutation to the Q half (gate is not RoPE'd).
    2. K halved->pair permutation. q_norm/k_norm vectors get the same
       per-head-internal permutation (their weight is sized ``head_dim``).

    Matches qwen3_5.post_load_permute. Tied-embedding mirror is omitted
    since Qwen3.5-MoE has tie_word_embeddings=False.
    """
    head_dim = int(dims["head_dim"])
    n_heads = int(dims["n_heads"])
    n_kv = int(dims["n_kv_heads"])
    n_layers = int(dims["n_layers"])
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    layer_types = hyperparams.get("layer_types") or []
    partial_rotary_factor = float(
        hyperparams.get("partial_rotary_factor", 0.25)
    )
    rope_dim = int(head_dim * partial_rotary_factor)

    def _halved_to_pair(dim: int, hd: int) -> torch.Tensor:
        """Per-head halved->pair on the first ``rope_dim`` channels."""
        out = torch.arange(dim, dtype=torch.int64)
        half = rope_dim // 2
        for h in range(dim // hd):
            base = h * hd
            for i in range(half):
                out[base + 2 * i] = base + i
                out[base + 2 * i + 1] = base + half + i
        return out

    q_perm = _halved_to_pair(attn_dim, head_dim)
    k_perm = _halved_to_pair(kv_dim, head_dim)
    qk_norm_perm = _halved_to_pair(head_dim, head_dim)

    def _split_perm_q_with_gate(host_t, n_heads: int, head_dim: int):
        """HF q_proj weight loaded as (in, n_heads*2*head_dim) with
        per-head [q | gate]. Reshape, split, RoPE-permute Q only,
        flatten back as [Q_all | gate_all]."""
        in_dim, out_dim = host_t.shape
        assert out_dim == n_heads * 2 * head_dim
        v = host_t.reshape(in_dim, n_heads, 2 * head_dim)
        q_part = v[:, :, :head_dim].contiguous()
        gate_part = v[:, :, head_dim:].contiguous()
        q_flat = q_part.reshape(in_dim, n_heads * head_dim)
        q_flat = q_flat[:, q_perm].contiguous()
        gate_flat = gate_part.reshape(in_dim, n_heads * head_dim)
        return torch.cat([q_flat, gate_flat], dim=-1)

    for L in range(n_layers):
        if L >= len(layer_types):
            continue
        host = am.buffers.host_params[L]
        if layer_types[L] != "full_attention":
            continue

        if "w_q" in host:
            new_w_q = _split_perm_q_with_gate(host["w_q"], n_heads, head_dim)
            host["w_q"].copy_(new_w_q)

        if "w_k" in host:
            host["w_k"].copy_(host["w_k"][:, k_perm])

        for nm in ("w_q_norm", "w_k_norm"):
            if nm in host:
                w = host[nm]
                if w.dim() == 1 and w.shape[0] == head_dim:
                    host[nm].copy_(w[qk_norm_perm])

    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        if name in am.buffers.host_head_params:
            dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Block-builder registration.
# ---------------------------------------------------------------------------


def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(
        ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"),
        _qwen3_5_moe_block_builder,
    )


_register_builder()
