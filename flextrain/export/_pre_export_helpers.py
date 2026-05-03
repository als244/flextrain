"""Shared helpers for ``ArchSpec.pre_export_hook`` implementations.

The pre-export hooks need to invert what each arch's ``post_load_hook``
did at load time. Three load-time operations recur across arches:

1. **Stack per-expert HF tensors → FT 3-D MoE weights**:
   HF stores ``experts.{e}.{gate,up,down}_proj.weight`` (per-expert
   triplet) or fused ``experts.gate_up_proj`` / ``experts.down_proj``;
   FT stacks them into ``w_up (E, 2F, d)`` chunked ``[up; gate]`` and
   ``w_down (E, d, F)``. Inverse: slice per ``e`` and emit either
   per-expert HF tensors or the fused HF tensors.

2. **Bundle per-K-head linear-attn projections → FT block-major**:
   HF stores split ``in_proj_qkv`` (key_dim*2 + value_dim, hidden) +
   ``in_proj_z`` (value_dim, hidden) + ``in_proj_b`` (num_v, hidden) +
   ``in_proj_a`` (num_v, hidden); FT bundles into ``w_lin_qkvz`` and
   ``w_lin_ba`` block-major over column axis. Inverse: split.

3. **(1+w) RMSNorm shift**: HF's ``forward(x) = x_norm * (1 + weight)``
   stores ``γ - 1``; FT pre-shifts to ``γ`` directly at load time.
   Inverse: subtract 1 from every non-gated norm γ.

These helpers operate on the HF-named ``dst`` dict (already populated
by ``_build_hf_state_dict_from_archspec``). They DELETE any FT-bundled
entry (which the ArchSpec walk emitted under a wrong HF name) and
WRITE the correctly-named per-expert / split / shifted tensors.
"""
from __future__ import annotations

import os
from typing import MutableMapping

import torch


# ---------------------------------------------------------------------------
# (1) MoE expert unstack.
# ---------------------------------------------------------------------------


def detect_fused_moe_format(hf_source_dir: str | None) -> bool | None:
    """Probe the source HF dir's safetensors index to decide whether
    experts are stored fused (``experts.gate_up_proj`` (E, 2F, d) +
    ``experts.down_proj`` (E, d, F)) or per-expert
    (``experts.{e}.{gate,up,down}_proj.weight``).

    Returns:
        True  -> fused format
        False -> per-expert format
        None  -> can't tell (no source dir, no index, no MoE keys)
    """
    if not hf_source_dir:
        return None
    idx_path = os.path.join(hf_source_dir, "model.safetensors.index.json")
    keys: list[str] = []
    if os.path.isfile(idx_path):
        import json
        with open(idx_path) as f:
            keys = list(json.load(f)["weight_map"].keys())
    else:
        # Single-shard fallback.
        try:
            from safetensors import safe_open
            single = os.path.join(hf_source_dir, "model.safetensors")
            if os.path.isfile(single):
                with safe_open(single, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
        except Exception:
            return None
    if not keys:
        return None
    for k in keys:
        if k.endswith(".mlp.experts.gate_up_proj"):
            return True
        if "mlp.experts." in k and (
            k.endswith(".gate_proj.weight")
            or k.endswith(".up_proj.weight")
            or k.endswith(".down_proj.weight")
        ):
            # Need the per-expert variant: ``mlp.experts.{e}.{kind}_proj.weight``.
            tail = k.split("mlp.experts.", 1)[1]
            if tail and tail[0].isdigit():
                return False
    return None


def emit_routed_experts(
    dst: MutableMapping[str, torch.Tensor],
    *,
    layer_prefix: str,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    fused: bool,
) -> None:
    """Inverse of MoE expert stacking.

    Args
    ----
    dst
        HF state dict to mutate. Per-expert / fused tensors are added;
        any pre-existing wrong entries (e.g. an ArchSpec emit of the
        FT-stacked tensor under a wrong HF name) are NOT touched here —
        the caller should ensure ``w_up`` / ``w_down`` aren't in
        ``ArchSpec.layer`` to avoid confusing entries.
    layer_prefix
        E.g. ``"model.layers.{L}"`` or ``"model.language_model.layers.{L}"``.
    w_up
        FT host master tensor, shape ``(E, 2F, D)`` chunked ``[up; gate]``
        along the 2F axis.
    w_down
        FT host master tensor, shape ``(E, D, F)`` (matches HF's
        ``down_proj`` orientation directly).
    fused
        True → emit ``mlp.experts.gate_up_proj`` (E, 2F, D) +
                ``mlp.experts.down_proj`` (E, D, F). Halves of w_up are
                swapped: HF stores ``[gate; up]``, FT stores ``[up; gate]``.
        False → emit per-expert
                ``mlp.experts.{e}.{gate,up,down}_proj.weight`` with
                shapes (F, D), (F, D), (D, F).
    """
    E, TwoF, D = w_up.shape
    assert w_down.shape == (E, D, TwoF // 2), (
        f"w_down shape {tuple(w_down.shape)} != (E={E}, D={D}, F={TwoF//2})"
    )
    F = TwoF // 2
    if fused:
        # FT [up; gate] along dim=1 -> HF [gate; up].
        up_part = w_up[:, :F, :]
        gate_part = w_up[:, F:, :]
        gate_up = torch.cat([gate_part, up_part], dim=1).contiguous()
        dst[f"{layer_prefix}.mlp.experts.gate_up_proj"] = gate_up.cpu()
        dst[f"{layer_prefix}.mlp.experts.down_proj"] = w_down.contiguous().cpu()
        return
    for e in range(E):
        # FT w_up[e]: (2F, D) chunked [up; gate]. HF: gate=(F,D), up=(F,D).
        up_e = w_up[e, :F, :].contiguous()
        gate_e = w_up[e, F:, :].contiguous()
        # FT w_down[e]: (D, F) — same as HF down_proj.
        down_e = w_down[e, :, :].contiguous()
        dst[f"{layer_prefix}.mlp.experts.{e}.gate_proj.weight"] = gate_e.cpu()
        dst[f"{layer_prefix}.mlp.experts.{e}.up_proj.weight"] = up_e.cpu()
        dst[f"{layer_prefix}.mlp.experts.{e}.down_proj.weight"] = down_e.cpu()


def emit_shared_experts(
    dst: MutableMapping[str, torch.Tensor],
    *,
    layer_prefix: str,
    w_shared_up: torch.Tensor,
    w_shared_down: torch.Tensor,
    w_shared_gate: torch.Tensor | None,
) -> None:
    """Inverse of shared-expert stacking (qwen3_next / qwen3_5_moe).

    Args
    ----
    w_shared_up
        FT host tensor, shape ``(S=1, D, 2F_s)`` with ``[up, gate]``
        cat along dim=-1.
    w_shared_down
        FT host tensor, shape ``(S=1, F_s, D)``.
    w_shared_gate
        FT host tensor, shape ``(D, S=1)`` — optional.
    """
    assert w_shared_up.shape[0] == 1, (
        f"shared-expert exporter supports S=1; got {w_shared_up.shape}"
    )
    S, D, TwoFs = w_shared_up.shape
    Fs = TwoFs // 2
    assert w_shared_down.shape == (S, Fs, D), (
        f"w_shared_down shape mismatch: {w_shared_down.shape}"
    )
    # FT (D, 2F_s) split [up | gate] -> HF gate_proj (F_s, D), up_proj (F_s, D).
    block = w_shared_up[0]                          # (D, 2F_s)
    up_block = block[:, :Fs]                        # (D, F_s)
    gate_block = block[:, Fs:]                      # (D, F_s)
    gate_hf = gate_block.T.contiguous()             # (F_s, D)
    up_hf = up_block.T.contiguous()                 # (F_s, D)
    # FT w_shared_down (1, F_s, D) -> HF down_proj (D, F_s).
    down_hf = w_shared_down[0].T.contiguous()       # (D, F_s)
    dst[f"{layer_prefix}.mlp.shared_expert.gate_proj.weight"] = gate_hf.cpu()
    dst[f"{layer_prefix}.mlp.shared_expert.up_proj.weight"] = up_hf.cpu()
    dst[f"{layer_prefix}.mlp.shared_expert.down_proj.weight"] = down_hf.cpu()
    if w_shared_gate is not None:
        # FT (D, 1) -> HF (1, D).
        sg = w_shared_gate.squeeze(-1).unsqueeze(0).contiguous()
        dst[f"{layer_prefix}.mlp.shared_expert_gate.weight"] = sg.cpu()


# ---------------------------------------------------------------------------
# (2) Linear-attn qkvz / ba unbundle.
# ---------------------------------------------------------------------------


def emit_linear_attn_unbundle(
    dst: MutableMapping[str, torch.Tensor],
    *,
    layer_prefix: str,
    w_lin_qkvz: torch.Tensor,           # (hidden, 2*key_dim + 2*value_dim)
    w_lin_ba: torch.Tensor,             # (hidden, 2*num_v_heads)
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    """Inverse of qkvz/ba bundling.

    FT layout (post-bundling):
      w_lin_qkvz : (hidden, proj_qkvz_dim) where proj_qkvz_dim = 2*key_dim +
                   2*value_dim, block-major ``[Q | K | V | Z]`` along the
                   column axis (each block laid out as (head, dim) row-major).
      w_lin_ba   : (hidden, 2*num_v_heads) block-major ``[B | A]``.

    HF target:
      in_proj_qkv : (out=key_dim*2 + value_dim, in=hidden)  flat [q | k | v]
      in_proj_z   : (out=value_dim,             in=hidden)  flat z
      in_proj_b   : (out=num_v_heads,           in=hidden)
      in_proj_a   : (out=num_v_heads,           in=hidden)
    """
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    hidden, proj_qkvz_dim = w_lin_qkvz.shape
    assert proj_qkvz_dim == 2 * key_dim + 2 * value_dim, (
        f"unbundle: proj_qkvz_dim={proj_qkvz_dim} != 2*key_dim ({key_dim}) "
        f"+ 2*value_dim ({value_dim})"
    )
    assert w_lin_ba.shape == (hidden, 2 * num_v_heads), (
        f"unbundle: w_lin_ba shape {w_lin_ba.shape} != "
        f"({hidden}, {2*num_v_heads})"
    )

    # FT layout in column blocks: [Q (key_dim) | K (key_dim) | V (value_dim) | Z (value_dim)]
    # Transpose to (out, in) HF orientation while slicing.
    bundled_T = w_lin_qkvz.T.contiguous()           # (proj_qkvz_dim, hidden)
    q_block = bundled_T[:key_dim, :]                # (key_dim, hidden)
    k_block = bundled_T[key_dim:2*key_dim, :]       # (key_dim, hidden)
    v_block = bundled_T[2*key_dim:2*key_dim+value_dim, :]   # (value_dim, hidden)
    z_block = bundled_T[2*key_dim+value_dim:, :]            # (value_dim, hidden)

    # HF in_proj_qkv is fused [Q | K | V] along axis 0.
    in_proj_qkv = torch.cat(
        [q_block, k_block, v_block], dim=0
    ).contiguous()                                   # (key_dim*2 + value_dim, hidden)
    in_proj_z = z_block.contiguous()                 # (value_dim, hidden)
    dst[f"{layer_prefix}.linear_attn.in_proj_qkv.weight"] = in_proj_qkv.cpu()
    dst[f"{layer_prefix}.linear_attn.in_proj_z.weight"] = in_proj_z.cpu()

    # ba unbundle: FT [B | A] along columns -> two (num_v, hidden) HF tensors.
    bundled_ba_T = w_lin_ba.T.contiguous()           # (2*num_v, hidden)
    in_proj_b = bundled_ba_T[:num_v_heads, :].contiguous()  # (num_v, hidden)
    in_proj_a = bundled_ba_T[num_v_heads:, :].contiguous()  # (num_v, hidden)
    dst[f"{layer_prefix}.linear_attn.in_proj_b.weight"] = in_proj_b.cpu()
    dst[f"{layer_prefix}.linear_attn.in_proj_a.weight"] = in_proj_a.cpu()


# ---------------------------------------------------------------------------
# (3) (1+w) RMSNorm shift inverse.
# ---------------------------------------------------------------------------


def subtract_one_from_norms(
    dst: MutableMapping[str, torch.Tensor],
    hf_norm_names: list[str],
) -> None:
    """Subtract 1.0 from every entry in ``dst`` whose key is in
    ``hf_norm_names``. No-op for entries not present (e.g. linear-attn
    layers don't have a self_attn.q_norm).

    This is an in-place subtract that returns CPU bf16 -> CPU bf16
    (going through fp32 to avoid bf16 rounding bias on the −1 op).
    """
    for nm in hf_norm_names:
        t = dst.get(nm)
        if t is None:
            continue
        # bf16 add of −1.0 has 1 ULP at 1.0 -> harmless; do via fp32
        # to keep the math precise.
        out = (t.float() - 1.0).to(t.dtype)
        dst[nm] = out


# ---------------------------------------------------------------------------
# (4) Drop FT-only ArchSpec entries that emitted under a wrong HF name.
# ---------------------------------------------------------------------------


def drop_keys(dst: MutableMapping[str, torch.Tensor], names: list[str]) -> None:
    """Pop every ``names`` entry from ``dst`` if present. Used when an
    ArchSpec entry exists for an FT-bundled tensor (e.g. ``w_lin_qkvz``
    under ``...in_proj_qkvz.weight``) — the export walk emits it, but
    the corresponding HF dir doesn't have such a tensor; we delete it
    and emit the unbundled per-K-head tensors instead."""
    for nm in names:
        dst.pop(nm, None)


def read_tie_word_embeddings(am) -> bool:
    """Read ``tie_word_embeddings`` from the source HF config.json that
    was loaded into ``am``. Some checkpoints in the same arch family
    are tied (Qwen3.5-2B, Llama-3.2-1B) and others are not (Qwen3.5-9B,
    Qwen3-30B). Defaults to False when the config is unavailable.
    """
    src_dir = getattr(am, "_hf_source_path", None)
    if not src_dir:
        return False
    cfg_path = os.path.join(src_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        import json
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return False
    # Multimodal Qwen3.5 wraps the LM-side config under ``text_config``;
    # tie_word_embeddings can live at top level OR under text_config
    # depending on the save's transformers version.
    if "tie_word_embeddings" in cfg:
        return bool(cfg["tie_word_embeddings"])
    text_cfg = cfg.get("text_config")
    if isinstance(text_cfg, dict) and "tie_word_embeddings" in text_cfg:
        return bool(text_cfg["tie_word_embeddings"])
    return False


__all__ = (
    "detect_fused_moe_format",
    "emit_routed_experts",
    "emit_shared_experts",
    "emit_linear_attn_unbundle",
    "subtract_one_from_norms",
    "drop_keys",
    "read_tie_word_embeddings",
)
