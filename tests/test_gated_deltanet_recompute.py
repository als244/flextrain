"""Verify GatedDeltaNetBlock partial-tier ``fwd_recompute_*`` correctness.

For each tier subset that the save-level solver might pick, we:

1. Run a normal fwd to populate ALL fields.
2. Snapshot every saved field.
3. Zero out the higher-tier fields (simulating a save level that
   dropped them).
4. Call the appropriate ``fwd_recompute_*`` helpers in the same order
   the layer's ``forward_recompute`` would.
5. Verify the recomputed fields match the original snapshot.

If any tier-recompute path is buggy, this catches it quickly without
running a full E2E training loop.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.core.activation_schema import (
    ActivationField, ActivationSchema, ActivationSlot,
)
from flextrain.nn.blocks.linear_attn import (
    GatedDeltaNetBlock, GatedDeltaNetConfig,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _build_block_and_slot(T: int):
    torch.manual_seed(0)
    cfg = GatedDeltaNetConfig(
        d_model=128, num_v_heads=8, num_k_heads=2,
        head_k_dim=32, head_v_dim=32,
        conv_kernel_size=4, rms_norm_eps=1e-6,
    )
    block = GatedDeltaNetBlock(cfg)
    schema = ActivationSchema(
        fields=block.fields() + (
            ActivationField(
                "x_inp", lambda n, d: (n, cfg.d_model), DTYPE, tier=0,
            ),
        ),
        max_tier=3,
    )
    dims = cfg.dims()
    slot_tensors = {}
    for f in schema.fields:
        shape = f.shape_fn(T, dims)
        slot_tensors[f.name] = torch.empty(shape, dtype=f.dtype, device=DEVICE)
    slot = ActivationSlot(
        schema=schema, level=schema.max_tier, tensors=slot_tensors,
    )
    weights = {
        "w_lin_qkvz": torch.randn(cfg.d_model, cfg.proj_qkvz_dim, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_lin_ba": torch.randn(cfg.d_model, cfg.proj_ba_dim, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_lin_conv": torch.randn(cfg.conv_dim, 1, cfg.conv_kernel_size, dtype=DTYPE, device=DEVICE) * 0.1,
        "w_lin_dt_bias": torch.ones(cfg.num_v_heads, dtype=DTYPE, device=DEVICE),
        "w_lin_A_log": torch.log(torch.empty(cfg.num_v_heads, dtype=DTYPE, device=DEVICE).uniform_(1, 16)),
        "w_lin_norm": torch.ones(cfg.head_v_dim, dtype=DTYPE, device=DEVICE),
        "w_lin_out": torch.randn(cfg.value_dim, cfg.d_model, dtype=DTYPE, device=DEVICE) * 0.02,
    }
    return cfg, block, slot, weights


def _snapshot(slot, fields):
    return {f: slot._tensors[f].detach().clone() for f in fields}


def _diff(name, ref, got):
    delta = (ref.float() - got.float()).abs()
    mx = float(delta.max().item())
    refmx = float(ref.abs().float().max().item())
    print(f"  {name:30s} max|Δ|={mx:.3e}  ref|max|={refmx:.3e}")
    return mx


def _tier_of_field(block, name):
    for f in block.fields():
        if f.name == name:
            return f.tier
    return 0  # x_inp tier-0


def main():
    T = 64
    cfg, block, slot, weights = _build_block_and_slot(T)
    x = torch.randn(T, cfg.d_model, dtype=DTYPE, device=DEVICE) * 0.1
    slot.x_inp.copy_(x)

    # 1. Full fwd → all fields populated.
    block.fwd(x, weights, slot, ctx=None)
    all_fields = [
        "lin_a", "lin_b", "lin_g", "lin_g_post", "lin_z",
        "lin_q", "lin_k", "lin_v", "lin_A_int", "lin_core_out",
        "lin_conv_in", "lin_post_conv_pre_silu",
    ]
    snap = _snapshot(slot, all_fields)

    # ==================================================================
    # Per-save-level test: simulate the engine dropping every field
    # with tier > save_level, then run the layer-style cascading
    # recompute, then verify all fields match the original fwd.
    # Covers save_level=0, 1, 2, 3 (the full range for max_tier=3).
    # ==================================================================
    for save_level in (3, 2, 1, 0):
        print(f"\n=== save_level={save_level} ===")

        # Reset slot to the post-fwd state, then zero out fields with
        # tier > save_level. (slot.has uses tier <= slot.level.)
        for f in all_fields:
            slot._tensors[f].copy_(snap[f])
        dropped = [f for f in all_fields if _tier_of_field(block, f) > save_level]
        for f in dropped:
            slot._tensors[f].zero_()
        slot.level = save_level
        print(f"  dropped fields (tier > {save_level}): {dropped}")

        # Mirror Qwen3NextLinearLayer.forward_recompute logic.
        if not slot.has("lin_post_conv_pre_silu"):
            block.fwd_recompute_post_conv(x, weights, slot)
        if not slot.has("lin_q"):
            block.fwd_recompute_qkv_heads(slot)
        if not slot.has("lin_core_out") or not slot.has("lin_A_int"):
            block.fwd_recompute_fla(weights, slot)

        # Restore level so tensors are accessible for diffing.
        slot.level = block.fields()[0].tier  # any tier-0 field
        # (We bypass slot.has for inspection; just diff against snapshot.)
        max_d = 0.0
        for f in all_fields:
            max_d = max(max_d, _diff(f, snap[f], slot._tensors[f]))
        # Allow ~1e-3 tolerance for bf16 reductions in the recompute path.
        assert max_d < 1e-3, f"save_level={save_level}: max|Δ|={max_d}"
        # Restore for next iteration.
        slot.level = 3

    # ==================================================================
    # Standalone unit tests for individual fwd_recompute_* methods.
    # ==================================================================
    print("\n=== Per-method (in-isolation): tier-3 ===")
    for f in ("lin_conv_in", "lin_post_conv_pre_silu", "lin_a", "lin_b", "lin_z"):
        slot._tensors[f].zero_()
    block.fwd_recompute_post_conv(x, weights, slot)
    max_d = 0.0
    for f in ("lin_conv_in", "lin_post_conv_pre_silu", "lin_a", "lin_b", "lin_z"):
        max_d = max(max_d, _diff(f, snap[f], slot._tensors[f]))
    assert max_d < 1e-3

    print("\n=== Per-method (in-isolation): tier-2 q/k/v ===")
    for f in ("lin_q", "lin_k", "lin_v"):
        slot._tensors[f].zero_()
    block.fwd_recompute_qkv_heads(slot)
    max_d = 0.0
    for f in ("lin_q", "lin_k", "lin_v"):
        max_d = max(max_d, _diff(f, snap[f], slot._tensors[f]))
    assert max_d < 1e-3

    print("\n=== Per-method (in-isolation): tier-2 FLA ===")
    for f in ("lin_core_out", "lin_A_int", "lin_g_post"):
        slot._tensors[f].zero_()
    block.fwd_recompute_fla(weights, slot)
    max_d = 0.0
    for f in ("lin_core_out", "lin_A_int", "lin_g_post"):
        max_d = max(max_d, _diff(f, snap[f], slot._tensors[f]))
    assert max_d < 1e-3

    print("\n✓ GatedDeltaNet partial-recompute parity PASSED across all save levels")


if __name__ == "__main__":
    main()
