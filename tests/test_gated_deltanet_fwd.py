"""Forward-only parity test for the Gated DeltaNet block.

Verifies that :class:`GatedDeltaNetBlock.fwd` produces the same output
as the HF Qwen3-Next ``Qwen3NextGatedDeltaNet.forward`` reference
implementation (reproduced inline below in pure PyTorch — uses FLA's
``chunk_gated_delta_rule`` *with* autograd as the trusted reference).

Layer of trust:
* ``chunk_gated_delta_rule_fwd`` (called directly by FT) — same kernel
  the autograd ``chunk_gated_delta_rule`` calls inside its
  ``Function.apply``. So the linear-attention math is identical.
* The differences exercised by this test are: QKVZ/BA split shaping,
  depthwise causal conv1d application, gate computation
  (``-exp(A_log) * softplus(a + dt_bias)``), and gated-RMSNorm.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.core.activation_schema import ActivationSchema, ActivationSlot
from flextrain.nn.blocks.linear_attn import (
    GatedDeltaNetBlock, GatedDeltaNetConfig, _split_qkvz, _split_ba,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _hf_reference_fwd(
    cfg: GatedDeltaNetConfig,
    x: torch.Tensor,            # (T, d_model)
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Pure-PyTorch reference matching the HF Qwen3Next implementation
    (autograd-allowed; this is just the trusted reference)."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    T = x.shape[0]
    H = cfg.num_k_heads
    HV = cfg.num_v_heads
    hk = cfg.head_k_dim
    hv = cfg.head_v_dim

    qkvz = x @ weights["w_lin_qkvz"]
    ba = x @ weights["w_lin_ba"]
    q_pre, k_pre, v_pre, z = _split_qkvz(qkvz, cfg)
    b, a = _split_ba(ba, cfg)

    q_flat = q_pre.reshape(T, cfg.key_dim)
    k_flat = k_pre.reshape(T, cfg.key_dim)
    v_flat = v_pre.reshape(T, cfg.value_dim)
    conv_in = torch.cat([q_flat, k_flat, v_flat], dim=-1)

    K = cfg.conv_kernel_size
    cx = conv_in.transpose(0, 1).unsqueeze(0)
    post_conv = F.conv1d(
        cx, weights["w_lin_conv"], bias=None,
        padding=K - 1, groups=cfg.conv_dim,
    )[..., :T]
    post_conv = F.silu(post_conv).squeeze(0).transpose(0, 1).contiguous()
    q_p, k_p, v_p = torch.split(
        post_conv, [cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1,
    )
    q_h = q_p.reshape(T, H, hk)
    k_h = k_p.reshape(T, H, hk)
    v_h = v_p.reshape(T, HV, hv)
    if HV // H > 1:
        rep = HV // H
        q_h = q_h.repeat_interleave(rep, dim=1)
        k_h = k_h.repeat_interleave(rep, dim=1)

    a_f32 = a.float()
    A_log = weights["w_lin_A_log"].float()
    dt_bias = weights["w_lin_dt_bias"].float()
    g = -A_log.exp() * F.softplus(a_f32 + dt_bias)
    beta = b.float().sigmoid().to(x.dtype)

    o, _ = chunk_gated_delta_rule(
        q_h.unsqueeze(0), k_h.unsqueeze(0), v_h.unsqueeze(0),
        g.unsqueeze(0), beta.unsqueeze(0),
        scale=hk ** -0.5, initial_state=None,
        output_final_state=False, cu_seqlens=None,
        use_qk_l2norm_in_kernel=False,
    )
    o = o.squeeze(0)  # (T, HV, hv)

    # Gated RMSNorm.
    o_f = o.float()
    rms = (o_f * o_f).mean(dim=-1, keepdim=True).add_(cfg.rms_norm_eps).rsqrt_()
    normed = (o_f * rms).to(x.dtype)
    o_norm = normed * weights["w_lin_norm"] * F.silu(z.float()).to(x.dtype)

    return o_norm.reshape(T, cfg.value_dim) @ weights["w_lin_out"]


def main():
    torch.manual_seed(7)
    cfg = GatedDeltaNetConfig(
        d_model=128, num_v_heads=8, num_k_heads=2,
        head_k_dim=32, head_v_dim=32,
        conv_kernel_size=4, rms_norm_eps=1e-6,
    )
    T = 64
    x = torch.randn(T, cfg.d_model, dtype=DTYPE, device=DEVICE)
    weights = {
        "w_lin_qkvz": torch.randn(
            cfg.d_model, cfg.proj_qkvz_dim, dtype=DTYPE, device=DEVICE,
        ) * 0.02,
        "w_lin_ba": torch.randn(
            cfg.d_model, cfg.proj_ba_dim, dtype=DTYPE, device=DEVICE,
        ) * 0.02,
        "w_lin_out": torch.randn(
            cfg.value_dim, cfg.d_model, dtype=DTYPE, device=DEVICE,
        ) * 0.02,
        "w_lin_conv": torch.randn(
            cfg.conv_dim, 1, cfg.conv_kernel_size, dtype=DTYPE, device=DEVICE,
        ) * 0.1,
        "w_lin_dt_bias": torch.ones(cfg.num_v_heads, dtype=DTYPE, device=DEVICE),
        "w_lin_A_log": torch.log(
            torch.empty(cfg.num_v_heads, dtype=DTYPE, device=DEVICE).uniform_(1, 16)
        ),
        "w_lin_norm": torch.ones(cfg.head_v_dim, dtype=DTYPE, device=DEVICE),
    }

    # Build the block + an in-memory activation slot.
    block = GatedDeltaNetBlock(cfg)
    schema = ActivationSchema(fields=block.fields(), max_tier=3)
    dims = cfg.dims()
    slot_tensors = {}
    for f in schema.fields:
        shape = f.shape_fn(T, dims)
        slot_tensors[f.name] = torch.empty(shape, dtype=f.dtype, device=DEVICE)
    slot = ActivationSlot(schema=schema, level=schema.max_tier, tensors=slot_tensors)

    # FT block fwd. Block doesn't use ctx (no kv_cache, no scratch).
    # Pass zero residual so y_ft == lin_out (matches reference).
    x_resid_zero = torch.zeros_like(x)
    y_ft = block.fwd(x_resid_zero, x, weights, slot, ctx=None)

    # Reference fwd.
    y_ref = _hf_reference_fwd(cfg, x, weights)

    # Compare.
    delta = (y_ft.float() - y_ref.float()).abs()
    print(f"  y_ft  shape={tuple(y_ft.shape)}  range=[{y_ft.min().item():.3f}, {y_ft.max().item():.3f}]")
    print(f"  y_ref shape={tuple(y_ref.shape)}  range=[{y_ref.min().item():.3f}, {y_ref.max().item():.3f}]")
    print(f"  max |Δ| = {delta.max().item():.6f}  mean |Δ| = {delta.mean().item():.6f}")
    assert delta.max().item() < 0.02, (
        f"GatedDeltaNet fwd diverges from HF reference: "
        f"max |Δ| = {delta.max().item():.4f} (expected < 0.02)"
    )
    print("\n✓ GatedDeltaNet forward parity PASSED (within bf16 noise)")


if __name__ == "__main__":
    main()
