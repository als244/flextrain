"""Backward parity test for GatedDeltaNetBlock.

Compares the block's hand-written ``bwd`` (single-autograd block over
projections + conv + gated-norm, with FLA's fwd/bwd called directly via
a custom ``torch.autograd.Function`` for the linear-attn core) against
a fully autograd-built reference forward path.

The reference path uses the FLA library's autograd-wrapped
``chunk_gated_delta_rule`` directly (not our custom Function). Both
share the same FLA kernel, so the comparison verifies our scoped
autograd graph for the surrounding ops + the custom-Function FLA wrap.
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


def _autograd_reference_loss(cfg, x, weights):
    """Reference: full autograd forward and an L2 norm loss; backprop."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    T = x.shape[0]
    qkvz = x @ weights["w_lin_qkvz"]
    ba = x @ weights["w_lin_ba"]
    q_pre, k_pre, v_pre, z = _split_qkvz(qkvz, cfg)
    b, a = _split_ba(ba, cfg)
    q_flat = q_pre.reshape(T, cfg.key_dim)
    k_flat = k_pre.reshape(T, cfg.key_dim)
    v_flat = v_pre.reshape(T, cfg.value_dim)
    conv_in = torch.cat([q_flat, k_flat, v_flat], dim=-1)
    cx = conv_in.transpose(0, 1).unsqueeze(0)
    K = cfg.conv_kernel_size
    post_conv = F.conv1d(
        cx, weights["w_lin_conv"], bias=None,
        padding=K - 1, groups=cfg.conv_dim,
    )[..., :T]
    post_conv = F.silu(post_conv).squeeze(0).transpose(0, 1).contiguous()
    q_p, k_p, v_p = torch.split(
        post_conv, [cfg.key_dim, cfg.key_dim, cfg.value_dim], dim=-1,
    )
    q_h = q_p.reshape(T, cfg.num_k_heads, cfg.head_k_dim)
    k_h = k_p.reshape(T, cfg.num_k_heads, cfg.head_k_dim)
    v_h = v_p.reshape(T, cfg.num_v_heads, cfg.head_v_dim)
    if cfg.num_v_heads // cfg.num_k_heads > 1:
        rep = cfg.num_v_heads // cfg.num_k_heads
        q_h = q_h.repeat_interleave(rep, dim=1)
        k_h = k_h.repeat_interleave(rep, dim=1)
    a_f32 = a.float()
    g = -weights["w_lin_A_log"].float().exp() * F.softplus(
        a_f32 + weights["w_lin_dt_bias"].float()
    )
    beta = b.float().sigmoid().to(x.dtype)
    o, _ = chunk_gated_delta_rule(
        q_h.unsqueeze(0), k_h.unsqueeze(0), v_h.unsqueeze(0),
        g.unsqueeze(0), beta.unsqueeze(0),
        scale=cfg.head_k_dim ** -0.5, initial_state=None,
        output_final_state=False, cu_seqlens=None,
        use_qk_l2norm_in_kernel=True,
    )
    o = o.squeeze(0)
    o_f = o.float()
    rms = (o_f * o_f).mean(dim=-1, keepdim=True).add_(cfg.rms_norm_eps).rsqrt_()
    normed = (o_f * rms).to(x.dtype)
    o_norm = normed * weights["w_lin_norm"] * F.silu(z.float()).to(x.dtype)
    y = o_norm.reshape(T, cfg.value_dim) @ weights["w_lin_out"]
    return y


def main():
    torch.manual_seed(11)
    cfg = GatedDeltaNetConfig(
        d_model=128, num_v_heads=8, num_k_heads=2,
        head_k_dim=32, head_v_dim=32,
        conv_kernel_size=4, rms_norm_eps=1e-6,
    )
    T = 64
    x = torch.randn(T, cfg.d_model, dtype=DTYPE, device=DEVICE)
    weight_specs = [
        ("w_lin_qkvz", (cfg.d_model, cfg.proj_qkvz_dim), 0.02),
        ("w_lin_ba", (cfg.d_model, cfg.proj_ba_dim), 0.02),
        ("w_lin_out", (cfg.value_dim, cfg.d_model), 0.02),
        ("w_lin_conv", (cfg.conv_dim, 1, cfg.conv_kernel_size), 0.1),
    ]
    weights_ref = {}
    for name, shape, scale in weight_specs:
        weights_ref[name] = (
            torch.randn(*shape, dtype=DTYPE, device=DEVICE) * scale
        ).requires_grad_(True)
    weights_ref["w_lin_dt_bias"] = torch.ones(
        cfg.num_v_heads, dtype=DTYPE, device=DEVICE,
    ).requires_grad_(True)
    weights_ref["w_lin_A_log"] = torch.log(
        torch.empty(cfg.num_v_heads, dtype=DTYPE, device=DEVICE).uniform_(1, 16)
    ).requires_grad_(True)
    weights_ref["w_lin_norm"] = torch.ones(
        cfg.head_v_dim, dtype=DTYPE, device=DEVICE,
    ).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)

    # Reference forward + sum-loss + backward via autograd.
    y_ref = _autograd_reference_loss(cfg, x_ref, weights_ref)
    upstream = torch.randn_like(y_ref) * 0.01
    y_ref.backward(upstream)

    ref_grads = {name: weights_ref[name].grad.clone() for name in weights_ref}
    ref_dx = x_ref.grad.clone()

    # FT block fwd + bwd.
    weights_ft = {
        name: weights_ref[name].detach().clone() for name in weights_ref
    }
    block = GatedDeltaNetBlock(cfg)
    schema = ActivationSchema(fields=block.fields(), max_tier=3)
    dims = cfg.dims()
    slot_tensors = {}
    for f in schema.fields:
        shape = f.shape_fn(T, dims)
        slot_tensors[f.name] = torch.empty(shape, dtype=f.dtype, device=DEVICE)
    # Add x_inp slot tensor (the layer would normally provide this).
    slot_tensors["x_inp"] = torch.empty(
        (T, cfg.d_model), dtype=DTYPE, device=DEVICE,
    )
    schema_with_xinp = ActivationSchema(
        fields=block.fields() + tuple([_dummy_x_inp(cfg)]), max_tier=3,
    )
    slot = ActivationSlot(
        schema=schema_with_xinp, level=schema_with_xinp.max_tier,
        tensors=slot_tensors,
    )
    slot.x_inp.copy_(x)

    # Run fwd to populate slot. Pass zero residual so y_ft == lin_out
    # (matches the autograd reference, which has no residual).
    x_resid_zero = torch.zeros_like(x)
    y_ft = block.fwd(x_resid_zero, x, weights_ft, slot, ctx=None)
    # Set up grad accumulators.
    grad_keys = [
        "g_lin_qkvz", "g_lin_ba", "g_lin_out", "g_lin_conv",
        "g_lin_dt_bias", "g_lin_A_log", "g_lin_norm",
    ]
    grads = {
        k: torch.zeros_like(weights_ft["w_" + k[2:]], dtype=torch.float32)
        for k in grad_keys
    }
    # Fwd output should match ref (sanity).
    assert (y_ft.float() - y_ref.float()).abs().max().item() < 0.02

    # Run bwd.
    dx = block.bwd(upstream, weights_ft, grads, slot, ctx=None)

    # Compare grads.
    print("=== gradient parity (ours vs autograd reference) ===")
    max_delta = 0.0
    for ref_name, grad_key in [
        ("w_lin_qkvz", "g_lin_qkvz"),
        ("w_lin_ba", "g_lin_ba"),
        ("w_lin_out", "g_lin_out"),
        ("w_lin_conv", "g_lin_conv"),
        ("w_lin_dt_bias", "g_lin_dt_bias"),
        ("w_lin_A_log", "g_lin_A_log"),
        ("w_lin_norm", "g_lin_norm"),
    ]:
        ref_g = ref_grads[ref_name].float()
        ft_g = grads[grad_key].float()
        d = (ref_g - ft_g).abs().max().item()
        m_ref = ref_g.abs().max().item()
        rel = d / (m_ref + 1e-12)
        max_delta = max(max_delta, d)
        print(
            f"  {ref_name:<18s}: |Δ|max={d:.4e}  "
            f"|ref|max={m_ref:.4e}  rel={rel:.4f}"
        )

    dx_d = (ref_dx.float() - dx.float()).abs().max().item()
    print(f"  dL/dx           : |Δ|max={dx_d:.4e}")

    assert max_delta < 0.05, f"grad mismatch: max |Δ|={max_delta:.4f}"
    assert dx_d < 0.05, f"dx mismatch: |Δ|={dx_d:.4f}"
    print("\n✓ GatedDeltaNet bwd parity PASSED")


def _dummy_x_inp(cfg):
    """Stub ActivationField so the test slot has an x_inp tensor (the
    enclosing layer normally adds this to the schema)."""
    from flextrain.core.activation_schema import ActivationField
    return ActivationField(
        "x_inp", lambda n, d: (n, cfg.d_model),
        DTYPE, tier=0,
    )


if __name__ == "__main__":
    main()
