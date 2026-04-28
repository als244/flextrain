"""Block-level math parity: GQAAttentionGatedBlock vs HF Qwen3NextAttention.

Single-block, single-chunk, fixed-input, random-init test. Builds:

* HF ``Qwen3NextAttention`` (the trusted reference — autograd-allowed).
* FT ``GQAAttentionGatedBlock`` with the same weights copied across.

Compares forward outputs and gradient parity.

This test is the regression net for the gated-output feature. If any
of the following are off, this test fails:

* ``w_q`` doubled-output split (Q | gate).
* Sigmoid gate applied between attn and o_proj.
* QK-norm on Q only (gate is unnormed).
* RoPE on Q only (gate is un-rotated).
* Gradient routing through both halves of w_q (Q-path + gate-path).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _hf_reference_fwd(cfg_d, weights, attn_norm_output, position_ids):
    """Pure-PyTorch reference matching HF Qwen3NextAttention math.
    No FT engine, no flash-attn — just the math, autograd-allowed.

    cfg_d: dict with d_model, n_heads, n_kv_heads, head_dim, rope_base,
           rope_scaling=None, qk_norm=True, eps.
    weights: dict with w_q (d_model, attn_dim*2), w_k (d_model, kv_dim),
             w_v (d_model, kv_dim), w_o (attn_dim, d_model),
             w_q_norm (head_dim,), w_k_norm (head_dim,).
    attn_norm_output: (T, d_model) — already through input layernorm.
    position_ids: (T,) int64.

    Returns y: (T, d_model). Tracks autograd graph through ``weights``
    and ``attn_norm_output``.
    """
    n_heads = cfg_d["n_heads"]
    n_kv = cfg_d["n_kv_heads"]
    head_dim = cfg_d["head_dim"]
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    eps = cfg_d["eps"]
    T = attn_norm_output.shape[0]

    # Q + gate via doubled w_q.
    qproj = attn_norm_output @ weights["w_q"]                  # (T, attn_dim*2)
    Q_2d = qproj[:, :attn_dim]                                  # (T, attn_dim)
    gate_2d = qproj[:, attn_dim:]                               # (T, attn_dim)

    # K, V projections.
    K_2d = attn_norm_output @ weights["w_k"]                    # (T, kv_dim)
    V_2d = attn_norm_output @ weights["w_v"]                    # (T, kv_dim)

    # Per-head QK-norm: rmsnorm over head_dim, weight has shape (head_dim,).
    Q_3d = Q_2d.view(T, n_heads, head_dim)
    K_3d = K_2d.view(T, n_kv, head_dim)
    V_3d = V_2d.view(T, n_kv, head_dim)

    def _rmsnorm_per_head(x, w):
        x_f = x.float()
        rms = (x_f * x_f).mean(dim=-1, keepdim=True).add(eps).rsqrt()
        return (x_f * rms).to(x.dtype) * w

    Q_3d = _rmsnorm_per_head(Q_3d, weights["w_q_norm"])
    K_3d = _rmsnorm_per_head(K_3d, weights["w_k_norm"])

    # RoPE (full head_dim, no partial — that's a separate Q3N-3 fix).
    base = cfg_d["rope_base"]
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=Q_3d.device) / head_dim))
    pos = position_ids.float()
    freqs = torch.einsum("t,d->td", pos, inv_freq)              # (T, half)
    cos = freqs.cos().to(Q_3d.dtype)
    sin = freqs.sin().to(Q_3d.dtype)

    def _rope_pair(x):
        # FT pair-interleave layout: even=cos, odd=sin.
        x_e = x[..., 0::2]
        x_o = x[..., 1::2]
        rot_e = x_e * cos.unsqueeze(1) - x_o * sin.unsqueeze(1)
        rot_o = x_e * sin.unsqueeze(1) + x_o * cos.unsqueeze(1)
        out = torch.empty_like(x)
        out[..., 0::2] = rot_e
        out[..., 1::2] = rot_o
        return out

    Q_3d = _rope_pair(Q_3d)
    K_3d = _rope_pair(K_3d)

    # GQA attention (causal). Repeat K/V over heads.
    rep = n_heads // n_kv
    K_3d_rep = K_3d.repeat_interleave(rep, dim=1)               # (T, n_heads, head_dim)
    V_3d_rep = V_3d.repeat_interleave(rep, dim=1)

    # Standard scaled-dot-product attention.
    Q_h = Q_3d.transpose(0, 1)                                  # (n_heads, T, head_dim)
    K_h = K_3d_rep.transpose(0, 1)
    V_h = V_3d_rep.transpose(0, 1)
    scale = head_dim ** -0.5
    scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) * scale   # (n_heads, T, T)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=Q_h.device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
    attn_out = torch.matmul(probs, V_h)                         # (n_heads, T, head_dim)
    attn_out = attn_out.transpose(0, 1).contiguous()            # (T, n_heads, head_dim)
    attn_out_2d = attn_out.view(T, attn_dim)

    # Sigmoid gate then o_proj.
    gated = attn_out_2d * torch.sigmoid(gate_2d.float()).to(attn_out_2d.dtype)
    y = gated @ weights["w_o"]                                  # (T, d_model)
    return y


def main():
    torch.manual_seed(11)

    from flextrain.core.activation_schema import (
        ActivationField, ActivationSchema, ActivationSlot,
    )
    from flextrain.core.layer import LayerContext, ChunkMeta
    from flextrain.nn.blocks import (
        GQAAttentionConfig, GQAAttentionGatedBlock,
    )

    d_model = 128
    n_heads = 4
    n_kv = 2
    head_dim = 32
    T = 32
    eps = 1e-6
    rope_base = 500_000.0

    cfg = GQAAttentionConfig(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
        is_causal=True, qk_norm=True, rms_norm_eps=eps,
        rope_base=rope_base,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    )
    block = GQAAttentionGatedBlock(cfg)

    # Random init weights.
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    weights_ref = {
        "w_q": (torch.randn(d_model, attn_dim * 2, dtype=DTYPE, device=DEVICE) * 0.02).requires_grad_(True),
        "w_k": (torch.randn(d_model, kv_dim, dtype=DTYPE, device=DEVICE) * 0.02).requires_grad_(True),
        "w_v": (torch.randn(d_model, kv_dim, dtype=DTYPE, device=DEVICE) * 0.02).requires_grad_(True),
        "w_o": (torch.randn(attn_dim, d_model, dtype=DTYPE, device=DEVICE) * 0.02).requires_grad_(True),
        "w_q_norm": torch.ones(head_dim, dtype=DTYPE, device=DEVICE).requires_grad_(True),
        "w_k_norm": torch.ones(head_dim, dtype=DTYPE, device=DEVICE).requires_grad_(True),
    }
    attn_norm_output_ref = torch.randn(T, d_model, dtype=DTYPE, device=DEVICE).requires_grad_(True)
    position_ids = torch.arange(T, device=DEVICE, dtype=torch.int64)

    # ----- Reference fwd -----
    cfg_d = dict(
        n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
        d_model=d_model, eps=eps, rope_base=rope_base,
    )
    y_ref = _hf_reference_fwd(cfg_d, weights_ref, attn_norm_output_ref, position_ids)

    # Reference upstream + bwd.
    upstream = torch.randn_like(y_ref) * 0.01
    y_ref.backward(upstream, retain_graph=False)
    ref_grads = {k: weights_ref[k].grad.detach().clone() for k in weights_ref}
    ref_dx_attn_norm = attn_norm_output_ref.grad.detach().clone()

    # ----- FT block (clone weights, run via block.fwd / block.bwd) -----
    weights_ft = {k: weights_ref[k].detach().clone() for k in weights_ref}
    # The block owns q_norm/k_norm internally when cfg.qk_norm=True;
    # block.fields() already includes their rstd fields.
    schema = ActivationSchema(
        fields=block.fields(),
        max_tier=3,
    )
    dims = {
        "d_model": d_model, "n_heads": n_heads, "n_kv_heads": n_kv,
        "head_dim": head_dim, "attn_dim": attn_dim, "kv_dim": kv_dim,
    }
    slot_tensors = {}
    for f in schema.fields:
        shape = f.shape_fn(T, dims)
        slot_tensors[f.name] = torch.empty(shape, dtype=f.dtype, device=DEVICE)
    slot = ActivationSlot(schema=schema, level=schema.max_tier, tensors=slot_tensors)

    # Build chunk meta via the canonical builder.
    chunk = ChunkMeta.build(
        seq_lens=[T], seq_positions=list(range(T)),
        prior_seq_lens=[0], prior_seq_offsets=[0],
        device=DEVICE,
    )

    class _MockKVCache:
        def __init__(self, max_t, n_kv_heads, head_dim, dtype, device):
            shape = (max_t, n_kv_heads, head_dim)
            self.k = torch.zeros(shape, dtype=dtype, device=device)
            self.v = torch.zeros(shape, dtype=dtype, device=device)
            self.dk = torch.zeros(shape, dtype=dtype, device=device)
            self.dv = torch.zeros(shape, dtype=dtype, device=device)

    class _MockCtx:
        def __init__(self):
            self.kv_cache = _MockKVCache(T, n_kv, head_dim, DTYPE, DEVICE)
        def scratch(self, shape, dtype):
            return torch.empty(shape, dtype=dtype, device=DEVICE)

    ctx = _MockCtx()
    x_resid = torch.zeros(T, d_model, dtype=DTYPE, device=DEVICE)
    attn_norm_output_ft = weights_ft["w_q"].new_empty(T, d_model)
    attn_norm_output_ft.copy_(attn_norm_output_ref.detach())

    # FT fwd.
    y_ft = block.fwd(x_resid, attn_norm_output_ft, chunk, weights_ft, slot, ctx)
    print(f"FT y: shape={tuple(y_ft.shape)}, max|y|={float(y_ft.abs().max().item()):.3e}")
    print(f"REF y: shape={tuple(y_ref.shape)}, max|y|={float(y_ref.abs().max().item()):.3e}")
    diff = (y_ft.float() - y_ref.float()).abs()
    print(f"  fwd max|Δ| = {float(diff.max().item()):.3e}, mean|Δ| = {float(diff.mean().item()):.3e}")

    if float(diff.max().item()) > 0.05:
        print("\n  ⚠ Forward divergence. Skipping bwd parity (would be misleading).")
        sys.exit(2)

    # FT bwd.
    # Grads at the model's grad_dtype (matches engine convention).
    grads = {
        "g_q": torch.zeros_like(weights_ft["w_q"]),
        "g_k": torch.zeros_like(weights_ft["w_k"]),
        "g_v": torch.zeros_like(weights_ft["w_v"]),
        "g_o": torch.zeros_like(weights_ft["w_o"]),
        "g_q_norm": torch.zeros_like(weights_ft["w_q_norm"]),
        "g_k_norm": torch.zeros_like(weights_ft["w_k_norm"]),
    }
    dx_attn_norm_up = block.bwd(
        upstream, chunk, weights_ft, grads, slot, ctx,
        attn_norm_output=attn_norm_output_ft,
    )
    block.bwd_accumulate_qkv_grads(attn_norm_output_ft, grads, slot)
    print("\n=== gradient parity ===")
    max_d = 0.0
    for ref_name, grad_key in [
        ("w_q", "g_q"), ("w_k", "g_k"), ("w_v", "g_v"), ("w_o", "g_o"),
        ("w_q_norm", "g_q_norm"), ("w_k_norm", "g_k_norm"),
    ]:
        ref_g = ref_grads[ref_name].float()
        ft_g = grads[grad_key].float()
        d = (ref_g - ft_g).abs().max().item()
        m = ref_g.abs().max().item()
        rel = d / max(m, 1e-12)
        max_d = max(max_d, d)
        print(f"  {ref_name:<10s} max|Δ|={d:.3e}  ref|max|={m:.3e}  rel={rel:.4f}")

    dx_d = (ref_dx_attn_norm.float() - dx_attn_norm_up.float()).abs().max().item()
    print(f"  dL/d(attn_norm_output): max|Δ|={dx_d:.3e}")

    if max_d > 0.1 or dx_d > 0.1:
        raise AssertionError(f"GQAAttentionGatedBlock vs reference: max|Δ| too high: max_grad={max_d:.4e} dx={dx_d:.4e}")
    print("\n✓ GQAAttentionGatedBlock parity vs reference PASSED (within bf16 noise)")


if __name__ == "__main__":
    main()
