"""Investigate the 67% naive-vs-kernel g_q / g_k disagreement.

The parity test in test_llama_parity.py shows:

    grad g_q   : naive/orig 6.657e-01   orig/ft 0.000e+00
    grad g_k   : naive/orig 6.345e-01   orig/ft 0.000e+00
    grad g_v   : naive/orig 3.287e-02   orig/ft 0.000e+00

That's ~20x larger error for Q/K than V. I hand-waved it away as
"flash-attn bf16 vs fp32 naive". This test actually measures that
hypothesis by running the NAIVE path in bf16 too (matching orig's
precision), and by isolating RoPE -- the most bf16-sensitive step in
the Q/K gradient path.

If the bf16-naive baseline matches orig to within bf16 tolerance (<1e-1),
the hand-wave was right. If it doesn't, we have a real bug.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
ORIG = os.path.join(ROOT, "orig")
if ORIG not in sys.path:
    sys.path.insert(0, ORIG)


DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Minimal ATTENTION-ONLY test: isolate the block that produces g_q / g_k.
# ---------------------------------------------------------------------------


def _rope(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """RoPE with the PAIR-INTERLEAVE convention orig's kernel uses.

    Orig's kernel (``orig/awsm_transformer/ops/rope.py:38-48``) pairs
    ``x[..., 2i]`` with ``x[..., 2i+1]`` -- NOT the halved-split convention
    (``x[..., :D/2]`` with ``x[..., D/2:]``) that Llama's HF implementation
    and GPT-NeoX popularized.

    Same mathematical frequency spectrum, different tensor-element
    assignment. If your reference uses one and the kernel uses the other,
    g_q / g_k will diverge by ~20% in relative norm (but g_v -- which
    bypasses RoPE -- stays fine). That's exactly what the earlier parity
    test exhibited.
    """
    T, H, D = x.shape
    assert D % 2 == 0
    half = D // 2
    p = pos.view(-1, 1, 1).float()
    # freqs in pair-index space: exponent = 2i / D for i in 0..D/2
    exponent = 2.0 * torch.arange(0, half, device=x.device).float() / D
    inv_freq = theta ** (-exponent)  # (D/2,)
    angles = p * inv_freq  # (T, 1, D/2)
    cos = angles.cos()
    sin = angles.sin()
    x_fp = x.float()
    even = x_fp[..., 0::2]  # x[..., 2i]
    odd = x_fp[..., 1::2]   # x[..., 2i+1]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    # Re-interleave back to original layout.
    out = torch.empty_like(x_fp)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out.to(x.dtype)


def naive_attention(
    x: torch.Tensor,  # (T, d_model)
    w_q, w_k, w_v, w_o,
    pos: torch.Tensor,
    n_heads: int,
    n_kv: int,
    head_dim: int,
    rope_base: float,
    *,
    internal_dtype: torch.dtype,
) -> torch.Tensor:
    """Pure-PyTorch attention path (no residual, no norms, no FFN).

    ``internal_dtype`` controls the precision of the softmax and matmuls
    INSIDE this function. Inputs/outputs always come in as the dtype the
    weights use (bf16). This is the knob that isolates bf16 vs fp32
    attention.
    """
    T = x.shape[0]
    d_model = x.shape[1]

    # Q/K/V projections -- keep in weight dtype so the projection matches
    # orig's bf16 matmul exactly.
    xq = (x @ w_q).view(T, n_heads, head_dim)
    xk = (x @ w_k).view(T, n_kv, head_dim)
    xv = (x @ w_v).view(T, n_kv, head_dim)

    # RoPE on Q / K. Orig does this in bf16 in-place. Naive does fp32 inside
    # and casts back -- that's the first place bf16 loses precision.
    rope_q = _rope(xq, pos, rope_base)
    rope_k = _rope(xk, pos, rope_base)

    # SDPA: cast to internal_dtype, run, cast back.
    q_ = rope_q.to(internal_dtype).transpose(0, 1)  # (H, T, D)
    if n_kv != n_heads:
        rep = n_heads // n_kv
        k_ = rope_k.repeat_interleave(rep, dim=1).to(internal_dtype).transpose(0, 1)
        v_ = xv.repeat_interleave(rep, dim=1).to(internal_dtype).transpose(0, 1)
    else:
        k_ = rope_k.to(internal_dtype).transpose(0, 1)
        v_ = xv.to(internal_dtype).transpose(0, 1)
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale
    mask = torch.triu(
        torch.full((T, T), float("-inf"), device=x.device, dtype=internal_dtype),
        diagonal=1,
    )
    scores = scores + mask
    probs = torch.softmax(scores.float(), dim=-1).to(internal_dtype)
    out = torch.matmul(probs, v_)
    out = out.transpose(0, 1).to(x.dtype).contiguous()

    # O-projection.
    return (out.reshape(T, -1) @ w_o)


def run_naive_with_autograd(
    x, w_q, w_k, w_v, w_o, pos, dy,
    *, n_heads, n_kv, head_dim, rope_base, internal_dtype,
):
    x_g = x.detach().clone().requires_grad_(True)
    wq_g = w_q.detach().clone().requires_grad_(True)
    wk_g = w_k.detach().clone().requires_grad_(True)
    wv_g = w_v.detach().clone().requires_grad_(True)
    wo_g = w_o.detach().clone().requires_grad_(True)
    y = naive_attention(
        x_g, wq_g, wk_g, wv_g, wo_g, pos,
        n_heads=n_heads, n_kv=n_kv, head_dim=head_dim, rope_base=rope_base,
        internal_dtype=internal_dtype,
    )
    y.backward(dy)
    return {
        "y": y.detach().clone(),
        "g_q": wq_g.grad.detach().clone(),
        "g_k": wk_g.grad.detach().clone(),
        "g_v": wv_g.grad.detach().clone(),
        "g_o": wo_g.grad.detach().clone(),
    }


def run_orig_attention_only(
    x, w_q, w_k, w_v, w_o, pos, dy,
    *, n_heads, n_kv, head_dim, rope_base, seq_len, d_model,
):
    """Call orig's ops directly in the exact order the layer calls them,
    but without the norm / FFN / residual.
    """
    from flextrain.ops import (
        flextrain_attention_bwd, flextrain_attention_fwd, flextrain_rope_bwd, flextrain_rope_fwd,
    )

    dtype = x.dtype

    # Zero grads.
    g_q = torch.zeros_like(w_q)
    g_k = torch.zeros_like(w_k)
    g_v = torch.zeros_like(w_v)
    g_o = torch.zeros_like(w_o)

    # Q/K/V projections.
    xq = (x @ w_q).view(-1, n_heads, head_dim).contiguous()
    xk = (x @ w_k).view(-1, n_kv, head_dim).contiguous()
    xv = (x @ w_v).view(-1, n_kv, head_dim).contiguous()

    # RoPE fwd in place.
    thetas = torch.tensor([rope_base], dtype=torch.float32, device=x.device)
    seq_pos = pos.to(torch.int32).reshape(-1, 1)
    flextrain_rope_fwd([xq, xk], seq_pos, thetas)

    # Attention fwd.
    attn_result = torch.empty(seq_len, n_heads, head_dim, device=x.device, dtype=dtype)
    lse = torch.empty(n_heads, seq_len, device=x.device, dtype=torch.float32)
    q_off = torch.tensor([0, seq_len], device=x.device, dtype=torch.int32)
    k_off = torch.tensor([0, seq_len], device=x.device, dtype=torch.int32)
    q_len = torch.tensor([seq_len], device=x.device, dtype=torch.int32)
    k_len = torch.tensor([seq_len], device=x.device, dtype=torch.int32)
    flextrain_attention_fwd(
        xq, xk, xv, attn_result, lse,
        q_off, k_off, q_len, k_len, seq_len, seq_len,
        causal=True, window_size=(-1, 0),
    )

    # O-projection (no residual here — this test isolates attention).
    y = (attn_result.view(seq_len, -1)) @ w_o

    # ---- backward ----
    # dy -> g_o, dx_up_attn
    torch.addmm(g_o, attn_result.view(seq_len, -1).T, dy, out=g_o)
    dx_up_attn = (dy @ w_o.T).view(seq_len, n_heads, head_dim).contiguous()

    # flash-attn backward.
    dq = torch.zeros_like(dx_up_attn)
    dk = torch.zeros(seq_len, n_kv, head_dim, device=x.device, dtype=dtype)
    dv = torch.zeros(seq_len, n_kv, head_dim, device=x.device, dtype=dtype)
    flextrain_attention_bwd(
        dx_up_attn, xq, xk, xv, attn_result, lse,
        dq, dk, dv,
        q_off, k_off, q_len, k_len, seq_len, seq_len,
        causal=True, window_size=(-1, 0),
    )

    # RoPE backward in place.
    flextrain_rope_bwd([dq, dk], seq_pos, thetas)

    # Project gradients back through W_Q / W_K / W_V (same as layer does).
    torch.addmm(g_q, x.T, dq.view(seq_len, -1), out=g_q)
    torch.addmm(g_k, x.T, dk.view(seq_len, -1), out=g_k)
    torch.addmm(g_v, x.T, dv.view(seq_len, -1), out=g_v)

    return {"y": y, "g_q": g_q, "g_k": g_k, "g_v": g_v, "g_o": g_o}


def _rel(a, b):
    a = a.float()
    b = b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-6)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Needs CUDA.")

    torch.manual_seed(0)
    d_model, n_heads, n_kv, head_dim = 128, 4, 2, 32
    seq_len = 64
    rope_base = 500000.0

    gen = torch.Generator(device=DEVICE).manual_seed(0)
    def rnd(*shape):
        return (torch.randn(*shape, generator=gen, device=DEVICE) * 0.02).to(torch.bfloat16)

    x = rnd(seq_len, d_model)
    w_q = rnd(d_model, n_heads * head_dim)
    w_k = rnd(d_model, n_kv * head_dim)
    w_v = rnd(d_model, n_kv * head_dim)
    w_o = rnd(n_heads * head_dim, d_model)
    pos = torch.arange(seq_len, device=DEVICE, dtype=torch.int32)

    dy = (torch.randn(seq_len, d_model, generator=gen, device=DEVICE) * 0.02).to(torch.bfloat16)

    # Three runs of naive autograd at different internal precisions:
    r_fp32 = run_naive_with_autograd(
        x, w_q, w_k, w_v, w_o, pos, dy,
        n_heads=n_heads, n_kv=n_kv, head_dim=head_dim, rope_base=rope_base,
        internal_dtype=torch.float32,
    )
    r_bf16 = run_naive_with_autograd(
        x, w_q, w_k, w_v, w_o, pos, dy,
        n_heads=n_heads, n_kv=n_kv, head_dim=head_dim, rope_base=rope_base,
        internal_dtype=torch.bfloat16,
    )
    r_orig = run_orig_attention_only(
        x, w_q, w_k, w_v, w_o, pos, dy,
        n_heads=n_heads, n_kv=n_kv, head_dim=head_dim, rope_base=rope_base,
        seq_len=seq_len, d_model=d_model,
    )

    print(f"\n  seq_len={seq_len}, n_heads={n_heads}, n_kv={n_kv}, head_dim={head_dim}")
    print(f"  naive(fp32_sdpa) vs orig:")
    for k in ("y", "g_q", "g_k", "g_v", "g_o"):
        print(f"    {k:5s}: {_rel(r_fp32[k], r_orig[k]):.4e}")
    print(f"  naive(bf16_sdpa) vs orig:")
    for k in ("y", "g_q", "g_k", "g_v", "g_o"):
        print(f"    {k:5s}: {_rel(r_bf16[k], r_orig[k]):.4e}")
    print(f"  naive(fp32) vs naive(bf16):")
    for k in ("y", "g_q", "g_k", "g_v", "g_o"):
        print(f"    {k:5s}: {_rel(r_fp32[k], r_bf16[k]):.4e}")


if __name__ == "__main__":
    main()
