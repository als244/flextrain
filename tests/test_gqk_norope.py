"""Isolate RoPE: run attention WITHOUT RoPE and see if g_q / g_k still
disagree at ~1e-3 or if we can get to ~1e-4.

Expected outcomes:

  * If no-RoPE achieves ~1e-4 (near zero): the ~1e-3 in the parity test
    is genuine RoPE-induced bf16 noise, and the naive reference is
    correct.
  * If no-RoPE still shows ~1e-3: the kernel and naive path disagree on
    attention itself, and we need to investigate flash-attn further.
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


def naive_attention_no_rope(
    x, w_q, w_k, w_v, w_o, n_heads, n_kv, head_dim,
    *, internal_dtype,
):
    """Attention without RoPE, for Q/K gradient comparison isolated from
    the RoPE convention."""
    T = x.shape[0]
    xq = (x @ w_q).view(T, n_heads, head_dim)
    xk = (x @ w_k).view(T, n_kv, head_dim)
    xv = (x @ w_v).view(T, n_kv, head_dim)

    if n_kv != n_heads:
        rep = n_heads // n_kv
        xk_rep = xk.repeat_interleave(rep, dim=1)
        xv_rep = xv.repeat_interleave(rep, dim=1)
    else:
        xk_rep = xk
        xv_rep = xv

    q_ = xq.to(internal_dtype).transpose(0, 1)
    k_ = xk_rep.to(internal_dtype).transpose(0, 1)
    v_ = xv_rep.to(internal_dtype).transpose(0, 1)
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
    return (out.reshape(T, -1) @ w_o)


def orig_attention_no_rope(
    x, w_q, w_k, w_v, w_o, n_heads, n_kv, head_dim, seq_len, dy,
):
    from flextrain.ops import flextrain_attention_bwd, flextrain_attention_fwd
    dtype = x.dtype
    g_q = torch.zeros_like(w_q)
    g_k = torch.zeros_like(w_k)
    g_v = torch.zeros_like(w_v)
    g_o = torch.zeros_like(w_o)

    xq = (x @ w_q).view(-1, n_heads, head_dim).contiguous()
    xk = (x @ w_k).view(-1, n_kv, head_dim).contiguous()
    xv = (x @ w_v).view(-1, n_kv, head_dim).contiguous()

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
    y = (attn_result.view(seq_len, -1)) @ w_o

    torch.addmm(g_o, attn_result.view(seq_len, -1).T, dy, out=g_o)
    dx_up = (dy @ w_o.T).view(seq_len, n_heads, head_dim).contiguous()

    dq = torch.zeros_like(dx_up)
    dk = torch.zeros(seq_len, n_kv, head_dim, device=x.device, dtype=dtype)
    dv = torch.zeros(seq_len, n_kv, head_dim, device=x.device, dtype=dtype)
    flextrain_attention_bwd(
        dx_up, xq, xk, xv, attn_result, lse,
        dq, dk, dv,
        q_off, k_off, q_len, k_len, seq_len, seq_len,
        causal=True, window_size=(-1, 0),
    )
    torch.addmm(g_q, x.T, dq.view(seq_len, -1), out=g_q)
    torch.addmm(g_k, x.T, dk.view(seq_len, -1), out=g_k)
    torch.addmm(g_v, x.T, dv.view(seq_len, -1), out=g_v)
    return {"y": y, "g_q": g_q, "g_k": g_k, "g_v": g_v, "g_o": g_o}


def run_naive_autograd(x, w_q, w_k, w_v, w_o, dy, *, n_heads, n_kv, head_dim, internal_dtype):
    x_g = x.detach().clone().requires_grad_(True)
    wq_g = w_q.detach().clone().requires_grad_(True)
    wk_g = w_k.detach().clone().requires_grad_(True)
    wv_g = w_v.detach().clone().requires_grad_(True)
    wo_g = w_o.detach().clone().requires_grad_(True)
    y = naive_attention_no_rope(
        x_g, wq_g, wk_g, wv_g, wo_g, n_heads, n_kv, head_dim,
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


def _rel(a, b):
    a = a.float()
    b = b.float()
    return (a - b).norm().item() / (b.norm().item() + 1e-6)


def main() -> None:
    torch.manual_seed(0)
    d_model, n_heads, n_kv, head_dim = 128, 4, 2, 32
    seq_len = 64

    gen = torch.Generator(device=DEVICE).manual_seed(0)
    def rnd(*shape):
        return (torch.randn(*shape, generator=gen, device=DEVICE) * 0.02).to(torch.bfloat16)

    x = rnd(seq_len, d_model)
    w_q = rnd(d_model, n_heads * head_dim)
    w_k = rnd(d_model, n_kv * head_dim)
    w_v = rnd(d_model, n_kv * head_dim)
    w_o = rnd(n_heads * head_dim, d_model)
    dy = (torch.randn(seq_len, d_model, generator=gen, device=DEVICE) * 0.02).to(torch.bfloat16)

    r_fp32 = run_naive_autograd(
        x, w_q, w_k, w_v, w_o, dy,
        n_heads=n_heads, n_kv=n_kv, head_dim=head_dim,
        internal_dtype=torch.float32,
    )
    r_bf16 = run_naive_autograd(
        x, w_q, w_k, w_v, w_o, dy,
        n_heads=n_heads, n_kv=n_kv, head_dim=head_dim,
        internal_dtype=torch.bfloat16,
    )
    r_orig = orig_attention_no_rope(
        x, w_q, w_k, w_v, w_o, n_heads, n_kv, head_dim, seq_len, dy,
    )

    print(f"\n  NO-ROPE attention: seq_len={seq_len}, n_heads={n_heads}, n_kv={n_kv}, head_dim={head_dim}")
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
