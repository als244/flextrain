"""Microbench: shared-expert fwd+bwd in MoESwiGLUSharedExpertFFN.

Compares the new bf16-fused path (in production) to a pure-pytorch
fp32 reference at Qwen3.5-MoE-35B-A3B shapes (S=1, d=2048, F=512,
T={4096, 16384, 32768}). Reports:
1. Max abs / max rel diff between new and reference outputs/grads.
2. Wall time + bytes-per-second for fwd and bwd.

Usage:
    python tests/test_shared_moe_microbench.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn.functional as F

from flextrain.nn.blocks.ffn_moe_shared import (
    MoESwiGLUSharedExpertFFN, MoESwiGLUSharedExpertConfig,
)


def make_inputs(T: int, d: int, F_s: int, S: int = 1, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x_2d = torch.randn(T, d, dtype=dtype, device="cuda", generator=g)
    w_up = torch.randn(S, d, 2 * F_s, dtype=dtype, device="cuda", generator=g) * 0.02
    w_down = torch.randn(S, F_s, d, dtype=dtype, device="cuda", generator=g) * 0.02
    w_gate = torch.randn(d, S, dtype=dtype, device="cuda", generator=g) * 0.02
    return x_2d, w_up, w_down, w_gate


def reference_fwd(x_2d, w_up, w_down, w_gate):
    """Pure-pytorch reference; explicit fp32 promotions, matches the
    pre-cleanup behavior. Returns (sh_pre, sh_each, sh_gate_pre, shared_out).
    """
    S = w_up.shape[0]
    Fs = w_up.shape[2] // 2
    bf = x_2d.dtype
    sh_pre = torch.einsum("td,sdf->tsf", x_2d.float(), w_up.float()).to(bf)
    up_h = sh_pre[..., :Fs]
    gate_h = sh_pre[..., Fs:]
    sh_act = up_h * F.silu(gate_h.float()).to(bf)
    sh_each = torch.einsum("tsf,sfd->tsd", sh_act.float(), w_down.float()).to(bf)
    sh_gate_pre = x_2d @ w_gate
    sig_gate = torch.sigmoid(sh_gate_pre.float()).to(bf)
    shared_out = (sig_gate.unsqueeze(-1) * sh_each).sum(dim=1)
    return sh_pre, sh_each, sh_gate_pre, shared_out


def reference_bwd(x_2d, w_up, w_down, w_gate, sh_pre, dy):
    """Pure-pytorch reference bwd, mirror of the original code.
    Returns (g_w_up, g_w_down, g_w_gate, dx)."""
    S = w_up.shape[0]
    Fs = w_up.shape[2] // 2
    bf = x_2d.dtype
    up_h = sh_pre[..., :Fs]
    gate_h = sh_pre[..., Fs:]
    sh_act = up_h * F.silu(gate_h.float()).to(bf)
    sh_each = torch.einsum("tsf,sfd->tsd", sh_act.float(), w_down.float()).to(bf)
    sh_gate_pre = x_2d @ w_gate
    sig_gate = torch.sigmoid(sh_gate_pre.float()).to(bf)

    d_sh_each = dy.unsqueeze(1) * sig_gate.unsqueeze(-1)
    d_sh_gate = (dy.unsqueeze(1) * sh_each).sum(dim=-1)
    d_sh_gate_pre = (
        d_sh_gate.float() * sig_gate.float() * (1.0 - sig_gate.float())
    ).to(bf)

    g_w_gate = (x_2d.float().T @ d_sh_gate_pre.float()).to(bf)
    dx_via_gate = (d_sh_gate_pre.float() @ w_gate.float().T).to(bf)

    g_w_down = torch.einsum(
        "tsf,tsd->sfd", sh_act.float(), d_sh_each.float()
    ).to(bf)
    d_sh_act = torch.einsum(
        "tsd,sfd->tsf", d_sh_each.float(), w_down.float()
    ).to(bf)

    gate_f = gate_h.float()
    sig_g = gate_f.sigmoid()
    silu_gate = (gate_f * sig_g).to(bf)
    dsilu = (sig_g * (1.0 + gate_f * (1.0 - sig_g))).to(bf)
    d_up = d_sh_act * silu_gate
    d_gate = d_sh_act * up_h * dsilu
    d_x_shared_pre = torch.cat([d_up, d_gate], dim=-1)

    g_w_up = torch.einsum("td,tsf->sdf", x_2d.float(), d_x_shared_pre.float()).to(bf)
    dx_via_shared = torch.einsum(
        "tsf,sdf->td", d_x_shared_pre.float(), w_up.float(),
    ).to(bf)

    dx = dx_via_gate + dx_via_shared
    return g_w_up, g_w_down, g_w_gate, dx


def block_fwd(block, x_2d, weights):
    sh_pre, sh_each = block._shared_swiglu_fwd(x_2d, weights)
    sh_gate_pre = block._shared_gate_fwd(x_2d, weights)
    sig = torch.sigmoid(sh_gate_pre.float()).to(x_2d.dtype)
    shared_out = (sig.unsqueeze(-1) * sh_each).sum(dim=1)
    return sh_pre, sh_each, sh_gate_pre, shared_out


def block_bwd(block, x_2d, sh_pre, sh_gate_pre, dy, weights, grads):
    """Run only the shared-expert portion of bwd by simulating the slot
    interface. Avoids running the routed half (which isn't part of this
    benchmark)."""

    class _MockSlot:
        def __init__(self, sh_pre, sh_gate_pre):
            self.x_shared_pre = sh_pre
            self.x_shared_gate = sh_gate_pre
            self.aux = {"recompute_ffn_norm_output": x_2d}

    slot = _MockSlot(sh_pre, sh_gate_pre)

    # We need to run only the shared half (not routed). Easiest: directly
    # call the shared-half code by inlining the relevant block of bwd.
    # Instead, we patch out the routed call by replacing
    # block._routed_ffn.bwd with a no-op returning zeros.
    class _NoopRouted:
        def bwd(self, dy_resid, *a, **kw):
            return torch.zeros_like(dy_resid)
        def fwd(self, *a, **kw): pass
        def fwd_recompute_x_up(self, *a, **kw): pass
        def fields(self): return ()
        def param_spec(self): return type(block._routed_ffn).param_spec(block._routed_ffn)

    saved = block._routed_ffn
    block._routed_ffn = _NoopRouted()
    try:
        dx = block.bwd(
            dy, weights, grads, slot, ctx=None, chunk=None, layer_id=0,
        )
    finally:
        block._routed_ffn = saved
    return dx


def time_fn(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms per iter


def report_diff(name, a, b, rel_floor=1e-3):
    abs_diff = (a.float() - b.float()).abs().max().item()
    abs_max = b.float().abs().max().item()
    rel_diff = abs_diff / max(abs_max, rel_floor)
    print(f"  {name:<20} max|Δ|={abs_diff:.4e}   |a|max={abs_max:.4e}   rel={rel_diff:.4e}")


def bench_one(T: int, d: int, F_s: int, S: int = 1):
    print(f"\n=== T={T} d={d} F={F_s} S={S} ===")
    x_2d, w_up, w_down, w_gate = make_inputs(T, d, F_s, S)
    weights = {
        "w_shared_up": w_up,
        "w_shared_down": w_down,
        "w_shared_expert_gate": w_gate,
    }

    cfg = MoESwiGLUSharedExpertConfig(
        d_model=d, expert_dim=F_s * 4, num_experts=4, top_k=2,
        compute_dtype=torch.bfloat16, master_dtype=torch.float32,
        grad_dtype=torch.bfloat16,
        num_shared_experts=S, shared_expert_dim=F_s,
    )
    block = MoESwiGLUSharedExpertFFN(cfg)

    # Correctness check.
    sh_pre_n, sh_each_n, sh_gate_pre_n, shared_out_n = block_fwd(block, x_2d, weights)
    sh_pre_r, sh_each_r, sh_gate_pre_r, shared_out_r = reference_fwd(x_2d, w_up, w_down, w_gate)

    print("FWD correctness (new vs fp32-reference):")
    report_diff("sh_pre",      sh_pre_n,      sh_pre_r)
    report_diff("sh_each",     sh_each_n,     sh_each_r)
    report_diff("sh_gate_pre", sh_gate_pre_n, sh_gate_pre_r)
    report_diff("shared_out",  shared_out_n,  shared_out_r)

    # Bwd correctness via the public bwd API.
    dy = torch.randn_like(shared_out_r)
    grads_new = {
        "g_shared_up":   torch.zeros_like(w_up),
        "g_shared_down": torch.zeros_like(w_down),
        "g_shared_expert_gate": torch.zeros_like(w_gate),
    }
    dx_new = block_bwd(block, x_2d, sh_pre_n, sh_gate_pre_n, dy, weights, grads_new)
    g_w_up_r, g_w_down_r, g_w_gate_r, dx_r = reference_bwd(
        x_2d, w_up, w_down, w_gate, sh_pre_r, dy,
    )
    print("BWD correctness:")
    report_diff("g_w_up",   grads_new["g_shared_up"],   g_w_up_r)
    report_diff("g_w_down", grads_new["g_shared_down"], g_w_down_r)
    report_diff("g_w_gate", grads_new["g_shared_expert_gate"], g_w_gate_r)
    report_diff("dx",       dx_new, dx_r)

    # Timing.
    print("Timing (fwd):")
    new_ms = time_fn(lambda: block_fwd(block, x_2d, weights))
    ref_ms = time_fn(lambda: reference_fwd(x_2d, w_up, w_down, w_gate))
    print(f"  new={new_ms:.3f}ms   reference={ref_ms:.3f}ms   speedup={ref_ms/new_ms:.2f}x")

    print("Timing (bwd):")
    def new_bwd_iter():
        # Rebuild grads each iter to keep the measurement isolated to a single bwd pass.
        for g in grads_new.values():
            g.zero_()
        block_bwd(block, x_2d, sh_pre_n, sh_gate_pre_n, dy, weights, grads_new)
    def ref_bwd_iter():
        reference_bwd(x_2d, w_up, w_down, w_gate, sh_pre_r, dy)
    new_ms = time_fn(new_bwd_iter)
    ref_ms = time_fn(ref_bwd_iter)
    print(f"  new={new_ms:.3f}ms   reference={ref_ms:.3f}ms   speedup={ref_ms/new_ms:.2f}x")


def main():
    if not torch.cuda.is_available():
        print("CUDA required.")
        sys.exit(1)
    print(f"Device: {torch.cuda.get_device_name()}")
    # Qwen3.5-MoE-35B-A3B: d=2048, shared_F=512, S=1.
    for T in (4096, 16384, 32768):
        bench_one(T, d=2048, F_s=512, S=1)


if __name__ == "__main__":
    main()
