"""Routed-MLP-scope MoE bench: flextrain vs scattermoe (and optionally
sonic-moe when its CUDA backend is available).

Comparison surface — same on all impls:
    inputs:  x (T, d), expert_p (T, K), expert_idxs (T, K),
             w_up (E, d, 2F), w_down (E, F, d)
    forward: scatter -> per-expert (up + SwiGLU + down) -> weighted gather
             (no router GEMM, no shared experts, no residual add)
    backward: dy (T, d) -> dx (T, d), dw_up, dw_down

Shapes target Qwen3.5-35B-A3B's MoE block: d=2048, F=512, E=256, K=8.
Chunk sizes: 16K, 32K, 64K tokens.

Run:
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \\
    PYTHONPATH=. python external_moe_impl/bench/bench_moe.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import torch

# ---------------------------------------------------------------------------
# flextrain wrapper — matches scattermoe.GLUMLP.forward scope.
# ---------------------------------------------------------------------------

_FLEXTRAIN_ROOT = "/home/shein/Documents/flextrain"
sys.path.insert(0, _FLEXTRAIN_ROOT)

from flextrain.ops.full_moe import (
    flextrain_moe_scatter,
    flextrain_moe_gather,
    flextrain_moe_scatter_routing_weights,
    flextrain_moe_sort,
    flextrain_copy_expert_counts,
    swiglu_expert_loop_fwd,
    swiglu_expert_loop_bwd,
)


@dataclass
class FlextrainBufs:
    """Pre-allocated buffers for flextrain wrapper. Sized for one chunk."""
    T: int
    d: int
    F: int
    E: int
    K: int
    dtype: torch.dtype
    device: torch.device

    def __post_init__(self):
        TK = self.T * self.K
        # Sort/scatter book-keeping
        self.index_mapping = torch.empty(self.T, self.K, dtype=torch.int32, device=self.device)
        self.expert_counts_gpu = torch.empty(self.E, dtype=torch.int32, device=self.device)
        self.expert_counts_cpu = torch.empty(self.E, dtype=torch.int32).pin_memory()
        # Forward-side
        self.scattered_x = torch.empty(TK, self.d, dtype=self.dtype, device=self.device)
        self.scattered_router_weights = torch.empty(TK, 1, dtype=self.dtype, device=self.device)
        self.x_preact = torch.empty(TK, 2 * self.F, dtype=self.dtype, device=self.device)
        # Per-stream act scratch (sized to TK as a safe upper bound;
        # max_T_e <= TK)
        self.x_act_even = torch.empty(TK, self.F, dtype=self.dtype, device=self.device)
        self.x_act_odd = self.x_act_even  # single-stream variant
        # MLP output (pre-gather)
        self.scattered_y = torch.empty(TK, self.d, dtype=self.dtype, device=self.device)


class FlextrainGLUMoE(torch.autograd.Function):
    """Same scope as scattermoe.GLUMLP.forward. Backward writes into
    pre-allocated grad accumulators on `w_up` / `w_down` via PyTorch's
    `.grad` attribute.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,            # (T, d)  -- input activations, requires grad
        expert_p: torch.Tensor,     # (T, K)  -- router probs (already top-k softmax'd)
        expert_idxs: torch.Tensor,  # (T, K)  -- chosen expert ids
        w_up: torch.Tensor,         # (E, d, 2F)  -- requires grad
        w_down: torch.Tensor,       # (E, F, d)   -- requires grad
        bufs: FlextrainBufs,
        primary_stream: torch.cuda.Stream,
    ) -> torch.Tensor:               # (T, d)
        # Sort + scatter.
        flextrain_moe_sort(
            expert_idxs,
            num_experts=bufs.E,
            indices=bufs.index_mapping,
            expert_counts_gpu=bufs.expert_counts_gpu,
        )
        flextrain_moe_scatter(x, bufs.index_mapping, out=bufs.scattered_x)
        flextrain_moe_scatter_routing_weights(
            expert_p, bufs.index_mapping, out=bufs.scattered_router_weights,
        )
        flextrain_copy_expert_counts(bufs.expert_counts_gpu, bufs.expert_counts_cpu)
        torch.cuda.current_stream().synchronize()

        # Per-expert SwiGLU MLP.
        swiglu_expert_loop_fwd(
            scattered_x=bufs.scattered_x,
            x_preact_buf=bufs.x_preact,
            w_up=w_up,
            w_down=w_down,
            expert_counts_cpu=bufs.expert_counts_cpu,
            primary_stream=primary_stream,
            secondary_stream=None,
            x_act_even=bufs.x_act_even,
            x_act_odd=bufs.x_act_odd,
        )
        # The expert loop overwrites scattered_x with the post-down output.
        # Save the inputs for bwd by stashing a copy if x is needed there
        # (we'll re-scatter x in bwd to avoid the copy).

        # Gather (weighted) — mirrors scattermoe's `gates @ output_expanded`.
        out = torch.empty_like(x)
        flextrain_moe_gather(
            bufs.scattered_x, bufs.index_mapping,
            weights=expert_p,
            residual=None,
            out=out,
        )

        ctx.bufs = bufs
        ctx.save_for_backward(x, expert_p, expert_idxs, w_up, w_down)
        ctx.primary_stream = primary_stream
        return out

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, expert_p, expert_idxs, w_up, w_down = ctx.saved_tensors
        bufs: FlextrainBufs = ctx.bufs
        primary_stream = ctx.primary_stream
        TK = bufs.T * bufs.K

        # Re-scatter the saved fwd input (overwriting scattered_x which
        # currently holds the post-down output — fine, we don't need that
        # anymore).
        scattered_x_recompute = bufs.scattered_x  # reuse buffer
        flextrain_moe_scatter(x, bufs.index_mapping, out=scattered_x_recompute)

        # Scatter dy by sorted expert order.
        scattered_upstream = torch.empty(TK, bufs.d, dtype=dy.dtype, device=dy.device)
        flextrain_moe_scatter(dy, bufs.index_mapping, out=scattered_upstream)

        srw = bufs.scattered_router_weights
        dprobs = torch.zeros_like(srw)

        # Allocate or reuse grad accumulators on w_up / w_down.
        # PyTorch creates .grad on first .backward; here we're inside
        # a custom Function so we manually return per-input grads. The
        # expert loop accumulates into a `grads` dict, but we'd rather
        # the autograd machinery wire `.grad` itself. So allocate
        # per-call grad tensors here.
        g_up = torch.zeros_like(w_up)
        g_down = torch.zeros_like(w_down)

        swiglu_expert_loop_bwd(
            scattered_upstream=scattered_upstream,
            scattered_x=scattered_x_recompute,
            x_preact_buf=bufs.x_preact,
            srw=srw,
            dprobs=dprobs,
            w_up=w_up,
            w_down=w_down,
            grads={"g_up": g_up, "g_down": g_down},
            expert_counts_cpu=bufs.expert_counts_cpu,
            expert_dim=bufs.F,
            primary_stream=primary_stream,
            secondary_stream=None,
        )

        # scattered_upstream now holds per-slot dx; gather back to per-token.
        dx = torch.empty_like(dy)
        flextrain_moe_gather(scattered_upstream, bufs.index_mapping, out=dx)

        # We're not differentiating through expert_p / expert_idxs in this
        # bench (matches scattermoe scope, which routes them as inputs).
        return dx, None, None, g_up, g_down, None, None


def flextrain_glu_moe(x, expert_p, expert_idxs, w_up, w_down, bufs, primary_stream):
    return FlextrainGLUMoE.apply(x, expert_p, expert_idxs, w_up, w_down, bufs, primary_stream)


# ---------------------------------------------------------------------------
# Bench harness.
# ---------------------------------------------------------------------------


def _make_routing(T: int, E: int, K: int, *, device, dtype):
    """Generate plausible router outputs: top-K from softmax of random logits."""
    logits = torch.randn(T, E, device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    expert_p, expert_idxs = probs.topk(K, dim=-1)
    expert_p = (expert_p / expert_p.sum(dim=-1, keepdim=True)).to(dtype)
    expert_idxs = expert_idxs.to(torch.int32)
    return expert_p, expert_idxs


def _time_calls(fn: Callable, *, warmup: int, iters: int) -> float:
    """Time `iters` calls of `fn`. Sync once at the end. Returns median ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    runs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        runs.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(runs)


def bench_flextrain(T: int, d: int, F: int, E: int, K: int, *, dtype, device,
                    iters: int, warmup: int):
    bufs = FlextrainBufs(T=T, d=d, F=F, E=E, K=K, dtype=dtype, device=device)
    primary_stream = torch.cuda.current_stream()

    # Detached tensors for fwd-only timing; fresh leaves with grad for bwd timing.
    x_fwd_only = torch.randn(T, d, device=device, dtype=dtype)
    w_up_fwd_only = torch.randn(E, d, 2 * F, device=device, dtype=dtype) / (d ** 0.5)
    w_down_fwd_only = torch.randn(E, F, d, device=device, dtype=dtype) / (F ** 0.5)
    expert_p, expert_idxs = _make_routing(T, E, K, device=device, dtype=dtype)
    dy = torch.randn(T, d, device=device, dtype=dtype)

    # Forward-only (no autograd graph at all)
    def fwd():
        with torch.no_grad():
            FlextrainGLUMoE.apply(x_fwd_only, expert_p, expert_idxs,
                                   w_up_fwd_only, w_down_fwd_only,
                                   bufs, primary_stream)
    t_fwd = _time_calls(fwd, warmup=warmup, iters=iters)

    # Separate leaf tensors with requires_grad for bwd timing.
    x = torch.randn(T, d, device=device, dtype=dtype, requires_grad=True)
    w_up = torch.randn(E, d, 2 * F, device=device, dtype=dtype) / (d ** 0.5)
    w_up.requires_grad_(True)
    w_down = torch.randn(E, F, d, device=device, dtype=dtype) / (F ** 0.5)
    w_down.requires_grad_(True)

    # Forward + backward (matches scattermoe's autograd path)
    def fwd_bwd():
        x.grad = None; w_up.grad = None; w_down.grad = None
        y = flextrain_glu_moe(x, expert_p, expert_idxs, w_up, w_down,
                              bufs, primary_stream)
        y.backward(dy)
    t_total = _time_calls(fwd_bwd, warmup=warmup, iters=iters)
    t_bwd = max(0.0, t_total - t_fwd)

    return {"fwd_ms": t_fwd, "bwd_ms": t_bwd, "total_ms": t_total}


def bench_scattermoe(T: int, d: int, F: int, E: int, K: int, *, dtype, device,
                     iters: int, warmup: int):
    try:
        from scattermoe.mlp import GLUMLP
    except ImportError:
        return None

    mlp = GLUMLP(input_size=d, hidden_size=F, num_experts=E, top_k=K, bias=False).to(device).to(dtype)
    x = torch.randn(T, d, device=device, dtype=dtype, requires_grad=True)
    expert_p, expert_idxs = _make_routing(T, E, K, device=device, dtype=dtype)
    dy = torch.randn(T, d, device=device, dtype=dtype)

    def fwd():
        with torch.no_grad():
            mlp(x.detach(), expert_p, expert_idxs)
    t_fwd = _time_calls(fwd, warmup=warmup, iters=iters)

    def fwd_bwd():
        x.grad = None
        for p in mlp.parameters(): p.grad = None
        y = mlp(x, expert_p, expert_idxs)
        y.backward(dy, retain_graph=False)
    t_total = _time_calls(fwd_bwd, warmup=warmup, iters=iters)
    t_bwd = max(0.0, t_total - t_fwd)

    return {"fwd_ms": t_fwd, "bwd_ms": t_bwd, "total_ms": t_total}


def bench_sonicmoe(T: int, d: int, F: int, E: int, K: int, *, dtype, device,
                   iters: int, warmup: int):
    """sonic-moe with the *scattermoe* backend. Same scope — but sonic
    embeds the router GEMM, so we're benching slightly more work. Not
    a true apples-to-apples; reported separately."""
    try:
        from sonicmoe.moe import MoE
        from sonicmoe.enums import ActivationType, KernelBackendMoE
    except ImportError:
        return None
    try:
        moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=d,
                  intermediate_size=F, activation_function=ActivationType.SWIGLU,
                  add_bias=False, std=0.02).to(device).to(dtype)
        # Try a single call to confirm backend works on this hardware.
        x_test = torch.randn(64, d, device=device, dtype=dtype)
        moe(x_test, kernel_backend_moe=KernelBackendMoE.scattermoe)
    except Exception as e:
        return {"error": f"sonicmoe unavailable: {type(e).__name__}: {str(e)[:120]}"}

    x = torch.randn(T, d, device=device, dtype=dtype, requires_grad=True)
    dy = torch.randn(T, d, device=device, dtype=dtype)

    def fwd():
        with torch.no_grad():
            moe(x.detach(), kernel_backend_moe=KernelBackendMoE.scattermoe)
    t_fwd = _time_calls(fwd, warmup=warmup, iters=iters)

    def fwd_bwd():
        x.grad = None
        for p in moe.parameters(): p.grad = None
        y, _ = moe(x, kernel_backend_moe=KernelBackendMoE.scattermoe)
        y.backward(dy)
    t_total = _time_calls(fwd_bwd, warmup=warmup, iters=iters)
    t_bwd = max(0.0, t_total - t_fwd)

    return {"fwd_ms": t_fwd, "bwd_ms": t_bwd, "total_ms": t_total}


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=int, nargs="+", default=[16384, 32768, 65536],
                   help="Chunk sizes (token counts) to benchmark.")
    p.add_argument("--d", type=int, default=2048)
    p.add_argument("--f", type=int, default=512)
    p.add_argument("--e", type=int, default=256)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--include-sonic", action="store_true",
                   help="Also try sonic-moe (likely only works on H100+).")
    args = p.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    print(f"Config: d={args.d} F={args.f} E={args.e} K={args.k} dtype=bf16")
    print(f"Iters: {args.iters} (warmup {args.warmup})")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    header = f"{'T':>8s}  {'impl':>14s}  {'fwd (ms)':>10s}  {'bwd (ms)':>10s}  {'total (ms)':>12s}"
    print(header)
    print("-" * len(header))

    for T in args.chunks:
        runners = [
            ("flextrain", bench_flextrain),
            ("scattermoe", bench_scattermoe),
        ]
        if args.include_sonic:
            runners.append(("sonicmoe", bench_sonicmoe))

        for name, fn in runners:
            r = fn(T, args.d, args.f, args.e, args.k,
                   dtype=dtype, device=device,
                   iters=args.iters, warmup=args.warmup)
            if r is None:
                print(f"{T:>8d}  {name:>14s}  {'(skipped — package not installed)':>40s}")
            elif "error" in r:
                print(f"{T:>8d}  {name:>14s}  {r['error']:>40s}")
            else:
                print(f"{T:>8d}  {name:>14s}  "
                      f"{r['fwd_ms']:10.2f}  {r['bwd_ms']:10.2f}  {r['total_ms']:12.2f}")
        print()


if __name__ == "__main__":
    main()
