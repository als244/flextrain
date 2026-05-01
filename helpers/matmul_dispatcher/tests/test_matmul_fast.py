"""Tests for the FASTCALL fast-path entry into the cuBLASLt dispatcher.

Validates math parity with the existing ``CublasLtDispatcher.matmul``
method (and with PyTorch eager) plus reports per-call CPU latency for
the realistic shape mix used by MoE-LoRA bwd.

Run: ``pytest helpers/matmul_dispatcher/tests/test_matmul_fast.py -v``
or as a script for the microbench output.
"""
from __future__ import annotations

import statistics
import time

import pytest
import torch

from matmul_dispatcher import (
    CublasLtDispatcher,
    matmul_fast,
    tensor_layout,
    dtype_enum,
)


def _check_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if matmul_fast is None:
        pytest.skip("matmul_fast extension module not built")


@pytest.fixture(scope="module")
def disp():
    _check_cuda()
    return CublasLtDispatcher(round_multiple=32)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,N,K,dtype",
    [
        (200, 1024, 2048, torch.bfloat16),
        (500, 2048, 512, torch.bfloat16),
        (37, 16, 4096, torch.bfloat16),  # odd shape stress
        (16, 1024, 16, torch.bfloat16),
    ],
)
def test_matmul_fast_matches_eager(disp, M, N, K, dtype):
    """``matmul_fast`` (no C/D, alpha=1, beta=0) writes A @ B into D."""
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    D = torch.empty(M, N, device="cuda", dtype=dtype)
    expected = A @ B

    a_ptr, lda, ta = tensor_layout(A)
    b_ptr, ldb, tb = tensor_layout(B)
    d_ptr, ldd, _ = tensor_layout(D)
    dt = dtype_enum(dtype)
    sp = torch.cuda.current_stream().cuda_stream

    matmul_fast(
        disp._ctx_int, sp,
        M, N, K,
        a_ptr, lda, ta,
        b_ptr, ldb, tb,
        0, N,                # no C
        d_ptr, ldd,
        disp.ws_ptr, disp.workspace_size,
        dt, dt, dt, dt, 0,
        1.0, 0.0,
    )
    torch.cuda.synchronize()

    # bf16 cuBLAS GEMM accumulates in fp32; compare in fp32 with tolerance.
    rel = (D.float() - expected.float()).abs() / expected.float().abs().clamp(min=1e-3)
    assert rel.max().item() < 5e-2, (
        f"matmul_fast diverged: max rel err {rel.max().item():.3e}"
    )


def test_matmul_fast_with_alpha_beta_C(disp):
    """Verify ``D = alpha * A @ B + beta * C`` with C=D (in-place accum)."""
    M, K, N = 200, 2048, 1024
    dtype = torch.bfloat16
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    D0 = torch.randn(M, N, device="cuda", dtype=dtype)

    alpha, beta = 0.5, 1.0
    expected = (alpha * (A.float() @ B.float()) + beta * D0.float()).to(dtype)

    D = D0.clone()
    a_ptr, lda, ta = tensor_layout(A)
    b_ptr, ldb, tb = tensor_layout(B)
    d_ptr, ldd, _ = tensor_layout(D)
    dt = dtype_enum(dtype)
    sp = torch.cuda.current_stream().cuda_stream

    matmul_fast(
        disp._ctx_int, sp,
        M, N, K,
        a_ptr, lda, ta,
        b_ptr, ldb, tb,
        d_ptr, ldd,          # C = D in-place
        d_ptr, ldd,
        disp.ws_ptr, disp.workspace_size,
        dt, dt, dt, dt, 0,
        alpha, beta,
    )
    torch.cuda.synchronize()
    rel = (D.float() - expected.float()).abs() / expected.float().abs().clamp(min=1e-3)
    assert rel.max().item() < 5e-2


def test_matmul_fast_matches_dispatcher_matmul(disp):
    """Bit-for-bit match between matmul_fast and the existing matmul()."""
    M, K, N = 500, 2048, 1024
    dtype = torch.bfloat16
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

    D_slow = torch.empty(M, N, device="cuda", dtype=dtype)
    sp = torch.cuda.current_stream().cuda_stream
    disp.matmul(sp, A=A, B=B, D=D_slow)

    D_fast = torch.empty(M, N, device="cuda", dtype=dtype)
    a_ptr, lda, ta = tensor_layout(A)
    b_ptr, ldb, tb = tensor_layout(B)
    d_ptr, ldd, _ = tensor_layout(D_fast)
    dt = dtype_enum(dtype)
    matmul_fast(
        disp._ctx_int, sp,
        M, N, K,
        a_ptr, lda, ta,
        b_ptr, ldb, tb,
        0, N,
        d_ptr, ldd,
        disp.ws_ptr, disp.workspace_size,
        dt, dt, dt, dt, 0,
        1.0, 0.0,
    )
    torch.cuda.synchronize()
    # Same algo path -> should be exactly equal.
    assert torch.equal(D_slow, D_fast)


# ---------------------------------------------------------------------------
# Microbench (also runnable as a script)
# ---------------------------------------------------------------------------


def _bench(disp, M, K, N, dtype, *, warmup=5000, inner=5000, outer=5):
    """Returns (slow_us_med, fast_us_med, fast_full_us_med)."""
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    D = torch.empty(M, N, device="cuda", dtype=dtype)
    sp = torch.cuda.current_stream().cuda_stream

    # Slow path = the existing CublasLtDispatcher.matmul.
    def slow():
        disp.matmul(sp, A=A, B=B, D=D)

    # Fast path with everything pre-extracted (best case).
    a_ptr, lda, ta = tensor_layout(A)
    b_ptr, ldb, tb = tensor_layout(B)
    d_ptr, ldd, _ = tensor_layout(D)
    dt = dtype_enum(dtype)
    ctx = disp._ctx_int
    ws_ptr = disp.ws_ptr
    ws_bytes = disp.workspace_size

    def fast_preextracted():
        matmul_fast(
            ctx, sp,
            M, N, K,
            a_ptr, lda, ta,
            b_ptr, ldb, tb,
            0, N,
            d_ptr, ldd,
            ws_ptr, ws_bytes,
            dt, dt, dt, dt, 0,
            1.0, 0.0,
        )

    # Fast path that re-extracts on every call (worst case for fast).
    def fast_with_reextract():
        a_ptr2, lda2, ta2 = tensor_layout(A)
        b_ptr2, ldb2, tb2 = tensor_layout(B)
        d_ptr2, ldd2, _ = tensor_layout(D)
        matmul_fast(
            ctx, sp,
            M, N, K,
            a_ptr2, lda2, ta2,
            b_ptr2, ldb2, tb2,
            0, N,
            d_ptr2, ldd2,
            ws_ptr, ws_bytes,
            dt, dt, dt, dt, 0,
            1.0, 0.0,
        )

    for _ in range(warmup):
        slow(); fast_preextracted(); fast_with_reextract()
    torch.cuda.synchronize()

    def time_fn(fn):
        runs = []
        for _ in range(outer):
            t0 = time.perf_counter()
            for _ in range(inner): fn()
            runs.append((time.perf_counter() - t0) / inner * 1e6)
        return statistics.median(runs)

    return time_fn(slow), time_fn(fast_preextracted), time_fn(fast_with_reextract)


def test_microbench_report():
    """Print a per-shape latency table. Always passes."""
    _check_cuda()
    disp = CublasLtDispatcher(round_multiple=32)
    shapes = [
        ("g_up   T_e=200",  200, 2048, 1024),
        ("g_up   T_e=500",  500, 2048, 1024),
        ("g_down T_e=200",  200,  512, 2048),
        ("g_down T_e=500",  500,  512, 2048),
    ]
    print(
        f"\n{'shape':18s}  {'slow (us)':>10s}  "
        f"{'fast-pre (us)':>14s}  {'fast-re (us)':>13s}  {'speedup':>8s}"
    )
    print("-" * 75)
    for name, M, K, N in shapes:
        s, fp, fr = _bench(disp, M, K, N, torch.bfloat16)
        print(
            f"{name:18s}  {s:8.2f}    {fp:12.2f}    {fr:11.2f}    "
            f"{s/fp:7.2f}x"
        )


if __name__ == "__main__":
    _check_cuda()
    test_microbench_report()
