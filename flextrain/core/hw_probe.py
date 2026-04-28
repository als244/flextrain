"""Light-weight hardware probe for DP-solver inputs.

The DP solver in :mod:`flextrain.core.save_level` needs two scalars:

* effective device throughput in TFLOPS (used by ``HardwareCost.flops_to_ms``);
* unidirectional concurrent host<->device PCIe bandwidth in GB/s
  (used by ``HardwareCost.bytes_to_ms``).

Orig's ``hardware_env.get_hardware_env`` (orig/awsm_transformer/hardware_env.py)
runs a per-component matmul sweep + multi-direction transfer benches at
~100 reps each, plus two extra reference benches. That is fine when working
set sizing actually consumes the per-component breakdown, but for the DP
solver's use case (two scalars, ~10% accuracy is plenty -- the binary
"does level 3 fit?" decision is robust to that) it is overkill and burns
seconds of GPU time at startup.

This module provides :func:`probe_hardware` -- one bf16 square matmul
plus one concurrent host<->device transfer, ~10 reps each, that returns a
:class:`flextrain.core.save_level.HardwareCost`. Total wall time on a
modern datacenter GPU is well under a second.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .save_level import HardwareCost


@dataclass(frozen=True)
class HardwareProbeResult:
    """Outputs of :func:`probe_hardware`. Wraps :class:`HardwareCost`
    plus a few extra fields useful for logging."""

    hw_cost: HardwareCost
    matmul_n: int
    matmul_per_call_ms: float
    transfer_bytes: int
    transfer_per_call_ms: float
    # GPU memory bandwidth in GB/s, measured via a memory-bound
    # ``(1 x n) @ (n x n)`` matmul (orig's basic_peak_mem_bandwidth probe).
    # Used by the working-set solver's arithmetic-intensity bound for
    # picking minimum chunk size.
    mem_bw_gbps: float


def _bench_matmul(
    n: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    n_warmup: int,
    n_reps: int,
) -> tuple[float, float]:
    """Run ``A @ B`` (n x n square) ``n_reps`` times and return
    ``(per_call_seconds, achieved_tflops)``.

    Uses ``torch.matmul`` directly so we don't depend on the custom CUTLASS
    dispatcher in ``orig.awsm_transformer.matmul_dispatchers`` -- this keeps
    the probe module self-contained inside ``flextrain``.
    """
    A = torch.randn(n, n, dtype=dtype, device=device)
    B = torch.randn(n, n, dtype=dtype, device=device)

    for _ in range(n_warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize(device)

    start = time.perf_counter_ns()
    for _ in range(n_reps):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize(device)
    total = (time.perf_counter_ns() - start) / 1e9

    per_call = total / n_reps
    flops = 2.0 * n * n * n
    tflops = (flops / per_call) / 1e12
    del A, B
    return per_call, tflops


def _bench_mem_bandwidth(
    n: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    n_warmup: int,
    n_reps: int,
) -> float:
    """Estimate device memory bandwidth via a memory-bound matmul shape:
    ``(1, n) @ (n, n)`` reads ~``n*n*itemsize`` weight bytes per call but
    only does ``2 * n * n`` flops, so it's bandwidth-limited rather than
    compute-limited (orig's ``get_basic_peak_mem_bandwidth_gb_per_sec``).

    Returns achieved GB/s.
    """
    A = torch.randn(1, n, dtype=dtype, device=device)
    B = torch.randn(n, n, dtype=dtype, device=device)
    for _ in range(n_warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize(device)

    start = time.perf_counter_ns()
    for _ in range(n_reps):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize(device)
    total = (time.perf_counter_ns() - start) / 1e9
    per_call = total / n_reps

    # Bytes touched per call: A (n) + B (n*n) + output (n).
    bytes_touched = dtype.itemsize * (n * n + 2 * n)
    gbps = (bytes_touched / per_call) / 1e9
    del A, B
    return gbps


def _bench_pcie_concurrent(
    nbytes: int,
    *,
    device: torch.device,
    n_warmup: int,
    n_reps: int,
) -> tuple[float, float]:
    """Time concurrent host->device + device->host transfers of ``nbytes``
    on two streams, returning ``(per_iter_seconds, gbps)``.

    Why concurrent: the engine overlaps inbound (params/acts fetch) and
    outbound (gradient/act offload) traffic, so the relevant bandwidth for
    DP scheduling is what each side gets while the other is also active.
    Matches what orig measures as
    ``overall_unidirectional_concurrent_bandwidth_gb_per_sec``.
    """
    h_in = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    d_in = torch.empty(nbytes, dtype=torch.uint8, device=device)
    d_out = torch.empty(nbytes, dtype=torch.uint8, device=device)
    h_out = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)

    s_in = torch.cuda.Stream(device=device)
    s_out = torch.cuda.Stream(device=device)

    for _ in range(n_warmup):
        with torch.cuda.stream(s_in):
            d_in.copy_(h_in, non_blocking=True)
        with torch.cuda.stream(s_out):
            h_out.copy_(d_out, non_blocking=True)
        torch.cuda.synchronize(device)

    start = time.perf_counter_ns()
    for _ in range(n_reps):
        with torch.cuda.stream(s_in):
            d_in.copy_(h_in, non_blocking=True)
        with torch.cuda.stream(s_out):
            h_out.copy_(d_out, non_blocking=True)
        torch.cuda.synchronize(device)
    total = (time.perf_counter_ns() - start) / 1e9

    per_iter = total / n_reps
    # Each iteration moves ``nbytes`` in each direction. We report the
    # unidirectional bandwidth -- ``nbytes / per_iter`` -- to mirror orig's
    # convention (orig:42, ``num_bytes / avg_duration_sec``).
    gbps = (nbytes / per_iter) / 1e9
    del h_in, d_in, d_out, h_out
    return per_iter, gbps


def probe_hardware(
    *,
    device: torch.device | str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    matmul_n: int = 4096,
    mem_bw_n: int = 8192,
    transfer_mib: int = 256,
    n_matmul_warmup: int = 3,
    n_matmul_reps: int = 10,
    n_mem_bw_warmup: int = 3,
    n_mem_bw_reps: int = 20,
    n_transfer_warmup: int = 2,
    n_transfer_reps: int = 5,
) -> HardwareProbeResult:
    """One-shot hardware probe sized to be fast (<1s on H100/A100) yet
    accurate enough for DP-solver inputs.

    Defaults are tuned so the matmul probe is well above the
    arithmetic-intensity roofline (4096^3 fp16 matmul ≈ 137 GFLOPs,
    plenty above kernel-launch overhead) and the transfer probe is
    large enough that PCIe steady-state dominates over per-call latency.
    Override the ``n_*_reps`` knobs for higher precision.
    """
    device = torch.device(device)

    matmul_per_call, achieved_tflops = _bench_matmul(
        matmul_n, dtype=dtype, device=device,
        n_warmup=n_matmul_warmup, n_reps=n_matmul_reps,
    )

    mem_bw_gbps = _bench_mem_bandwidth(
        mem_bw_n, dtype=dtype, device=device,
        n_warmup=n_mem_bw_warmup, n_reps=n_mem_bw_reps,
    )

    transfer_bytes = transfer_mib * (1 << 20)
    transfer_per_call, gbps = _bench_pcie_concurrent(
        transfer_bytes, device=device,
        n_warmup=n_transfer_warmup, n_reps=n_transfer_reps,
    )

    torch.cuda.empty_cache()

    hw_cost = HardwareCost(
        peak_tflops=achieved_tflops,
        pcie_bw_gbps=gbps,
    )
    return HardwareProbeResult(
        hw_cost=hw_cost,
        matmul_n=matmul_n,
        matmul_per_call_ms=matmul_per_call * 1e3,
        transfer_bytes=transfer_bytes,
        transfer_per_call_ms=transfer_per_call * 1e3,
        mem_bw_gbps=mem_bw_gbps,
    )
