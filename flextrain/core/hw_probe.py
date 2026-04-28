"""Light-weight hardware probe for DP-solver inputs.

The DP solver in :mod:`flextrain.core.save_level` needs:

* sustained device throughput in TFLOPS (for ``HardwareCost.flops_to_ms``);
* unidirectional concurrent host<->device PCIe bandwidth in GB/s (for
  ``HardwareCost.bytes_to_ms``);
* device memory bandwidth in GB/s (for the working-set solver's
  arithmetic-intensity bound).

Why "sustained" and not "peak"
------------------------------
Consumer cards (3090, 4090) and even datacenter cards under modest
cooling start at boost clocks and drop to base clocks within a few
seconds of sustained matmul load. A 10-rep, ~50ms probe sees the boost
window only -- it reports peak-burst TFLOPS, not what training will
actually sustain. The DP solver then thinks the per-task arrival window
is small (compute is "fast") and saves everything at the highest tier
because transfers seem to fit; on real training the throughput drops
and saves spill and stall.

The fix is wall-time bounded probing: run matmuls for at least a few
seconds, then return the **second-half average** so the steady-state
dominates the answer (the first half captures the boost-to-base
transition; throwing it out gives a cleaner sustained number).

Defaults run for ~3-4 seconds total; pass ``matmul_target_seconds`` to
trade speed for precision.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .save_level import HardwareCost


@dataclass(frozen=True)
class HardwareProbeResult:
    """Outputs of :func:`probe_hardware`. Wraps :class:`HardwareCost`
    plus extra fields useful for logging.

    Attributes
    ----------
    matmul_total_seconds, mem_bw_total_seconds
        Wall time the corresponding probe actually ran for. Useful for
        verifying you sized ``*_target_seconds`` long enough for
        thermal/clock throttling to settle.
    achieved_tflops_first_half, achieved_tflops_second_half
        Throughput in each half of the matmul probe. A noticeable drop
        from first to second half is the signature of thermal/clock
        throttling kicking in -- the second-half number is the one fed
        into ``hw_cost``.
    """

    hw_cost: HardwareCost
    matmul_n: int
    matmul_per_call_ms: float
    matmul_total_seconds: float
    achieved_tflops_first_half: float
    achieved_tflops_second_half: float
    transfer_bytes: int
    transfer_per_call_ms: float
    mem_bw_gbps: float
    mem_bw_total_seconds: float


def _bench_matmul_sustained(
    n: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    target_seconds: float,
    n_warmup: int,
    min_reps_per_half: int,
) -> tuple[float, float, float, float, float]:
    """Run an n x n bf16 matmul long enough for steady-state to settle.

    Returns ``(per_call_seconds, achieved_tflops_overall,
    achieved_tflops_first_half, achieved_tflops_second_half,
    total_seconds)``. Use ``second_half`` as the DP-solver input -- it
    excludes the boost-clock window at the top of the run.

    Implementation: warmup, then run two wall-clock-bounded halves of
    ``target_seconds / 2`` each. Each half launches matmuls in a tight
    loop until the elapsed wall time crosses the budget, syncing once
    at the end. This avoids per-call calibration: on fast hardware
    (H100, ~0.18 ms per 4096^3 matmul) we naturally fit ~8000 reps in
    1.5 s; on slow hardware we still run for the full target time,
    just with fewer reps.

    ``min_reps_per_half`` is a floor used only when the device is so
    slow that even a single sync'd matmul exceeds the per-half budget
    (rare; e.g. if you bumped ``n`` aggressively).
    """
    A = torch.randn(n, n, dtype=dtype, device=device)
    B = torch.randn(n, n, dtype=dtype, device=device)

    for _ in range(n_warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize(device)

    half_budget_seconds = target_seconds / 2.0

    def _run_half() -> tuple[int, float]:
        """Launch matmuls until the wall budget expires; sync once at the
        end. Returns ``(n_reps, elapsed_seconds)`` where ``elapsed`` is
        measured wall-to-wall around the launches and the trailing sync.
        """
        reps = 0
        start = time.perf_counter_ns()
        deadline_ns = start + int(half_budget_seconds * 1e9)
        # Burst-then-sync: launch matmuls in batches of ~50 between
        # wall-clock checks. Checking the clock every iteration adds
        # measurable overhead on fast kernels (~hundreds of ns each).
        burst = 50
        while True:
            for _ in range(burst):
                _ = torch.matmul(A, B)
            reps += burst
            if (
                time.perf_counter_ns() >= deadline_ns
                and reps >= min_reps_per_half
            ):
                break
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        return reps, elapsed

    reps_h1, h1 = _run_half()
    reps_h2, h2 = _run_half()

    total = h1 + h2
    n_reps = reps_h1 + reps_h2

    flops_per_call = 2.0 * n * n * n
    tflops_overall = (flops_per_call * n_reps / total) / 1e12 if total > 0 else 0.0
    tflops_h1 = (flops_per_call * reps_h1 / h1) / 1e12 if h1 > 0 else 0.0
    tflops_h2 = (flops_per_call * reps_h2 / h2) / 1e12 if h2 > 0 else 0.0
    per_call_overall = total / n_reps if n_reps > 0 else 0.0

    del A, B
    return per_call_overall, tflops_overall, tflops_h1, tflops_h2, total


def _bench_mem_bandwidth_sustained(
    n: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    target_seconds: float,
    n_warmup: int,
    min_reps_per_half: int,
) -> tuple[float, float]:
    """Memory-bandwidth probe (``(1, n) @ (n, n)`` matmul). Returns
    ``(gbps_second_half, total_seconds)``. Same wall-clock-bounded
    strategy as :func:`_bench_matmul_sustained` -- memory bandwidth on
    a clocked-down GPU is also lower than peak, so we want sustained
    numbers.
    """
    A = torch.randn(1, n, dtype=dtype, device=device)
    B = torch.randn(n, n, dtype=dtype, device=device)

    for _ in range(n_warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize(device)

    half_budget_seconds = target_seconds / 2.0

    def _run_half() -> tuple[int, float]:
        reps = 0
        start = time.perf_counter_ns()
        deadline_ns = start + int(half_budget_seconds * 1e9)
        burst = 50
        while True:
            for _ in range(burst):
                _ = torch.matmul(A, B)
            reps += burst
            if (
                time.perf_counter_ns() >= deadline_ns
                and reps >= min_reps_per_half
            ):
                break
        torch.cuda.synchronize(device)
        elapsed = (time.perf_counter_ns() - start) / 1e9
        return reps, elapsed

    # Discard first half (boost-clock window); measure second half.
    _reps_h1, h1 = _run_half()
    reps_h2, h2 = _run_half()

    per_call = h2 / reps_h2 if reps_h2 > 0 else 0.0
    bytes_touched = dtype.itemsize * (n * n + 2 * n)
    gbps = (bytes_touched / per_call) / 1e9 if per_call > 0 else 0.0

    del A, B
    return gbps, h1 + h2


def _bench_pcie_concurrent(
    nbytes: int,
    *,
    device: torch.device,
    n_warmup: int,
    n_reps: int,
) -> tuple[float, float]:
    """Concurrent host<->device PCIe transfer. PCIe bandwidth doesn't
    thermal-throttle the way matmul does (the link controller stays at
    the negotiated PCIe gen rate), so this stays at a small rep count.
    Returns ``(per_iter_seconds, gbps)``.
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
    matmul_target_seconds: float = 10.0,
    mem_bw_target_seconds: float = 4.0,
    n_matmul_warmup: int = 5,
    n_mem_bw_warmup: int = 5,
    min_matmul_reps_per_half: int = 50,
    min_mem_bw_reps_per_half: int = 50,
    n_transfer_warmup: int = 2,
    n_transfer_reps: int = 5,
) -> HardwareProbeResult:
    """Probe sustained (post-throttle) compute + memory + PCIe bandwidth.

    Total wall time is roughly ``matmul_target_seconds + mem_bw_target_seconds``
    plus a small PCIe component (defaults: ~14s). Tuned so consumer GPUs
    have time to drop from boost clocks to sustained clocks before the
    measurement window closes -- if you only ran for ~50ms you'd see
    boost-clock TFLOPS, which is misleading for sizing the DP solver.

    Each probe runs in two equal-budget wall-clock halves: matmuls
    launch in tight bursts until each half's budget expires, with a
    sync at the end. The first half captures the boost-to-base clock
    transition, the second is steady state -- the second-half
    throughput is what feeds ``hw_cost``.

    Set ``matmul_target_seconds`` higher (e.g. 30+) if you suspect the
    GPU takes longer than 5 s to thermally settle (multi-GPU rigs with
    shared cooling, datacenter cards under heavy ambient load).

    Returns a :class:`HardwareProbeResult` whose ``hw_cost`` field
    contains the second-half (sustained) numbers; the per-half
    breakdown is also exposed so you can detect throttling for logging.
    """
    device = torch.device(device)

    (
        matmul_per_call,
        _achieved_tflops_overall,
        tflops_h1,
        tflops_h2,
        matmul_total,
    ) = _bench_matmul_sustained(
        matmul_n, dtype=dtype, device=device,
        target_seconds=matmul_target_seconds,
        n_warmup=n_matmul_warmup,
        min_reps_per_half=min_matmul_reps_per_half,
    )

    mem_bw_gbps, mem_bw_total = _bench_mem_bandwidth_sustained(
        mem_bw_n, dtype=dtype, device=device,
        target_seconds=mem_bw_target_seconds,
        n_warmup=n_mem_bw_warmup,
        min_reps_per_half=min_mem_bw_reps_per_half,
    )

    transfer_bytes = transfer_mib * (1 << 20)
    transfer_per_call, gbps = _bench_pcie_concurrent(
        transfer_bytes, device=device,
        n_warmup=n_transfer_warmup, n_reps=n_transfer_reps,
    )

    torch.cuda.empty_cache()

    hw_cost = HardwareCost(
        peak_tflops=tflops_h2,  # sustained second-half average
        pcie_bw_gbps=gbps,
    )
    return HardwareProbeResult(
        hw_cost=hw_cost,
        matmul_n=matmul_n,
        matmul_per_call_ms=matmul_per_call * 1e3,
        matmul_total_seconds=matmul_total,
        achieved_tflops_first_half=tflops_h1,
        achieved_tflops_second_half=tflops_h2,
        transfer_bytes=transfer_bytes,
        transfer_per_call_ms=transfer_per_call * 1e3,
        mem_bw_gbps=mem_bw_gbps,
        mem_bw_total_seconds=mem_bw_total,
    )
