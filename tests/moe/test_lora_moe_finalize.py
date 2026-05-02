"""Unit test for the deferred-LoRA-wgrad finalize math
(``LoRAWrapperLayer._accumulate_moe_lora_grads_from_capture``).

Builds a small MoE-LoRA setup with known random tensors, runs the
finalize, and compares dA/dB against an autograd reference that
computes the same gradients via PyTorch's per-expert math.
Backend-agnostic — the finalize only consumes (X, dY, A, B, scale,
offs); backends produce these via their own bwd.

When ``torch.nn.functional.grouped_mm`` is available (torch 2.10+,
sm_80+, bf16), the test runs both the grouped_mm fast path AND the
per-expert fallback path, comparing each to the reference.

When grouped_mm is NOT available, only the fallback runs.

Run from repo root with the CUDA runtime libs on LD_LIBRARY_PATH:
  PYTHONPATH=. python tests/moe/test_lora_moe_finalize.py
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from flextrain.nn.layers.lora_wrapper import (
    LoRAWrapperLayer,
    LoRATargetConfig,
)


def _make_groups(TK: int, E: int, device, dtype=torch.int32):
    """Random per-expert split summing to TK. Returns (counts, offsets)."""
    # Random non-negative ints summing to TK. Simple: split TK / E with some
    # jitter to test variable T_e.
    base = TK // E
    counts = torch.full((E,), base, dtype=torch.int64, device=device)
    rem = TK - base * E
    counts[:rem] += 1
    perm = torch.randperm(E, device=device)
    counts = counts[perm]
    assert counts.sum().item() == TK
    offs = counts.cumsum(0).to(dtype)
    return counts, offs


def _autograd_reference(
    X: torch.Tensor,    # (TK, in) bf16
    dY: torch.Tensor,   # (TK, out) bf16
    A: torch.Tensor,    # (E, in, r) bf16
    B: torch.Tensor,    # (E, r, out) bf16
    counts: torch.Tensor,  # (E,)
    scale: float,
):
    """Compute dA, dB per expert via per-expert PyTorch matmul.

    Math:
      Per expert e (with X_e = X[start:end], dY_e = dY[start:end]):
        dY_B_e = dY_e @ B[e]^T          (T_e, r)
        dA[e]  = X_e^T @ dY_B_e * scale (in, r)
        X_A_e  = X_e @ A[e]             (T_e, r)
        dB[e]  = X_A_e^T @ dY_e * scale (r, out)
    """
    E = A.shape[0]
    in_dim, r = A.shape[1], A.shape[2]
    out_dim = B.shape[2]
    dA = torch.zeros((E, in_dim, r), dtype=torch.float32, device=X.device)
    dB = torch.zeros((E, r, out_dim), dtype=torch.float32, device=X.device)
    cur = 0
    for e in range(E):
        T_e = int(counts[e].item())
        if T_e == 0:
            continue
        X_e = X[cur:cur+T_e].float()      # (T_e, in)
        dY_e = dY[cur:cur+T_e].float()    # (T_e, out)
        A_e = A[e].float()                # (in, r)
        B_e = B[e].float()                # (r, out)

        dY_B_e = dY_e @ B_e.transpose(-1, -2)   # (T_e, r)
        dA[e] = (X_e.transpose(-1, -2) @ dY_B_e) * scale

        X_A_e = X_e @ A_e                 # (T_e, r)
        dB[e] = (X_A_e.transpose(-1, -2) @ dY_e) * scale

        cur += T_e
    return dA, dB


def _diffstats(label, a, b, tol=0.99):
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    diff = (a_f - b_f)
    cos = torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0), dim=-1
    ).item()
    abs_diff = diff.abs()
    out = {
        "cos": cos,
        "mean_diff": diff.mean().item(),
        "std_diff": diff.std().item(),
        "mean_abs": abs_diff.mean().item(),
        "max_abs": abs_diff.max().item(),
        "ref_scale": b_f.abs().mean().item(),
    }
    ok = cos >= tol
    print(
        f"  {'OK ' if ok else 'FAIL'}  {label:30s} cos={cos:.6f}  "
        f"mean_diff={diff.mean().item():+.3e}  "
        f"max_abs={abs_diff.max().item():.3e}  "
        f"ref_scale={out['ref_scale']:.3e}"
    )
    return ok


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=int, default=2048)
    ap.add_argument("-K", type=int, default=8)
    ap.add_argument("-E", type=int, default=64)
    ap.add_argument("-F", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--cos-tol", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    T, K, E, F_dim, d, r = args.T, args.K, args.E, args.F, args.d_model, args.rank
    TK = T * K
    print(f"=== test_lora_moe_finalize: TK={TK} E={E} d={d} F={F_dim} r={r} "
          f"scale={args.scale} ===")

    # Random A/B for w_up and w_down.
    weights = {
        "w_up_lora_a":   torch.randn(E, d, r,         device=device, dtype=dtype),
        "w_up_lora_b":   torch.randn(E, r, 2 * F_dim, device=device, dtype=dtype),
        "w_down_lora_a": torch.randn(E, F_dim, r,     device=device, dtype=dtype),
        "w_down_lora_b": torch.randn(E, r, d,         device=device, dtype=dtype),
    }
    grads = {
        "g_up_lora_a":   torch.zeros_like(weights["w_up_lora_a"]),
        "g_up_lora_b":   torch.zeros_like(weights["w_up_lora_b"]),
        "g_down_lora_a": torch.zeros_like(weights["w_down_lora_a"]),
        "g_down_lora_b": torch.zeros_like(weights["w_down_lora_b"]),
    }

    # Build a "capture" dict the way a backend would.
    counts, offs = _make_groups(TK, E, device)
    scattered_x_grouped = torch.randn(TK, d, device=device, dtype=dtype)
    dx_up_up_grouped = torch.randn(TK, 2 * F_dim, device=device, dtype=dtype)
    scattered_upstream_grouped = torch.randn(TK, d, device=device, dtype=dtype)
    # x_up_grouped: random pre-SwiGLU. fwd_act = silu(gate) * value computed
    # by the finalize. Our autograd reference does the same recompute.
    x_up_grouped = torch.randn(TK, 2 * F_dim, device=device, dtype=dtype)

    capture = {
        "scattered_x_grouped": scattered_x_grouped,
        "dx_up_up_grouped": dx_up_up_grouped,
        "scattered_upstream_grouped": scattered_upstream_grouped,
        "x_up_grouped": x_up_grouped,
        "expert_offsets": offs,
        "TK": TK,
    }

    # Stub LoRAWrapperLayer with just the targets/_target_set/_MOE_CALLBACK_TARGETS
    # we need for _accumulate_moe_lora_grads_from_capture.
    class _StubWrapper:
        targets = (
            LoRATargetConfig(target_name="w_up",   rank=r, alpha=float(r) * args.scale),
            LoRATargetConfig(target_name="w_down", rank=r, alpha=float(r) * args.scale),
        )
        _target_set = frozenset(("w_up", "w_down"))
        _MOE_CALLBACK_TARGETS = LoRAWrapperLayer._MOE_CALLBACK_TARGETS

    stub = _StubWrapper()

    # Reference: compute dA, dB per expert via PyTorch.
    # w_up: X = scattered_x_grouped, dY = dx_up_up_grouped
    dA_up_ref, dB_up_ref = _autograd_reference(
        scattered_x_grouped, dx_up_up_grouped,
        weights["w_up_lora_a"], weights["w_up_lora_b"],
        counts, args.scale,
    )
    # w_down: X = silu(gate)*value from x_up_grouped (chunked [up, gate]),
    # dY = scattered_upstream_grouped
    value, gate = x_up_grouped.chunk(2, dim=-1)
    fwd_act_ref = (F.silu(gate.float()) * value.float()).to(dtype)
    dA_down_ref, dB_down_ref = _autograd_reference(
        fwd_act_ref, scattered_upstream_grouped,
        weights["w_down_lora_a"], weights["w_down_lora_b"],
        counts, args.scale,
    )

    def _run_finalize_and_compare(path_name: str, force_fallback: bool):
        """Zero the grads dict, run the finalize, compare to reference."""
        for k in grads:
            grads[k].zero_()

        # Optionally hide grouped_mm so the finalize takes the fallback
        # branch. Restore after.
        saved = getattr(torch.nn.functional, "grouped_mm", None)
        if force_fallback and saved is not None:
            del torch.nn.functional.grouped_mm
        try:
            LoRAWrapperLayer._accumulate_moe_lora_grads_from_capture(
                stub, capture, weights, grads,
            )
            torch.cuda.synchronize()
        finally:
            if force_fallback and saved is not None:
                torch.nn.functional.grouped_mm = saved

        print(f"\n=== {path_name} vs per-expert reference ===")
        path_fail = False
        for label, got, ref in [
            ("g_up_lora_a (E, d, r)",   grads["g_up_lora_a"],   dA_up_ref),
            ("g_up_lora_b (E, r, 2F)",  grads["g_up_lora_b"],   dB_up_ref),
            ("g_down_lora_a (E, F, r)", grads["g_down_lora_a"], dA_down_ref),
            ("g_down_lora_b (E, r, d)", grads["g_down_lora_b"], dB_down_ref),
        ]:
            if not _diffstats(label, got, ref, tol=args.cos_tol):
                path_fail = True
        return path_fail

    has_grouped_mm = hasattr(torch.nn.functional, "grouped_mm")

    fail = False
    if has_grouped_mm:
        if _run_finalize_and_compare(
            "grouped_mm fast path", force_fallback=False,
        ):
            fail = True
    else:
        print("\n[skip] grouped_mm not available (torch < 2.10) — "
              "skipping fast-path test.")
    # Always test the fallback path — both for boxes without grouped_mm
    # and as a regression guard alongside the fast path.
    if _run_finalize_and_compare(
        "per-expert fallback", force_fallback=True,
    ):
        fail = True

    print()
    if fail:
        print("FAIL — LoRA finalize disagrees with per-expert reference.")
        sys.exit(1)
    print("PASS — LoRA finalize matches per-expert reference within bf16 noise.")


if __name__ == "__main__":
    main()
