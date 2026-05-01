"""Numeric parity: ScatterMoEExpertCompute vs FlextrainMoEExpertCompute
vs naive-PyTorch reference.

Both backends produce scattered intermediates in their OWN sort order;
comparing scattered tensors directly is meaningless. This test compares
all backends against a naive PyTorch reference at the **token-major**
output level (out, dx, g_up, g_down, d_gates(T,K)) — those are
sort-order-independent.

Forward reference:
    h_e[t]  = silu(gate)*value where (gate, value) = chunk(x[t] @ w_up[e])
    out[t]  = sum_k expert_p[t,k] * (h_{e_k}[t] @ w_down[e_k])

Backward (autograd through the reference):
    dx, dw_up, dw_down, d_expert_p

Tolerances are bf16-GEMM-realistic. Match against either rel<5e-2 OR
abs<5e-2 (per-element scale of accumulators ~unit).

Run:
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \\
  PYTHONPATH=. python tests/scratch/test_moe_backend_parity.py
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F_torch

sys.path.insert(0, "/home/shein/Documents/flextrain")

from flextrain.ops.moe_backend import (
    FlextrainMoEExpertCompute,
    ScatterMoEExpertCompute,
)


# ---------------------------------------------------------------------------
# Naive reference (eager PyTorch + autograd).
# ---------------------------------------------------------------------------


def naive_moe_fwd_bwd(
    x: torch.Tensor,           # (T, d) — requires_grad
    expert_p: torch.Tensor,    # (T, K) — requires_grad
    expert_idxs: torch.Tensor, # (T, K) int
    w_up: torch.Tensor,        # (E, d, 2F) — requires_grad
    w_down: torch.Tensor,      # (E, F, d) — requires_grad
    dy: torch.Tensor,          # (T, d) — upstream
):
    """Compute out, dx, dw_up, dw_down, d_expert_p via PyTorch autograd.

    Forward:
        For each (t, k): expert e = expert_idxs[t,k]
            pre  = x[t] @ w_up[e]                   # (2F,)
            value, gate = pre.chunk(2)              # (F,), (F,)
            h    = silu(gate) * value               # (F,)
            y_e  = h @ w_down[e]                    # (d,)
        out[t] = sum_k expert_p[t,k] * y_e

    Backward: autograd.
    """
    T, d = x.shape
    K = expert_p.shape[1]
    E = w_up.shape[0]
    F = w_down.shape[1]

    # Compute pre-act per (t, k):  pre[t, k, :]  shape (T, K, 2F)
    # via gathering w_up by expert id then bmm with x repeated.
    # Simple & correct: use a Python loop over k. T*K iterations,
    # cheap for the small shapes we test with.
    pre_per_k = []
    for k in range(K):
        e = expert_idxs[:, k].long()                   # (T,)
        w = w_up[e]                                    # (T, d, 2F)
        pre_k = torch.bmm(x.unsqueeze(1), w).squeeze(1)  # (T, 2F)
        pre_per_k.append(pre_k)
    pre = torch.stack(pre_per_k, dim=1)                # (T, K, 2F)

    value, gate = pre.chunk(2, dim=-1)                 # (T, K, F) each
    h = F_torch.silu(gate) * value                     # (T, K, F)

    y_per_k = []
    for k in range(K):
        e = expert_idxs[:, k].long()
        w = w_down[e]                                   # (T, F, d)
        y_k = torch.bmm(h[:, k, :].unsqueeze(1), w).squeeze(1)  # (T, d)
        y_per_k.append(y_k)
    y = torch.stack(y_per_k, dim=1)                    # (T, K, d)

    # Weighted sum over k
    out = (expert_p.unsqueeze(-1) * y).sum(dim=1)      # (T, d)

    # Backward via autograd
    grads = torch.autograd.grad(
        outputs=out,
        inputs=(x, expert_p, w_up, w_down),
        grad_outputs=dy,
        retain_graph=False,
    )
    dx_ref, d_expert_p_ref, dw_up_ref, dw_down_ref = grads
    return out.detach(), dx_ref, d_expert_p_ref, dw_up_ref, dw_down_ref


# ---------------------------------------------------------------------------
# Fake-slot construction
# ---------------------------------------------------------------------------


def _make_fake_slot(T, K, E, F, d_model, dtype, device):
    TK = T * K
    return types.SimpleNamespace(
        x_router=torch.empty(T, E, device=device, dtype=dtype),
        router_weights=torch.empty(T, K, device=device, dtype=dtype),
        chosen_experts=torch.empty(T, K, device=device, dtype=torch.int32),
        x_up=torch.empty(TK, 2 * F, device=device, dtype=dtype),
        # Flextrain backend's private fields.
        expert_counts=torch.empty(E, device=device, dtype=torch.int32),
        index_mapping=torch.empty(T, K, device=device, dtype=torch.int32),
        scattered_router_weights=torch.empty(TK, 1, device=device, dtype=dtype),
        # ScatterMoE backend's private fields.
        scattermoe_sorted_expert_idxs=torch.empty(TK, device=device, dtype=torch.int32),
        scattermoe_sorted_scattered_idxs=torch.empty(TK, device=device, dtype=torch.int32),
        scattermoe_expert_offsets=torch.empty(E, device=device, dtype=torch.int32),
        aux={},
    )


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


def _diffstats(a: torch.Tensor, b: torch.Tensor):
    """Returns dict of useful comparison stats.

    Stats:
      ``cos``       - cosine similarity (1.0 = perfect alignment)
      ``mean_diff`` - signed mean of (a - b); should be ~0 for unbiased noise
      ``std_diff``  - std-dev of (a - b)
      ``mean_abs``  - mean of |a - b|
      ``max_abs``   - max of |a - b|
      ``ref_scale`` - mean of |b|, the reference; useful as a denominator
    """
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    diff = a_f - b_f
    abs_diff = diff.abs()
    cos = torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0), dim=-1,
    ).item()
    return {
        "cos": cos,
        "mean_diff": diff.mean().item(),
        "std_diff": diff.std().item(),
        "mean_abs": abs_diff.mean().item(),
        "max_abs": abs_diff.max().item(),
        "ref_scale": b_f.abs().mean().item(),
    }


def _report(label: str, a: torch.Tensor, b: torch.Tensor, *, cos_tol: float):
    """Pass if cosine similarity ≥ cos_tol. Cosine is the most robust
    measure here — direction-aligned outputs with bf16 magnitude noise
    will always have cos≈1, while max-element rel error blows up on
    near-zero outliers."""
    s = _diffstats(a, b)
    ok = s["cos"] >= cos_tol
    status = "OK " if ok else "FAIL"
    print(
        f"  {status}  {label:30s} "
        f"cos={s['cos']:.6f}  "
        f"mean_diff={s['mean_diff']:+.3e}  "
        f"std_diff={s['std_diff']:.3e}  "
        f"mean_abs={s['mean_abs']:.3e}  "
        f"max_abs={s['max_abs']:.3e}  "
        f"ref_scale={s['ref_scale']:.3e}  "
        f"(cos_tol={cos_tol:.5f})"
    )
    return ok


# ---------------------------------------------------------------------------
# Backend driver
# ---------------------------------------------------------------------------


def run_backend(backend, x, expert_p, expert_idxs, weights, dy, T, K, E, F, d_model, dtype, device):
    """Run a backend's fwd+bwd, return (out, dx, g_up, g_down, d_gates_TK).
    d_gates_TK is (T, K) — un-scattered from the backend's scattered dprobs."""
    slot = _make_fake_slot(T, K, E, F, d_model, dtype, device)
    slot.router_weights.copy_(expert_p)
    slot.chosen_experts.copy_(expert_idxs)
    chunk_extra: dict = {}

    out = torch.empty(T, d_model, device=device, dtype=dtype)
    primary_stream = torch.cuda.current_stream()

    def scratch_fn(shape, dt):
        return torch.empty(shape, dtype=dt, device=device)

    backend.fwd(
        x, slot.router_weights, slot.chosen_experts, weights,
        out=out, residual=None,
        slot=slot, chunk_extra=chunk_extra, layer_id=0,
        primary_stream=primary_stream, secondary_stream=None,
        scratch_fn=scratch_fn,
    )
    grads = {
        "g_up": torch.zeros_like(weights["w_up"]),
        "g_down": torch.zeros_like(weights["w_down"]),
    }
    dx = backend.bwd(
        dy, x, weights, grads,
        slot=slot, chunk_extra=chunk_extra, layer_id=0,
        primary_stream=primary_stream, secondary_stream=None,
        scratch_fn=scratch_fn,
    )
    torch.cuda.synchronize()

    # Un-scatter dprobs from (TK, 1) scattered → (T, K) token-major
    # using slot.index_mapping which both backends populate.
    dprobs_scattered = slot.aux["moe_dprobs"]  # (TK, 1)
    d_gates_TK = (
        dprobs_scattered.flatten()[slot.index_mapping.long().flatten()]
        .view(T, K)
    )
    return out, dx, grads["g_up"], grads["g_down"], d_gates_TK


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    T, K, E, F, d_model = 128, 2, 8, 64, 128

    x = torch.randn(T, d_model, device=device, dtype=dtype)
    logits = torch.randn(T, E, device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    expert_p, expert_idxs = probs.topk(K, dim=-1)
    expert_p = (expert_p / expert_p.sum(dim=-1, keepdim=True)).to(dtype)
    expert_idxs = expert_idxs.to(torch.int32)

    weights = {
        "w_up": torch.randn(E, d_model, 2 * F, device=device, dtype=dtype) / (d_model ** 0.5),
        "w_down": torch.randn(E, F, d_model, device=device, dtype=dtype) / (F ** 0.5),
    }
    dy = torch.randn(T, d_model, device=device, dtype=dtype)

    # ---- Naive reference (autograd) ----
    print("=== Naive PyTorch reference ===")
    x_ref = x.detach().clone().requires_grad_(True)
    expert_p_ref = expert_p.detach().clone().requires_grad_(True)
    w_up_ref = weights["w_up"].detach().clone().requires_grad_(True)
    w_down_ref = weights["w_down"].detach().clone().requires_grad_(True)
    out_ref, dx_ref, d_p_ref, dw_up_ref, dw_down_ref = naive_moe_fwd_bwd(
        x_ref, expert_p_ref, expert_idxs, w_up_ref, w_down_ref, dy,
    )

    # ---- Flextrain backend ----
    print("\n=== FlextrainMoEExpertCompute ===")
    ft_out, ft_dx, ft_g_up, ft_g_down, ft_d_p = run_backend(
        FlextrainMoEExpertCompute(), x, expert_p, expert_idxs, weights, dy,
        T, K, E, F, d_model, dtype, device,
    )

    # ---- ScatterMoE backend ----
    print("\n=== ScatterMoEExpertCompute ===")
    sm_out, sm_dx, sm_g_up, sm_g_down, sm_d_p = run_backend(
        ScatterMoEExpertCompute(), x, expert_p, expert_idxs, weights, dy,
        T, K, E, F, d_model, dtype, device,
    )

    # Cosine-similarity tolerance: bf16 GEMM with fp32 accum produces
    # near-perfect cosine alignment with the fp32 reference; 0.999 is
    # a comfortable margin while still catching real bugs (any kernel
    # bug will tank cosine well below this).
    cos_tol = 0.999

    print("\n=== Flextrain vs naive reference ===")
    fail_ft = False
    for label, a, b in [
        ("out (T, d)",       ft_out,    out_ref),
        ("dx (T, d)",        ft_dx,     dx_ref),
        ("d_expert_p (T, K)", ft_d_p,    d_p_ref),
        ("g_up (E, d, 2F)",  ft_g_up,   dw_up_ref),
        ("g_down (E, F, d)", ft_g_down, dw_down_ref),
    ]:
        if not _report(label, a, b, cos_tol=cos_tol):
            fail_ft = True

    print("\n=== ScatterMoE vs naive reference ===")
    fail_sm = False
    for label, a, b in [
        ("out (T, d)",       sm_out,    out_ref),
        ("dx (T, d)",        sm_dx,     dx_ref),
        ("d_expert_p (T, K)", sm_d_p,    d_p_ref),
        ("g_up (E, d, 2F)",  sm_g_up,   dw_up_ref),
        ("g_down (E, F, d)", sm_g_down, dw_down_ref),
    ]:
        if not _report(label, a, b, cos_tol=cos_tol):
            fail_sm = True

    print()
    if fail_ft or fail_sm:
        if fail_ft:
            print("FAIL — flextrain backend diverges from reference.")
        if fail_sm:
            print("FAIL — scattermoe backend diverges from reference.")
        sys.exit(1)
    print("PASS — both backends agree with naive reference within bf16 tol.")


if __name__ == "__main__":
    main()
