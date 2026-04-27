"""Unit-test the fused router fwd+bwd kernels against a pure-PyTorch
reference for both modes (topk_then_softmax, softmax_then_topk).

Forward: kernel output should match PyTorch topk+softmax construction
to within fp32→bf16 cast noise.

Backward: upstream dprobs are pre-gathered into scatter order (matching
how the expert loop emits them); dlogits must match autograd w.r.t. the
corresponding fwd path.
"""
from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.ops import (
    flextrain_fused_topk_softmax,
    flextrain_moe_router_gate_bwd,
    flextrain_moe_sort,
)


DEVICE = "cuda:0"


def _ref_fwd(logits: torch.Tensor, top_k: int, mode: str):
    """PyTorch reference. Returns (topk_weights, topk_ids)."""
    if mode == "topk_then_softmax":
        topk_logits, topk_ids = torch.topk(logits, top_k, dim=-1)
        topk_w = torch.softmax(topk_logits.float(), dim=-1).to(logits.dtype)
    else:
        probs = torch.softmax(logits.float(), dim=-1)
        topk_w_f, topk_ids = torch.topk(probs, top_k, dim=-1)
        topk_w = topk_w_f.to(logits.dtype)
    return topk_w, topk_ids.to(torch.int32)


def _test_fwd(mode: str, T=64, E=16, K=4, dtype=torch.bfloat16):
    torch.manual_seed(0)
    logits = torch.randn(T, E, device=DEVICE, dtype=dtype)
    ids_out = torch.empty((T, K), dtype=torch.int32, device=DEVICE)
    w_out = torch.empty((T, K), dtype=dtype, device=DEVICE)

    w_k, ids_k = flextrain_fused_topk_softmax(
        logits, top_k=K,
        topk_ids_out=ids_out, topk_weights_out=w_out, mode=mode,
    )
    w_ref, ids_ref = _ref_fwd(logits, K, mode)

    # IDs must match exactly.
    assert torch.equal(ids_k.cpu(), ids_ref.cpu()), (
        f"[fwd {mode}] topk_ids mismatch: kernel={ids_k} ref={ids_ref}"
    )
    dmax = (w_k.float() - w_ref.float()).abs().max().item()
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    assert dmax < tol, f"[fwd {mode}] weights |Δ|={dmax:.4f} > {tol}"
    return dmax


def _ref_bwd_dlogits(
    logits: torch.Tensor, topk_ids: torch.Tensor,
    dprobs_tk: torch.Tensor, mode: str,
) -> torch.Tensor:
    """PyTorch-autograd reference for dlogits.

    ``dprobs_tk`` is dL/d(topk_weights), shape [T, K].
    Returns dL/d(logits), shape [T, E].
    """
    logits = logits.detach().clone().float().requires_grad_(True)
    if mode == "topk_then_softmax":
        # Simulate topk + softmax-over-K using gather at fixed topk_ids.
        topk_logits = logits.gather(1, topk_ids.long())
        w = torch.softmax(topk_logits, dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)
        w = probs.gather(1, topk_ids.long())
    # Backprop dprobs_tk into logits.
    (w * dprobs_tk.float()).sum().backward()
    return logits.grad


def _test_bwd(mode: str, T=64, E=16, K=4, dtype=torch.bfloat16):
    torch.manual_seed(1)
    logits = torch.randn(T, E, device=DEVICE, dtype=dtype)

    # Forward.
    ids_out = torch.empty((T, K), dtype=torch.int32, device=DEVICE)
    w_out = torch.empty((T, K), dtype=dtype, device=DEVICE)
    flextrain_fused_topk_softmax(
        logits, top_k=K,
        topk_ids_out=ids_out, topk_weights_out=w_out, mode=mode,
    )

    # Simulate an upstream dL/d(topk_weights) and scatter it into scatter-order
    # (same layout the kernel consumes).
    dprobs_tk = torch.randn(T, K, device=DEVICE, dtype=dtype)
    # Build sort indices via flextrain_moe_sort and scatter dprobs_tk.
    index_mapping, _ = flextrain_moe_sort(ids_out, num_experts=E)
    # dprobs_scattered[indices[t,k]] = dprobs_tk[t,k] → scatter
    dprobs_scattered = torch.empty(T * K, dtype=dtype, device=DEVICE)
    dprobs_scattered[index_mapping.view(-1).long()] = dprobs_tk.view(-1)

    dlogits = torch.zeros(T, E, device=DEVICE, dtype=dtype)
    flextrain_moe_router_gate_bwd(
        probs=w_out, dprobs=dprobs_scattered,
        indices=index_mapping, chosen_experts=ids_out,
        dlogits=dlogits,
        mode=mode, logits=logits if mode == "softmax_then_topk" else None,
    )

    ref_dlogits = _ref_bwd_dlogits(logits, ids_out, dprobs_tk, mode=mode)
    dmax = (dlogits.float() - ref_dlogits.float()).abs().max().item()
    tol = 5e-2 if dtype == torch.bfloat16 else 1e-4
    assert dmax < tol, (
        f"[bwd {mode}] dlogits |Δ|={dmax:.4f} > {tol}\n"
        f"  kernel[0]={dlogits[0].float().cpu()}\n"
        f"  ref[0]={ref_dlogits[0].float().cpu()}"
    )
    return dmax


def main():
    print("=== fwd parity ===")
    for mode in ("topk_then_softmax", "softmax_then_topk"):
        d = _test_fwd(mode)
        print(f"  {mode}: max |Δw| = {d:.4e}")

    print("=== bwd parity vs autograd ===")
    for mode in ("topk_then_softmax", "softmax_then_topk"):
        d = _test_bwd(mode)
        print(f"  {mode}: max |Δdlogits| = {d:.4e}")

    print("\n✓ routing kernel parity PASSED")


if __name__ == "__main__":
    main()
