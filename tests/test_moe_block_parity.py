"""Parity test for :class:`MoESwiGLUFFN` block vs pure PyTorch.

Runs a single forward+backward on random inputs and weights,
comparing FlexTrain's MoE block (using orig kernels) against a
hand-written PyTorch reference. Validates:

1. Forward output matches within bf16 noise.
2. Gradients w.r.t. input match within bf16 noise.
3. Weight gradients (g_router, g_up, g_down) match.

This test doesn't touch the AdaWS engine; it instantiates the block
directly and drives fwd/bwd with hand-built chunk / ctx objects. The
next step (pending) is to plug this block into the engine via a
``MoELlamaBlock`` layer.

Usage:
    PYTHONPATH=. python tests/test_moe_block_parity.py
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Naive PyTorch reference.
# ---------------------------------------------------------------------------


class NaiveMoESwiGLU(torch.nn.Module):
    """Pure-PyTorch MoE SwiGLU FFN.

    Forward:
        logits = h @ w_router                          # (T, E)
        topk_w, topk_i = softmax(topk(logits, K))      # (T, K)
        out = 0
        for k in range(K):
            for e in range(E):
                mask = (topk_i[:, k] == e)             # (T,) bool
                if not mask.any(): continue
                h_e = h[mask]                          # (T_e, d)
                up = h_e @ w_up[e]                     # (T_e, 2F)
                a, b = up.chunk(2, dim=-1)
                act = silu(a) * b                      # (T_e, F)
                down = act @ w_down[e]                 # (T_e, d)
                out[mask] += down * topk_w[mask, k:k+1]
    """

    def __init__(self, d_model, expert_dim, num_experts, top_k):
        super().__init__()
        self.d_model = d_model
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.w_router = torch.nn.Parameter(
            torch.zeros(d_model, num_experts, dtype=DTYPE)
        )
        # Weights match FT's stacked (E, d, 2F) and (E, F, d) layout.
        self.w_up = torch.nn.Parameter(
            torch.zeros(num_experts, d_model, 2 * expert_dim, dtype=DTYPE)
        )
        self.w_down = torch.nn.Parameter(
            torch.zeros(num_experts, expert_dim, d_model, dtype=DTYPE)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        T = h.shape[0]
        K = self.top_k
        E = self.num_experts
        logits = h @ self.w_router  # (T, E)
        # Top-k + softmax (on the top-k logits, per-token).
        topk_vals, topk_ids = torch.topk(logits, k=K, dim=-1)  # (T, K)
        topk_w = torch.softmax(topk_vals.float(), dim=-1).to(DTYPE)  # (T, K)

        out = torch.zeros(T, self.d_model, dtype=DTYPE, device=h.device)
        for e in range(E):
            # (T, K) mask for positions where expert e was chosen.
            mask = (topk_ids == e)  # bool
            if not mask.any():
                continue
            # For each (t, k) where expert e is chosen, weight = topk_w[t, k].
            # Aggregate all such (t, k) pairs into a flat list.
            tk_positions = mask.nonzero(as_tuple=False)  # (N_e, 2) = (t, k)
            t_idx = tk_positions[:, 0]
            k_idx = tk_positions[:, 1]
            h_e = h[t_idx]                      # (N_e, d)
            up = h_e @ self.w_up[e]             # (N_e, 2F)
            a, b = up.chunk(2, dim=-1)
            act = torch.nn.functional.silu(a.float()).to(DTYPE) * b  # (N_e, F)
            down = act @ self.w_down[e]         # (N_e, d)
            # Scale by router weight and scatter-add into out.
            scale = topk_w[t_idx, k_idx].unsqueeze(-1)  # (N_e, 1)
            out.index_add_(0, t_idx, down * scale)
        return out


# ---------------------------------------------------------------------------
# FlexTrain MoESwiGLUFFN driver (bypasses engine; constructs minimal ctx/slot).
# ---------------------------------------------------------------------------


class _MockSlot:
    """Minimal stand-in for ActivationSlot so the block can run without
    the engine. We pre-allocate every declared field up front."""

    def __init__(self, T: int, cfg):
        self.x_router = torch.zeros(T, cfg.num_experts, dtype=DTYPE, device=DEVICE)
        self.expert_counts = torch.zeros(cfg.num_experts, dtype=torch.int32, device=DEVICE)
        self.router_weights = torch.zeros(T, cfg.top_k, dtype=DTYPE, device=DEVICE)
        self.chosen_experts = torch.zeros(T, cfg.top_k, dtype=torch.int32, device=DEVICE)
        self.scattered_router_weights = torch.zeros(
            T * cfg.top_k, 1, dtype=DTYPE, device=DEVICE,
        )
        self.x_up = torch.zeros(
            T * cfg.top_k, 2 * cfg.expert_dim, dtype=DTYPE, device=DEVICE,
        )
        self.aux: dict = {}


class _MockCtx:
    """Minimal LayerContext with scratch only."""

    def __init__(self):
        self.total_tokens_per_step = None

    def scratch(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, device=DEVICE)


class _MockChunk:
    """Minimal ChunkMeta that carries MoE scratch in .extra."""

    def __init__(self, T: int, cfg):
        index_mapping = torch.zeros(
            T, cfg.top_k, dtype=torch.int32, device=DEVICE,
        )
        expert_counts_host = torch.zeros(
            cfg.num_experts, dtype=torch.int32, device="cpu", pin_memory=True,
        )
        self.extra = {
            "moe_token_index_mapping": {0: index_mapping},
            "moe_expert_counts_host": {0: expert_counts_host},
        }


def _run_ft_forward(block, cfg, h, weights):
    T = h.shape[0]
    slot = _MockSlot(T, cfg)
    ctx = _MockCtx()
    chunk = _MockChunk(T, cfg)
    residual = torch.zeros_like(h)  # pre-residual FFN = 0 for the test
    out_tensor = torch.zeros_like(h)
    y = block.fwd(
        h, weights, residual, out_tensor, slot, ctx, chunk, layer_id=0,
    )
    return y, slot, ctx, chunk


def _run_ft_backward(block, cfg, h, weights, slot, ctx, chunk, dy):
    # Provide ffn_norm_output for bwd (identity in this test: input h
    # goes straight in, no RMSNorm).
    slot.aux["recompute_ffn_norm_output"] = h
    grads = {
        "g_router": torch.zeros_like(weights["w_router"]),
        "g_up": torch.zeros_like(weights["w_up"]),
        "g_down": torch.zeros_like(weights["w_down"]),
    }
    dx = block.bwd(dy, weights, grads, slot, ctx, chunk, layer_id=0)
    return dx, grads


def main() -> None:
    from flextrain.nn.blocks.ffn_moe import MoESwiGLUConfig, MoESwiGLUFFN

    torch.manual_seed(4242)
    d_model = 128
    expert_dim = 256
    num_experts = 4
    top_k = 2
    T = 64

    cfg = MoESwiGLUConfig(
        d_model=d_model, expert_dim=expert_dim,
        num_experts=num_experts, top_k=top_k,
        compute_dtype=DTYPE,
    )
    block = MoESwiGLUFFN(cfg)

    # Shared weights across naive + FT.
    h = torch.randn(T, d_model, dtype=DTYPE, device=DEVICE, requires_grad=False)
    weights = {
        "w_router": torch.randn(d_model, num_experts, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_up": torch.randn(num_experts, d_model, 2 * expert_dim, dtype=DTYPE, device=DEVICE) * 0.02,
        "w_down": torch.randn(num_experts, expert_dim, d_model, dtype=DTYPE, device=DEVICE) * 0.02,
    }

    # Naive reference with PyTorch autograd.
    naive = NaiveMoESwiGLU(d_model, expert_dim, num_experts, top_k).to(DEVICE)
    with torch.no_grad():
        naive.w_router.copy_(weights["w_router"])
        naive.w_up.copy_(weights["w_up"])
        naive.w_down.copy_(weights["w_down"])

    h_naive = h.clone().requires_grad_(True)
    y_naive = naive(h_naive)
    # Choose a random upstream gradient to test backward.
    dy = torch.randn_like(y_naive) * 0.01
    y_naive.backward(dy)

    # FlexTrain run.
    y_ft, slot, ctx, chunk = _run_ft_forward(block, cfg, h, weights)
    dx_ft, grads_ft = _run_ft_backward(
        block, cfg, h, weights, slot, ctx, chunk, dy,
    )

    # Compare.
    def _cmp(name, a, b, atol=1e-2):
        delta = (a.float() - b.float()).abs()
        max_d = float(delta.max())
        mean_d = float(delta.mean())
        status = "OK" if max_d < atol else "MISMATCH"
        print(f"  {name:20s}  max|Δ|={max_d:.4f}  mean|Δ|={mean_d:.5f}  {status}")
        return max_d < atol

    print("\n=== Comparison ===")
    ok_fwd = _cmp("fwd output", y_naive.detach(), y_ft.detach(), atol=1e-2)
    ok_dx = _cmp("d_input", h_naive.grad, dx_ft, atol=1e-2)
    ok_router = _cmp("g_router", naive.w_router.grad, grads_ft["g_router"], atol=1e-2)
    ok_up = _cmp("g_up", naive.w_up.grad, grads_ft["g_up"], atol=1e-2)
    ok_down = _cmp("g_down", naive.w_down.grad, grads_ft["g_down"], atol=1e-2)

    if all([ok_fwd, ok_dx, ok_router, ok_up, ok_down]):
        print("\n✓ MoE block matches naive PyTorch within bf16 noise")
    else:
        raise AssertionError("MoE block parity failed — see mismatches above")


if __name__ == "__main__":
    main()
