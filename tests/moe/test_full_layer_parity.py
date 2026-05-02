"""Full-layer parity test: routed_swiglu_moe_fwd + bwd via the wrapper
that includes residual handling.

The block-level test (``test_moe_backend_parity.py``) compares only
the MoE expert-compute block in isolation — it does NOT exercise the
residual-add path. flextrain's backend has
``supports_residual_in_gather=True`` (residual added inline in the
gather kernel, fp32-accumulated reduction), while sonic and scattermoe
have it ``False`` (residual added via ``out.add_(residual)`` after
the gather, bf16+bf16 add).

When |residual| >> |moe_contrib|, the post-add bf16 path can lose
precision in the small-magnitude expert contributions due to
catastrophic cancellation. This test exercises that path with
realistic-magnitude residuals to check whether the precision loss is
a real correctness concern or just bf16 noise.

Reference: hand-rolled per-expert PyTorch loop + explicit residual
add in fp32, downcast at the end.

Run on a Hopper machine (sonic requires sm_90+):
  PYTHONPATH=. python tests/moe/test_full_layer_parity.py \\
    -T 2048 -K 8 -E 256 -F 1024 --d-model 2048 \\
    --residual-scale 10 --cos-tol 0.99
"""
from __future__ import annotations

import argparse
import sys
import types

import torch
import torch.nn.functional as F_torch

sys.path.insert(0, "/home/shein/Documents/flextrain")

from flextrain.ops.moe_backend import (
    FlextrainMoEExpertCompute,
    ScatterMoEExpertCompute,
)
from flextrain.ops.full_moe import routed_swiglu_moe_fwd


def _make_fake_slot(T, K, E, F, d_model, dtype, device):
    TK = T * K
    return types.SimpleNamespace(
        x_router=torch.empty(T, E, device=device, dtype=dtype),
        router_weights=torch.empty(T, K, device=device, dtype=dtype),
        chosen_experts=torch.empty(T, K, device=device, dtype=torch.int32),
        x_up=torch.empty(TK, 2 * F, device=device, dtype=dtype),
        # Flextrain backend's private fields
        expert_counts=torch.empty(E, device=device, dtype=torch.int32),
        index_mapping=torch.empty(T, K, device=device, dtype=torch.int32),
        scattered_router_weights=torch.empty(TK, 1, device=device, dtype=dtype),
        # ScatterMoE backend's private fields
        scattermoe_sorted_expert_idxs=torch.empty(TK, device=device, dtype=torch.int32),
        scattermoe_sorted_scattered_idxs=torch.empty(TK, device=device, dtype=torch.int32),
        scattermoe_expert_offsets=torch.empty(E, device=device, dtype=torch.int32),
        # SonicMoE backend's private fields
        sonic_s_scatter_idx=torch.empty(TK, device=device, dtype=torch.int32),
        sonic_s_reverse_scatter_idx=torch.empty(TK, device=device, dtype=torch.int32),
        sonic_x_gather_idx=torch.empty(TK, device=device, dtype=torch.int32),
        sonic_expert_frequency=torch.empty(E, device=device, dtype=torch.int32),
        sonic_expert_frequency_offset=torch.empty(E + 1, device=device, dtype=torch.int32),
        aux={},
    )


def naive_full_layer_fwd(x, w_router, w_up, w_down, residual, top_k):
    """Full-layer reference: route → scatter → per-expert MLP → gather
    + residual-add, all in fp32 then downcast at the end."""
    T, d = x.shape
    E = w_router.shape[1]
    F = w_down.shape[1]

    # Router (fp32). w_router stored (d, E); x @ w_router → (T, E).
    router_logits = x.float() @ w_router.float()  # (T, E)
    probs = F_torch.softmax(router_logits, dim=-1)
    expert_p, expert_idxs = probs.topk(top_k, dim=-1)  # (T, K)
    expert_p = expert_p / expert_p.sum(dim=-1, keepdim=True)

    # Per-expert accumulation (fp32) into out_unscaled
    out_unscaled = torch.zeros(T, d, dtype=torch.float32, device=x.device)
    for e in range(E):
        mask = expert_idxs == e
        if not mask.any():
            continue
        token_idx, k_idx = mask.nonzero(as_tuple=True)
        x_e = x[token_idx].float()
        p_e = expert_p[token_idx, k_idx]
        pre_e = x_e @ w_up[e].float()
        value_e, gate_e = pre_e.chunk(2, dim=-1)
        h_e = F_torch.silu(gate_e) * value_e
        y_e = h_e @ w_down[e].float()
        contrib = p_e.unsqueeze(-1) * y_e
        out_unscaled.index_add_(0, token_idx, contrib)

    # Add residual in fp32, then downcast.
    out_fp32 = out_unscaled + residual.float()
    return out_fp32.to(x.dtype)


def diffstats(a, b):
    af = a.float().flatten()
    bf = b.float().flatten()
    diff = af - bf
    cos = torch.nn.functional.cosine_similarity(
        af.unsqueeze(0), bf.unsqueeze(0), dim=-1).item()
    return {
        "cos": cos,
        "mean_diff": diff.mean().item(),
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "ref_scale": bf.abs().mean().item(),
    }


def report(label, a, b, cos_tol):
    s = diffstats(a, b)
    ok = s["cos"] >= cos_tol
    print(
        f"  {'OK ' if ok else 'FAIL'}  {label:30s} "
        f"cos={s['cos']:.6f}  mean_diff={s['mean_diff']:+.3e}  "
        f"mean_abs={s['mean_abs']:.3e}  max_abs={s['max_abs']:.3e}  "
        f"ref_scale={s['ref_scale']:.3e}"
    )
    return ok


def run_backend_fwd(backend, x, w_router, w_up, w_down, residual,
                    top_k, num_experts, dtype, device):
    """Run a single fwd through routed_swiglu_moe_fwd. Returns out tensor."""
    T, d = x.shape
    F = w_down.shape[1]
    slot = _make_fake_slot(T, top_k, num_experts, F, d, dtype, device)
    weights = {"w_router": w_router, "w_up": w_up, "w_down": w_down}
    out = torch.empty(T, d, device=device, dtype=dtype)
    primary_stream = torch.cuda.current_stream()
    secondary_stream = None

    def scratch_fn(shape, dt):
        return torch.empty(shape, dtype=dt, device=device)

    chunk_extra = {}
    routed_swiglu_moe_fwd(
        ffn_norm_output=x,
        weights=weights,
        out_tensor=out,
        residual=residual,
        slot=slot,
        chunk_extra=chunk_extra,
        layer_id=0,
        top_k=top_k,
        num_experts=num_experts,
        routing_mode="softmax_then_topk",
        primary_stream=primary_stream,
        secondary_stream=secondary_stream,
        scratch_fn=scratch_fn,
        expert_compute=backend,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=int, default=2048)
    ap.add_argument("-K", type=int, default=8)
    ap.add_argument("-E", type=int, default=256)
    ap.add_argument("-F", type=int, default=1024)
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--residual-scale", type=float, default=1.0,
                    help="multiplier on residual norm (1.0 = same as MoE output, "
                         "10.0 = realistic for a deep-layer residual stream)")
    ap.add_argument("--cos-tol", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    T, K, E, F, d = args.T, args.K, args.E, args.F, args.d_model
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    print(
        f"=== test_full_layer_parity: T={T} K={K} E={E} F={F} d={d} "
        f"residual_scale={args.residual_scale} cos_tol={args.cos_tol} "
        f"seed={args.seed} ==="
    )

    x = torch.randn(T, d, device=device, dtype=dtype)
    # w_router: (d, E) so that x @ w_router → (T, E) router logits.
    w_router = torch.randn(d, E, device=device, dtype=dtype) / (d ** 0.5)
    w_up = torch.randn(E, d, 2 * F, device=device, dtype=dtype) / (d ** 0.5)
    w_down = torch.randn(E, F, d, device=device, dtype=dtype) / (F ** 0.5)
    residual = torch.randn(T, d, device=device, dtype=dtype) * args.residual_scale

    # Reference
    print("\n=== Naive PyTorch reference (fp32 residual add) ===")
    out_ref = naive_full_layer_fwd(x, w_router, w_up, w_down, residual, K)
    print(f"  out_ref:  abs_mean={out_ref.abs().mean():.3e}  max={out_ref.abs().max():.3e}")
    print(f"  residual: abs_mean={residual.abs().mean():.3e}  max={residual.abs().max():.3e}")
    print(f"  ratio (residual/moe_contrib estimate): "
          f"{(residual.abs().mean() / max((out_ref - residual).abs().mean(), 1e-30)):.2f}")

    # Backends
    fail_any = False
    for name, backend in [
        ("FlextrainMoEExpertCompute", FlextrainMoEExpertCompute()),
        ("ScatterMoEExpertCompute", ScatterMoEExpertCompute()),
    ]:
        print(f"\n=== {name} ===")
        out = run_backend_fwd(backend, x, w_router, w_up, w_down, residual,
                              K, E, dtype, device)
        ok = report("out (T, d)", out, out_ref, args.cos_tol)
        if not ok:
            fail_any = True

    # Sonic — skip on non-Hopper
    cap = torch.cuda.get_device_capability()
    if cap >= (9, 0):
        try:
            from flextrain.ops.moe_backend import SonicMoEExpertCompute
            sonic = SonicMoEExpertCompute()
            print(f"\n=== SonicMoEExpertCompute ===")
            out = run_backend_fwd(sonic, x, w_router, w_up, w_down, residual,
                                  K, E, dtype, device)
            ok = report("out (T, d)", out, out_ref, args.cos_tol)
            if not ok:
                fail_any = True
        except (ImportError, RuntimeError) as e:
            print(f"\n=== SonicMoE: SKIP ({e}) ===")
    else:
        print(f"\n=== SonicMoE: SKIP (sm_{cap[0]}{cap[1]}, requires sm_90+) ===")

    print()
    if fail_any:
        print("FAIL — at least one backend diverges from fp32 reference.")
        return 1
    print("PASS — all backends agree with fp32 reference at the full-layer level.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
