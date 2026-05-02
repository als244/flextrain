"""Test that ``g_up`` / ``g_down`` accumulate INTO existing grad buffers
across all 3 MoE backends.

The ``test_moe_backend_parity.py`` block parity test starts with
zero-initialized ``g_up`` / ``g_down``, which means it cannot
distinguish between "backend writes (overwrites) into the grad buffer"
and "backend adds (accumulates) into the grad buffer" — both produce
the same final value when starting from zero.

Under chunked bwd, the engine accumulates per-chunk gradient
contributions into the same persistent ``grads`` dict across the
chunk loop. A backend that overwrites instead of accumulating would
silently lose all but the last chunk's contributions.

This test starts with NON-ZERO random ``g_up`` / ``g_down`` and
verifies the post-bwd result equals ``initial + computed_dW`` (within
bf16 noise) — distinguishing accumulate from overwrite.

Scope: this test exercises ``expert_compute.bwd`` directly and so
covers ONLY ``g_up`` / ``g_down`` (the backend-owned routed-expert
wgrads). The other wgrads in the MoE backward stack are accumulated
by flextrain-owned code and don't need a separate accumulation test:

* ``g_router``: accumulated by ``routed_swiglu_moe_bwd`` via
  ``dispatcher.matmul(C=g_router, D=g_router, beta=1.0)`` — the
  cuBLASLt epilogue does the accumulate. Same code regardless of
  backend.
* ``g_shared_*``: accumulated by ``MoESwiGLUSharedExpertFFN.bwd``
  via ``Tensor.addmm_`` or ``Tensor.add_(einsum(...))`` — both are
  PyTorch's accumulating-by-construction ops.

The accumulation risk is concentrated in ``g_up`` / ``g_down`` because
scattermoe and sonic write GEMM output via the kernel's ``out=`` param
(which OVERWRITES) and add into the grad in a separate pass. A
refactor that accidentally points the GEMM ``out=`` directly at
``grads["g_up"]`` or ``grads["g_down"]`` would silently destroy
prior chunks' contributions — that's exactly what this test catches.

Run from repo root with the CUDA runtime libs on LD_LIBRARY_PATH:
  PYTHONPATH=. python tests/moe/test_grad_accumulation.py
"""
from __future__ import annotations

import argparse
import sys
import types

import torch

from flextrain.ops.moe_backend import (
    FlextrainMoEExpertCompute,
    ScatterMoEExpertCompute,
)
from tests.moe.test_moe_backend_parity import (
    naive_moe_fwd_bwd,
    _make_fake_slot,
    _diffstats,
)


def _run_backend_with_initial_grads(
    backend, x, expert_p, expert_idxs, weights, dy,
    g_up_init, g_down_init,
    T, K, E, F, d_model, dtype, device,
):
    """Run the backend's fwd+bwd starting from non-zero (g_up_init,
    g_down_init). Returns the final g_up, g_down."""
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

    # Start grads from non-zero initial state. Clone so the caller's
    # tensors aren't mutated (we want to compare against them later).
    grads = {
        "g_up": g_up_init.clone(),
        "g_down": g_down_init.clone(),
    }
    backend.bwd(
        dy, x, weights, grads,
        slot=slot, chunk_extra=chunk_extra, layer_id=0,
        primary_stream=primary_stream, secondary_stream=None,
        scratch_fn=scratch_fn,
    )
    torch.cuda.synchronize()
    return grads["g_up"], grads["g_down"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-T", type=int, default=2048, help="num tokens")
    ap.add_argument("-K", type=int, default=8, help="top-k")
    ap.add_argument("-E", type=int, default=64, help="num experts")
    ap.add_argument("-F", type=int, default=512, help="expert intermediate dim")
    ap.add_argument("--d-model", type=int, default=1024)
    ap.add_argument("--cos-tol", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    T, K, E, F, d_model = args.T, args.K, args.E, args.F, args.d_model
    torch.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    print(f"=== test_grad_accumulation: T={T} K={K} E={E} F={F} d={d_model} ===")
    print(f"    Verifying g_up / g_down ACCUMULATE (not overwrite)\n")

    # Random inputs.
    x = torch.randn(T, d_model, device=device, dtype=dtype)
    logits = torch.randn(T, E, device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    expert_p, expert_idxs = probs.topk(K, dim=-1)
    expert_p = (expert_p / expert_p.sum(dim=-1, keepdim=True)).to(dtype)
    expert_idxs = expert_idxs.to(torch.int32)

    weights = {
        "w_up": torch.randn(E, 2 * F, d_model, device=device, dtype=dtype) / (d_model ** 0.5),
        "w_down": torch.randn(E, d_model, F, device=device, dtype=dtype) / (F ** 0.5),
    }
    dy = torch.randn(T, d_model, device=device, dtype=dtype)

    # NON-ZERO initial grads. Use a known random tensor with realistic
    # magnitude (close to what the actual gradient will be) so the
    # accumulated result is sensitive to both terms.
    g_up_init = torch.randn_like(weights["w_up"]) * 0.1
    g_down_init = torch.randn_like(weights["w_down"]) * 0.1

    # Reference: compute dW via autograd, then expected_g = initial + dW.
    print("=== Computing autograd reference dW ===")
    x_ref = x.detach().clone().requires_grad_(True)
    expert_p_ref = expert_p.detach().clone().requires_grad_(True)
    w_up_ref = weights["w_up"].detach().clone().requires_grad_(True)
    w_down_ref = weights["w_down"].detach().clone().requires_grad_(True)
    _, _, _, dw_up_ref, dw_down_ref = naive_moe_fwd_bwd(
        x_ref, expert_p_ref, expert_idxs, w_up_ref, w_down_ref, dy,
    )

    # Expected: initial + computed_dW.
    expected_g_up = (g_up_init.float() + dw_up_ref.float()).to(dtype)
    expected_g_down = (g_down_init.float() + dw_down_ref.float()).to(dtype)

    backends_to_test = [
        ("FlextrainMoEExpertCompute", FlextrainMoEExpertCompute()),
        ("ScatterMoEExpertCompute", ScatterMoEExpertCompute()),
    ]
    cap = torch.cuda.get_device_capability()
    if cap >= (9, 0):
        try:
            from flextrain.ops.moe_backend import SonicMoEExpertCompute
            backends_to_test.append(
                ("SonicMoEExpertCompute", SonicMoEExpertCompute())
            )
        except (ImportError, RuntimeError) as e:
            print(f"\n=== SonicMoE: SKIP ({e}) ===")
    else:
        print(f"\n=== SonicMoE: SKIP (sm_{cap[0]}{cap[1]}, requires sm_90+) ===")

    fail_any = False
    for name, backend in backends_to_test:
        print(f"\n=== {name} ===")
        try:
            got_g_up, got_g_down = _run_backend_with_initial_grads(
                backend, x, expert_p, expert_idxs, weights, dy,
                g_up_init, g_down_init,
                T, K, E, F, d_model, dtype, device,
            )
        except Exception as e:
            print(f"  FAIL — backend raised: {e}")
            fail_any = True
            continue

        # ALSO compute "what overwrite would look like" to make the
        # diagnostic informative when a test fails: if got_g matches
        # dw_ref but NOT (initial + dw_ref), the backend overwrote.
        for label, got, expected, ref_dw, init in [
            ("g_up (E, 2F, d)",   got_g_up,   expected_g_up,   dw_up_ref,   g_up_init),
            ("g_down (E, d, F)",  got_g_down, expected_g_down, dw_down_ref, g_down_init),
        ]:
            stats = _diffstats(got, expected)
            ok = stats["cos"] >= args.cos_tol
            status = "OK " if ok else "FAIL"
            print(
                f"  {status}  {label:30s} cos={stats['cos']:.6f}  "
                f"max_abs={stats['max_abs']:.3e}  ref_scale={stats['ref_scale']:.3e}"
            )
            if not ok:
                fail_any = True
                # Diagnostic: did the backend overwrite?
                overwrite_stats = _diffstats(got, ref_dw.to(dtype))
                # Initial-preserve check: did the backend leave the grad alone?
                noop_stats = _diffstats(got, init)
                print(
                    f"        → vs dW_ref alone (would mean OVERWRITE): "
                    f"cos={overwrite_stats['cos']:.6f}"
                )
                print(
                    f"        → vs initial alone (would mean BACKEND-NO-OP): "
                    f"cos={noop_stats['cos']:.6f}"
                )

    print()
    if fail_any:
        print("FAIL — at least one backend doesn't accumulate correctly.")
        sys.exit(1)
    print("PASS — all backends correctly accumulate g_up / g_down "
          "into existing buffers.")


if __name__ == "__main__":
    main()
