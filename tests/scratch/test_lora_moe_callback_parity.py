"""End-to-end parity test for the MoE LoRA per-expert callback's
fast path (matmul_fast via dispatcher.matmul_fast) vs slow path
(eager PyTorch @ + .add_()).

Builds the actual ``LoRAWrapperLayer._make_moe_callback`` closure used
in production, exercises both paths over realistic Qwen3.5 MoE shapes,
and asserts dA/dB match within bf16 tolerance.

Run from the repo root with the env's libcudart on LD_LIBRARY_PATH:

  LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \\
  PYTHONPATH=. python tests/scratch/test_lora_moe_callback_parity.py
"""
import sys

import torch
from matmul_dispatcher import CublasLtDispatcher

from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer, LoRATargetConfig


def _make_callback(targets, weights, grads):
    """Build the closure the wrapper would build, without the attach
    plumbing — only ``self.targets`` is referenced inside."""
    class _StubWrapper:
        pass
    stub = _StubWrapper()
    stub.targets = targets
    moe_set = frozenset(t.target_name for t in targets)
    return LoRAWrapperLayer._make_moe_callback(stub, moe_set, weights, grads)


def main():
    torch.manual_seed(0)
    torch.cuda.set_device(0)

    # Qwen3.5-A3B-style: d=2048, F=512, top_k unused at this layer.
    # Use E=8 (small) for test speed.
    d_model, F, E, r = 2048, 512, 8, 16
    alpha = float(r)

    weights = {
        "w_up_lora_a":   torch.randn(E, d_model, r, device="cuda", dtype=torch.bfloat16),
        "w_up_lora_b":   torch.randn(E, r, 2 * F, device="cuda", dtype=torch.bfloat16),
        "w_down_lora_a": torch.randn(E, F, r, device="cuda", dtype=torch.bfloat16),
        "w_down_lora_b": torch.randn(E, r, d_model, device="cuda", dtype=torch.bfloat16),
    }

    def fresh_grads():
        return {
            "g_up_lora_a":   torch.zeros(E, d_model, r, device="cuda", dtype=torch.bfloat16),
            "g_up_lora_b":   torch.zeros(E, r, 2 * F, device="cuda", dtype=torch.bfloat16),
            "g_down_lora_a": torch.zeros(E, F, r, device="cuda", dtype=torch.bfloat16),
            "g_down_lora_b": torch.zeros(E, r, d_model, device="cuda", dtype=torch.bfloat16),
        }

    targets = [
        LoRATargetConfig(target_name="w_up",   rank=r, alpha=alpha),
        LoRATargetConfig(target_name="w_down", rank=r, alpha=alpha),
    ]

    disp = CublasLtDispatcher(round_multiple=32)
    sp = torch.cuda.current_stream().cuda_stream

    grads_slow = fresh_grads()
    grads_fast = fresh_grads()
    cb_slow = _make_callback(targets, weights, grads_slow)
    cb_fast = _make_callback(targets, weights, grads_fast)

    # Fast-path scratch (per-stream, but we only use primary here).
    max_T_e = 1024
    dY_B_buf = torch.empty(max_T_e, r, device="cuda", dtype=torch.bfloat16)
    X_A_buf  = torch.empty(max_T_e, r, device="cuda", dtype=torch.bfloat16)

    # Drive both paths with identical (X, dY) per expert.
    for eid in range(E):
        T_e = 100 + eid * 50
        # g_up:   X=(T_e, d), dY=(T_e, 2F)
        X_up  = torch.randn(T_e, d_model, device="cuda", dtype=torch.bfloat16)
        dY_up = torch.randn(T_e, 2 * F,   device="cuda", dtype=torch.bfloat16)
        # g_down: X=(T_e, F), dY=(T_e, d)
        X_dn  = torch.randn(T_e, F,       device="cuda", dtype=torch.bfloat16)
        dY_dn = torch.randn(T_e, d_model, device="cuda", dtype=torch.bfloat16)

        # Slow (no dispatcher kwargs)
        cb_slow("g_up",   eid, X_up, dY_up)
        cb_slow("g_down", eid, X_dn, dY_dn)
        # Fast (dispatcher + scratch)
        cb_fast("g_up",   eid, X_up, dY_up, disp, sp, dY_B_buf, X_A_buf)
        cb_fast("g_down", eid, X_dn, dY_dn, disp, sp, dY_B_buf, X_A_buf)

    torch.cuda.synchronize()

    fail = False
    for k in grads_slow:
        s = grads_slow[k].float()
        f = grads_fast[k].float()
        diff = (s - f).abs()
        denom = s.abs().clamp(min=1e-3)
        max_rel = (diff / denom).max().item()
        max_abs = diff.max().item()
        ok = max_rel < 5e-2  # bf16 GEMM accumulation tolerance
        status = "OK " if ok else "FAIL"
        print(f"  {status}  {k:20s}  max_abs={max_abs:.4e}  max_rel={max_rel:.4e}")
        if not ok:
            fail = True

    if fail:
        sys.exit(1)
    print("\nAll grads match: fast path == slow path (within bf16 tolerance)")


if __name__ == "__main__":
    main()
