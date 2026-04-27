"""8B divergence diagnostic: isolate whether the bug is in forward,
backward, or optimizer step.

Pattern we want to reproduce / isolate:
    step 0 loss = 0.83 (matches HF) ✓
    step 1 loss = 0.62 (healthy drop)   ✓
    step 2 loss = 100  (saturated — broken) ✗

Procedure:
    A. Run forward-only 5 times on different batches, no optimizer,
       no backward. Loss should be ~0.8 on each batch (with slight
       variation for the random input).
    B. Run full training 5 steps — reproduce divergence.
    C. Compare weight-norm before/after each step.
    D. Run backward, but do NOT call optimizer.step. Then run forward
       again. Loss should match (A) since weights unchanged.

The goal is to pinpoint which phase is broken.
"""

from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import ModelShape, _Seq, _flextrain_step
from tests.test_llama31_8b_training import (  # noqa: E402
    _build_llama31_8b_shape, _build_flextrain_8b,
)
from tests.test_llama32_1b_parity import (  # noqa: E402
    _permute_qk_for_pair_interleave, _pull_step_batches,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _weight_stats(am) -> dict:
    """Snapshot max/min/L2 norm of all weights (embed + backbone + head)."""
    stats = {}
    # Backbone host params (fp32 or bf16 master).
    for i, host_p in enumerate(am.buffers.host_params):
        for name, t in host_p.items():
            max_abs = float(t.abs().max())
            l2 = float(t.float().norm())
            stats[f"L{i}/{name}"] = (max_abs, l2)
    for name, t in am.buffers.host_head_params.items():
        max_abs = float(t.abs().max())
        l2 = float(t.float().norm())
        stats[f"head/{name}"] = (max_abs, l2)
    for name, t in am.buffers.host_embed_params.items():
        max_abs = float(t.abs().max())
        l2 = float(t.float().norm())
        stats[f"embed/{name}"] = (max_abs, l2)
    return stats


def _log_weight_delta(prev: dict, cur: dict) -> None:
    # Print top-5 biggest param moves by |max_abs change|.
    changes = []
    for k, (mx_cur, l2_cur) in cur.items():
        if k not in prev:
            continue
        mx_prev, l2_prev = prev[k]
        dmax = abs(mx_cur - mx_prev)
        dl2 = abs(l2_cur - l2_prev)
        changes.append((dmax, dl2, k, mx_prev, mx_cur, l2_prev, l2_cur))
    changes.sort(reverse=True)
    print("  top weight changes (max|Δmax|):")
    for dmax, dl2, k, mp, mc, lp, lc in changes[:5]:
        print(f"    {k:30s}  max: {mp:.4f} -> {mc:.4f}  L2: {lp:.2f} -> {lc:.2f}")
    # Flag anything that became NaN/Inf:
    for k, (mx_cur, l2_cur) in cur.items():
        if mx_cur != mx_cur or mx_cur > 1e6:  # nan or huge
            print(f"  !! BLOWUP in {k}: max={mx_cur}, L2={l2_cur}")


def main() -> None:
    hf_path = os.path.join(ROOT, "models", "Llama-3.1-8B")
    shape = _build_llama31_8b_shape()
    n_steps = 4
    batches = _pull_step_batches(
        hf_path, n_steps=n_steps, target_tokens_per_step=2048,
        min_len=128, max_len=512,
    )
    print(f"\n=== Building 8B FlexTrain engine ===")
    am = _build_flextrain_8b(shape, 1e-5)
    print("load HF...")
    t0 = time.time()
    am.load_hf(hf_path, strict=False)
    print(f"  load: {time.time() - t0:.1f}s")
    print("Q/K permute...")
    for i in range(shape.n_layers):
        for name in ("w_q", "w_k"):
            w = am.buffers.host_params[i][name]
            am.buffers.host_params[i][name].copy_(
                _permute_qk_for_pair_interleave(w, shape.head_dim)
            )
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()

    # ===========================================================
    # A) Forward-only, 3 times on SAME batch — loss should be identical.
    # ===========================================================
    print("\n=== A) Forward-only (NO training) on batch 0, 3x ===")
    for i in range(3):
        seqs = [_Seq(s.tokens.clone()) for s in batches[0]]
        for d, s in zip(seqs, batches[0]):
            d.targets = s.targets.clone()
        active = sum(int((s.targets != -100).sum().item()) for s in seqs)
        stats = am.fwd_bwd(seqs, loss_scale_factor=1.0 / active, verbose=False)
        # Zero the grads (no optimizer step), reset zero_grad flag.
        am._zero_grad = True
        print(f"  A.{i}: fwd-only on batch 0 loss = {stats.total_loss / active:.4f}")

    # ===========================================================
    # B) Train 4 steps, logging loss + top weight changes.
    # ===========================================================
    print("\n=== B) Train 4 steps with lr=1e-5 ===")
    prev_stats = _weight_stats(am)
    for step, batch in enumerate(batches):
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        ts = time.time()
        loss = _flextrain_step(am, seqs)
        print(f"\n  step {step}: loss = {loss:.4f}  ({(time.time()-ts):.1f}s)")
        cur_stats = _weight_stats(am)
        _log_weight_delta(prev_stats, cur_stats)
        prev_stats = cur_stats

    # Cleanup.
    am.buffers.destroy()
    del am
    import gc; gc.collect()
    try:
        from flextrain.engine import unregister_all_process_pinned_memory
        unregister_all_process_pinned_memory()
    except Exception:
        pass
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
