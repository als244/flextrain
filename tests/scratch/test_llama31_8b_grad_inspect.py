"""Dig into the step-1 NaN bug on Llama-3.1-8B.

Replaces ``_flextrain_step`` with an inline version so we can inspect
per-layer grad stats BEFORE calling ``am.step()``, then again AFTER.
That tells us whether:

  * Grads arrive at step 1's optimizer already NaN (fwd/bwd bug)
  * Grads are finite but AdamW produces NaN weights (optimizer bug)
  * Or something else (stream/event race)
"""

from __future__ import annotations

import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import ModelShape, _Seq
from tests.test_llama31_8b_training import (  # noqa: E402
    _build_llama31_8b_shape, _build_flextrain_8b,
)
from tests.test_llama32_1b_parity import (  # noqa: E402
    _permute_qk_for_pair_interleave, _pull_step_batches,
)


DEVICE = "cuda:0"
DTYPE = torch.bfloat16


def _grad_stats(am) -> dict:
    """Read host grad tensors (mirrored after fwd_bwd) and report
    max|g|, L2 norm, and NaN count per layer+tensor."""
    stats = {}
    # Backbone grads on host.
    for i, host_g in enumerate(am.buffers.host_grads):
        for name, t in host_g.items():
            flat = t.float().flatten()
            nans = int(torch.isnan(flat).sum().item())
            infs = int(torch.isinf(flat).sum().item())
            if nans > 0 or infs > 0:
                stats[f"L{i}/{name}"] = (float('nan'), float('nan'), nans, infs)
            else:
                max_abs = float(flat.abs().max())
                l2 = float(flat.norm())
                stats[f"L{i}/{name}"] = (max_abs, l2, 0, 0)
    return stats


def _weight_stats(am) -> dict:
    stats = {}
    for i, host_p in enumerate(am.buffers.host_params):
        for name, t in host_p.items():
            flat = t.float().flatten()
            nans = int(torch.isnan(flat).sum().item())
            if nans > 0:
                stats[f"L{i}/{name}"] = (float('nan'), float('nan'), nans)
            else:
                stats[f"L{i}/{name}"] = (float(flat.abs().max()), float(flat.norm()), 0)
    return stats


def _print_top(label: str, stats: dict, k: int = 5) -> None:
    # Sort by max_abs or NaN count
    items = []
    for key, v in stats.items():
        max_abs, l2 = v[0], v[1]
        nans = v[2] if len(v) >= 3 else 0
        items.append((nans, max_abs if not (max_abs != max_abs) else 1e20, key, max_abs, l2, nans))
    items.sort(reverse=True)
    print(f"  top {k} {label}:")
    for _, _, k2, mx, l2, nn in items[:k]:
        print(f"    {k2:30s}  max={mx:10.4f}  L2={l2:10.4f}  nans={nn}")


def main() -> None:
    hf_path = os.path.join(ROOT, "models", "Llama-3.1-8B")
    shape = _build_llama31_8b_shape()
    n_steps = 3
    batches = _pull_step_batches(
        hf_path, n_steps=n_steps, target_tokens_per_step=2048,
        min_len=128, max_len=512,
    )
    am = _build_flextrain_8b(shape, 1e-10)  # effectively frozen
    print("load HF...")
    am.load_hf(hf_path, strict=False)
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

    for step_i, batch in enumerate(batches):
        print(f"\n{'='*60}\nStep {step_i}\n{'='*60}")
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        for d, s in zip(seqs, batch):
            d.targets = s.targets.clone()
        active = sum(int((s.targets != -100).sum().item()) for s in seqs)

        print("-> fwd_bwd")
        stats_fwd = am.fwd_bwd(seqs, loss_scale_factor=1.0 / active, verbose=False)
        loss = stats_fwd.total_loss / active
        print(f"  loss = {loss:.4f}")

        # Need to offload GPU grads to host first so we can inspect them.
        # In steady state fwd_bwd leaves grads on GPU ring; host mirror
        # happens during step. Force a mirror for inspection.
        # Simplest: read GPU grad slots directly.
        # Inspect GPU grads BEFORE step(). fwd_bwd leaves grads either
        # on GPU ring slots or mirrored to host (depends on engine flow).
        # Check head grads first (resident on GPU):
        print("GPU head-grad stats (pre-step):")
        for name, t in am.buffers.gpu_head_grads.items():
            flat = t.float().flatten()
            nans = int(torch.isnan(flat).sum().item())
            infs = int(torch.isinf(flat).sum().item())
            if nans + infs == 0:
                print(f"  head/{name}: max={float(flat.abs().max()):.4f}  L2={float(flat.norm()):.2f}")
            else:
                print(f"  head/{name}: NaNs={nans}  Infs={infs}")

        # Embed grad:
        print("GPU embed-grad stats (pre-step):")
        for name, t in am.buffers.gpu_embed_grads.items():
            flat = t.float().flatten()
            nans = int(torch.isnan(flat).sum().item())
            infs = int(torch.isinf(flat).sum().item())
            if nans + infs == 0:
                print(f"  embed/{name}: max={float(flat.abs().max()):.4f}  L2={float(flat.norm()):.2f}")
            else:
                print(f"  embed/{name}: NaNs={nans}  Infs={infs}")

        # Backbone grads: check each of the 7 GPU grad slots. After
        # fwd_bwd finishes, the ring holds the first 7 layers (L0-L6)
        # because backward ran in reverse (last thing fetched was L0-6).
        print("GPU backbone grad-slot stats (pre-step):")
        N_G = am.working_set.n_gpu_grads
        for slot_idx in range(N_G):
            # The ring entry at slot_idx holds whichever layer was last
            # fetched into that slot.
            # After fwd_bwd ends at layer 0, the ring has been rotated
            # so slot 0 holds some layer near the start.
            layer = am.backbone[slot_idx]  # assume slot_idx == layer_id after restore
            grads = am.buffers.gpu_grad_slot(slot_idx, layer.param_spec)
            any_nan = False
            max_abs = 0.0
            for name, t in grads.items():
                flat = t.float().flatten()
                nans = int(torch.isnan(flat).sum().item())
                if nans > 0:
                    any_nan = True
                    max_abs = float('nan')
                else:
                    max_abs = max(max_abs, float(flat.abs().max()))
            marker = "NaN!" if any_nan else f"max={max_abs:.4f}"
            print(f"  slot[{slot_idx}] (layer {layer.layer_id}): {marker}")

        print("-> step")
        t0 = time.time()
        ret = am.step()
        print(f"  step returned {ret}, {time.time()-t0:.1f}s")

        w_stats = _weight_stats(am)
        # Count total NaNs across all params.
        total_nans = sum(v[2] for v in w_stats.values() if len(v) >= 3)
        print(f"  total NaN params in host master weights: {total_nans}")
        # Which layers first went NaN?
        nan_layers = sorted(
            set(k.split("/")[0] for k, v in w_stats.items() if len(v) >= 3 and v[2] > 0)
        )
        print(f"  NaN layers: {nan_layers}")
        _print_top("non-NaN max weights", {k: v for k, v in w_stats.items() if v[2] == 0}, k=3)

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
