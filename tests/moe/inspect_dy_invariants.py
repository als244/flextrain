"""Position-invariant comparison of paired ``dy`` dumps.

The naive cos(flex.flatten(), sonic.flatten()) collapses to ~0 in deep
layers because topk tiebreak reorders expert assignments, which makes
per-token gradient spikes land at different positions in the (T, d)
matrix. The aggregate distribution and per-token magnitudes are
essentially identical, but cosine on the raw flattened tensors is
scrambled by position swaps.

This script computes three metrics that ARE invariant (or nearly so)
to that scrambling:

  1. Histogram comparison: bucket values into log-magnitude bins and
     compare counts. If the value distributions match, both runs
     produce gradients of the same shape — just at different rows.

  2. Per-token L2-norm cosine: reduce (T, d) to (T,) by L2-norming each
     row, then compare. Invariant to within-row reordering of features.
     If two runs route the same tokens to slightly different experts,
     their per-token gradient magnitudes should still match.

  3. Sorted-value cosine: sort all (T*d) values by magnitude, compare
     the sorted sequences. This is fully position-invariant — it asks
     "does the sample distribution match?" Insensitive to *which*
     positions have spikes; only checks *that* spikes of similar
     magnitude exist.

Run on della (across all 40 layers):
  python tests/scratch/inspect_dy_invariants.py \\
    --dir-a /home/as1669/storage/flextrain/moe_dump/flextrain \\
    --dir-b /home/as1669/storage/flextrain/moe_dump/sonicmoe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _per_token_norm(t: torch.Tensor) -> torch.Tensor:
    """Reduce (T, d) → (T,): L2 norm along feature dim."""
    return t.float().pow(2).sum(dim=-1).sqrt()


def _sorted_values(t: torch.Tensor) -> torch.Tensor:
    """Flatten + sort by magnitude (ascending). Position-invariant
    representation of the value distribution."""
    return t.float().flatten().abs().sort().values


def _hist_compare(a: torch.Tensor, b: torch.Tensor, n_bins: int = 50) -> dict:
    """Log-magnitude histogram comparison. Returns total-variation
    distance and per-bin max divergence. Lower = better."""
    af = a.float().flatten().abs()
    bf = b.float().flatten().abs()
    # Both runs share the same magnitude scale; pool min/max.
    lo = max(min(af.min().item(), bf.min().item()), 1e-30)
    hi = max(af.max().item(), bf.max().item())
    if lo >= hi:
        return {"tv": 0.0, "max_div": 0.0}
    edges = torch.logspace(
        torch.log10(torch.tensor(lo)).item(),
        torch.log10(torch.tensor(hi)).item(),
        steps=n_bins + 1,
    )
    h_a = torch.histogram(af, edges).hist
    h_b = torch.histogram(bf, edges).hist
    p_a = h_a / h_a.sum().clamp(min=1.0)
    p_b = h_b / h_b.sum().clamp(min=1.0)
    tv = 0.5 * (p_a - p_b).abs().sum().item()
    max_div = (p_a - p_b).abs().max().item()
    return {"tv": tv, "max_div": max_div}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float().unsqueeze(0),
        b.flatten().float().unsqueeze(0),
        dim=-1,
    ).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    args = ap.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    files_a = sorted(dir_a.glob("bwd_layer*_dy.pt"))

    print(f"=== {dir_a.name} vs {dir_b.name} ===")
    print(
        f"{'layer':<6} | "
        f"{'cos_full':>10s} {'cos_norm':>10s} {'cos_sort':>10s} | "
        f"{'TV':>8s} {'max_div':>8s} | "
        f"{'norm_mean_a':>12s} {'norm_mean_b':>12s} {'rel_norm':>10s}"
    )
    print("-" * 110)

    for fa in files_a:
        layer = int(fa.stem.split("_")[1].replace("layer", ""))
        fb = dir_b / fa.name
        if not fb.exists():
            continue
        a = torch.load(fa, map_location="cpu", weights_only=True)
        b = torch.load(fb, map_location="cpu", weights_only=True)

        # 1. Original full-tensor cosine (the ~0 baseline)
        cos_full = _cos(a, b)

        # 2. Per-token L2 norm cosine
        na = _per_token_norm(a)
        nb = _per_token_norm(b)
        cos_norm = _cos(na, nb)
        norm_mean_a = na.mean().item()
        norm_mean_b = nb.mean().item()
        rel_norm = abs(norm_mean_a - norm_mean_b) / max(norm_mean_b, 1e-30)

        # 3. Sorted-value cosine (fully position-invariant)
        sa = _sorted_values(a)
        sb = _sorted_values(b)
        cos_sort = _cos(sa, sb)

        # 4. Histogram comparison
        h = _hist_compare(a, b)

        print(
            f"{layer:<6} | "
            f"{cos_full:>10.5f} {cos_norm:>10.5f} {cos_sort:>10.5f} | "
            f"{h['tv']:>8.4f} {h['max_div']:>8.4f} | "
            f"{norm_mean_a:>12.3e} {norm_mean_b:>12.3e} {rel_norm:>10.3e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
