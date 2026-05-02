"""Quantify how close-to-tied router logits are around the top-k boundary.

If top-k=8 and the 8th-vs-9th expert logits are within bf16-epsilon of
each other, kernel reduction-order differences (between flextrain and
sonic backends) can flip the ranking, producing different chosen_experts
even though both are "correct" picks. This script shows how often that
boundary is "tight" enough to be sensitive to noise.

Per-layer columns:
  logit_std  - std of router logits across (T, E)
  logit_kurt - kurtosis (heaviness of tail; >0 = heavy)
  exact_ties - number of (token, expert_a, expert_b) triples with
               logit_a == logit_b exactly. Almost always 0 in fp.
  rank8_gap_min/median/max - gap = logit[rank=8] - logit[rank=9]
                              over all tokens. Smaller = easier to flip.
  near_tie_pct - fraction of tokens where rank8_gap < 1e-3
                 (a coarse "noise can flip me" threshold)

Run:
  python tests/moe/inspect_routing_ties.py \\
    --dir /path/to/dump --top-k 8 --num-experts 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _kurtosis(x: torch.Tensor) -> float:
    x = x.float() - x.float().mean()
    s = x.std(unbiased=False)
    if s.item() == 0:
        return 0.0
    return ((x ** 4).mean() / s ** 4 - 3).item()  # excess kurtosis


def _layer_id(p: Path) -> int:
    return int(p.stem.split("_")[1].replace("layer", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=256)
    ap.add_argument("--near-tie-thresh", type=float, default=1e-3,
                    help="rank8 - rank9 gap below this is 'near-tie'")
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob("fwd_layer*_x_router.pt"))
    if not files:
        print(f"no fwd_layer*_x_router.pt in {d}")
        return 1

    print(f"=== {d.name} (top_k={args.top_k}, E={args.num_experts}, "
          f"near-tie<={args.near_tie_thresh}) ===\n")
    print(
        f"{'layer':<6} | {'logit_std':>10s} {'kurt':>7s} {'exact_ties':>11s} | "
        f"{'rank8_gap_min':>14s} {'rank8_gap_med':>14s} {'rank8_gap_p10':>14s} | "
        f"{'near_tie%':>10s}"
    )
    print("-" * 110)

    for f in files:
        L = _layer_id(f)
        x = torch.load(f, map_location="cpu", weights_only=True).float()
        T, E = x.shape

        std = x.std().item()
        kurt = _kurtosis(x)

        # Exact ties: per-token, count pairs with equal logits.
        # That's a per-row count of duplicate values among E logits.
        # Approximation: rows with at least 1 dup — much faster than
        # all pairs.
        sx = x.sort(dim=-1).values
        diffs = sx[:, 1:] - sx[:, :-1]
        rows_with_exact_tie = int(((diffs == 0).any(dim=-1)).sum())

        # Top-k+1 sort to compute rank-k vs rank-(k+1) gap per token.
        topk_plus1 = x.topk(args.top_k + 1, dim=-1).values  # (T, K+1) descending
        gap = topk_plus1[:, args.top_k - 1] - topk_plus1[:, args.top_k]  # rank-k minus rank-(k+1)
        gap_min = gap.min().item()
        gap_med = gap.median().item()
        gap_p10 = gap.quantile(0.10).item()
        near_tie_count = int((gap < args.near_tie_thresh).sum())
        near_tie_pct = near_tie_count / T * 100

        print(
            f"{L:<6} | {std:>10.4f} {kurt:>7.2f} {rows_with_exact_tie:>11d} | "
            f"{gap_min:>14.4e} {gap_med:>14.4e} {gap_p10:>14.4e} | "
            f"{near_tie_pct:>9.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
