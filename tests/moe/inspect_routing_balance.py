"""Print routing-balance + router-weight distribution metrics per layer
from the dump dir. Helps see how skewed expert load is on a real
pretrained MoE.

Metrics per layer:
  Expert-load (from chosen_experts):
    - per-expert token-count: min / median / mean / max
    - cv = std/mean       (0 = balanced; 1 = std == mean)
    - imbalance = max / mean   (1 = ideal; up to E*top_k worst case)
    - entropy_norm = H(p_e) / log(E)  in [0, 1]; 1 = uniform
    - load on top-1 / top-5 experts (% of total slot assignments)
  Router weights (from router_weights):
    - per-topk-slot mean weight (first-pick is usually largest)
    - per-token entropy of normalized topk weights
      (low = sharp 1-hot routing; high = soft routing)

Run on della:
  python tests/scratch/inspect_routing_balance.py \\
    --dir /home/as1669/storage/flextrain/moe_dump/flextrain
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def _load_balance_stats(chosen_experts: torch.Tensor, num_experts: int) -> dict:
    flat = chosen_experts.long().view(-1)
    total = flat.numel()
    counts = torch.bincount(flat, minlength=num_experts).float()
    mean = counts.mean().item()
    std = counts.std(unbiased=False).item()
    cv = std / mean if mean > 0 else float("nan")
    imbalance = counts.max().item() / mean if mean > 0 else float("nan")
    p = counts / total
    nz = p > 0
    entropy = -(p[nz] * p[nz].log()).sum().item()
    entropy_norm = entropy / math.log(num_experts) if num_experts > 1 else 0.0

    sorted_counts = counts.sort(descending=True).values
    top1_pct = sorted_counts[0].item() / total * 100
    top5_pct = sorted_counts[:5].sum().item() / total * 100

    return {
        "min": int(counts.min().item()),
        "median": int(counts.median().item()),
        "mean": mean,
        "max": int(counts.max().item()),
        "cv": cv,
        "imbalance": imbalance,
        "entropy_norm": entropy_norm,
        "top1_pct": top1_pct,
        "top5_pct": top5_pct,
        "frac_unused": (counts == 0).float().mean().item(),
    }


def _router_weight_stats(router_weights: torch.Tensor) -> dict:
    """router_weights: (T, top_k) -- normalized softmax weights summing to 1
    over top_k. Per-token entropy of the topk distribution."""
    rw = router_weights.float()
    T, K = rw.shape
    # Normalize defensively (some pipelines re-normalize, some don't).
    s = rw.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    p = rw / s
    nz = p > 0
    # Per-token entropy: sum over k of -p log p
    log_p = torch.zeros_like(p)
    log_p[nz] = p[nz].log()
    H = -(p * log_p).sum(dim=-1)  # (T,)
    H_norm = H / math.log(K) if K > 1 else H

    # Mean weight by top-k slot (sorted descending). p might already
    # be sorted by topk; sort to be safe.
    p_sorted = p.sort(dim=-1, descending=True).values
    per_slot_mean = p_sorted.mean(dim=0)  # (K,)

    return {
        "per_slot_mean": per_slot_mean.tolist(),
        "entropy_mean": H.mean().item(),
        "entropy_norm_mean": H_norm.mean().item(),
        "entropy_min": H.min().item(),
        "entropy_max": H.max().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dump dir")
    ap.add_argument("--num-experts", type=int, default=256)
    args = ap.parse_args()

    d = Path(args.dir)
    layer_files = sorted(d.glob("fwd_layer*_chosen_experts.pt"))
    if not layer_files:
        print(f"no fwd_layer*_chosen_experts.pt in {d}; run with FLEXTRAIN_MOE_DUMP_DIR set first")
        return 1

    print(f"=== {d} ===\n")
    print(f"E={args.num_experts}\n")
    print(
        f"{'layer':<6} | {'min':>4} {'med':>5} {'mean':>7} {'max':>5} "
        f"{'cv':>5} {'imb':>5} {'entH':>5} {'top1%':>6} {'top5%':>6} {'unused':>7} | "
        f"{'rw_top1':>7} {'rw_top2':>7} {'rw_topk':>7} {'rw_Hnorm':>8}"
    )
    print("-" * 130)

    for f in layer_files:
        layer_id = int(f.stem.split("_")[1].replace("layer", ""))
        ce = torch.load(f, map_location="cpu", weights_only=True)
        rw_path = d / f"fwd_layer{layer_id:03d}_router_weights.pt"
        rw_stats = None
        if rw_path.exists():
            rw = torch.load(rw_path, map_location="cpu", weights_only=True)
            rw_stats = _router_weight_stats(rw)

        s = _load_balance_stats(ce, args.num_experts)
        line = (
            f"{layer_id:<6} | {s['min']:>4} {s['median']:>5} {s['mean']:>7.1f} {s['max']:>5} "
            f"{s['cv']:>5.2f} {s['imbalance']:>5.2f} {s['entropy_norm']:>5.3f} "
            f"{s['top1_pct']:>5.2f}% {s['top5_pct']:>5.2f}% {s['frac_unused']*100:>6.1f}% |"
        )
        if rw_stats is not None:
            psm = rw_stats["per_slot_mean"]
            line += (
                f" {psm[0]:>7.4f} "
                f"{psm[1]:>7.4f} "
                f"{psm[-1]:>7.4f} "
                f"{rw_stats['entropy_norm_mean']:>8.4f}"
            )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
