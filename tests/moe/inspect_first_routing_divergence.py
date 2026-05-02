"""Find first (layer, token) where chosen_experts diverges between two
runs, then dump all 256 router logits side-by-side for that token so
you can see whether the disagreement is at a "tied/near-tied" boundary
or somewhere genuinely unexpected.

Walks layers in order, finds the earliest layer where any token's
chosen_experts (a sorted-set comparison) differs between flex and
sonic. For that layer, finds the first divergent token, then prints:

  - The two runs' chosen-expert sets (and the symmetric difference)
  - All 256 (logit_flex, logit_sonic) pairs at that token, sorted by
    flex-logit descending
  - The router-logit gap between the two runs at the divergent
    expert positions
  - Highlights the experts at the topk boundary

If the flex-only experts have flex-logits that rank just below the
sonic-only experts in flex's run (and vice versa), that's a
boundary-tiebreak divergence — expected from bf16 reduction order.
If the flex-only experts have wildly different ranks, it's a real
routing disagreement upstream.

Run:
  python tests/moe/inspect_first_routing_divergence.py \\
    --dir-a /path/dump_a --dir-b /path/dump_b \\
    --top-k 8 [--max-layer N]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _layer_id(p: Path) -> int:
    return int(p.stem.split("_")[1].replace("layer", ""))


def _find_first_diff_layer(dir_a: Path, dir_b: Path) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Return (layer_id, ce_a, ce_b) for the earliest layer where
    any token's chosen-expert SET differs."""
    for f in sorted(dir_a.glob("fwd_layer*_chosen_experts.pt")):
        L = _layer_id(f)
        ce_a = torch.load(f, map_location="cpu", weights_only=True).long()
        fb = dir_b / f.name
        if not fb.exists():
            continue
        ce_b = torch.load(fb, map_location="cpu", weights_only=True).long()
        # Compare as SETS per row (top-k order doesn't matter; routing
        # is set-equivalent).
        sa = ce_a.sort(dim=-1).values
        sb = ce_b.sort(dim=-1).values
        if not torch.equal(sa, sb):
            return L, ce_a, ce_b
    return -1, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True)
    ap.add_argument("--dir-b", required=True)
    ap.add_argument("--top-k", type=int, default=8)
    args = ap.parse_args()

    da, db = Path(args.dir_a), Path(args.dir_b)
    L, ce_a, ce_b = _find_first_diff_layer(da, db)
    if L < 0:
        print("no chosen_experts divergence found across all layers (sets agree everywhere)")
        return 0

    # Find first divergent token.
    sa = ce_a.sort(dim=-1).values
    sb = ce_b.sort(dim=-1).values
    diff_rows = (sa != sb).any(dim=-1)
    first_token = int(torch.nonzero(diff_rows)[0].item())

    expers_a = set(ce_a[first_token].tolist())
    expers_b = set(ce_b[first_token].tolist())
    only_a = sorted(expers_a - expers_b)
    only_b = sorted(expers_b - expers_a)
    shared = sorted(expers_a & expers_b)

    # Load both runs' x_router for this layer.
    xr_a_path = da / f"fwd_layer{L:03d}_x_router.pt"
    xr_b_path = db / f"fwd_layer{L:03d}_x_router.pt"
    xr_a = torch.load(xr_a_path, map_location="cpu", weights_only=True)[first_token].float()
    xr_b = torch.load(xr_b_path, map_location="cpu", weights_only=True)[first_token].float()

    print(f"=== first chosen_experts divergence ===")
    print(f"layer {L}, token {first_token}")
    print()
    print(f"  shared {len(shared)}: {shared}")
    print(f"  only A ({da.name}): {only_a}  (with logit_a={[xr_a[e].item() for e in only_a]})")
    print(f"  only B ({db.name}): {only_b}  (with logit_b={[xr_b[e].item() for e in only_b]})")
    print()
    print(f"=== all {xr_a.numel()} router logits at this token (sorted by logit_A desc) ===")
    print(f"  {'rank':>5}  {'expert':>6}  {'logit_A':>14}  {'logit_B':>14}  {'A-B':>14}  {'in_A':>5}  {'in_B':>5}  flag")

    sorted_idx = xr_a.argsort(descending=True)
    rank_a_of = {int(sorted_idx[r]): r for r in range(xr_a.numel())}
    sorted_idx_b = xr_b.argsort(descending=True)
    rank_b_of = {int(sorted_idx_b[r]): r for r in range(xr_b.numel())}

    K = args.top_k
    boundary_show = 4  # show K-4..K+4 around the topk boundary
    for r, e in enumerate(sorted_idx.tolist()):
        in_a = e in expers_a
        in_b = e in expers_b
        # Show only: top-(K + boundary_show) of A, plus any expert in
        # only_b that isn't already shown
        is_top = r < K + boundary_show
        is_disagreement = (in_a != in_b)
        if not (is_top or is_disagreement):
            continue
        a, b = xr_a[e].item(), xr_b[e].item()
        diff = a - b
        flag = ""
        if r == K - 1:
            flag = "<-- A's last picked"
        elif r == K:
            flag = "<-- A's first not-picked"
        if is_disagreement:
            flag = (flag + " DISAGREE").strip()
        print(
            f"  {r:>5}  {e:>6}  {a:>14.6f}  {b:>14.6f}  {diff:>+14.6f}  "
            f"{'A' if in_a else '.':>5}  {'B' if in_b else '.':>5}  {flag}"
        )

    # Also print B's perspective: rank in B of each disagreement
    print()
    print("=== ranks of disagreements in B's ordering ===")
    for e in sorted(set(only_a + only_b)):
        ra = rank_a_of[e]
        rb = rank_b_of[e]
        in_a = "A" if e in expers_a else "-"
        in_b = "B" if e in expers_b else "-"
        print(f"  expert {e:>3}: rank in A={ra:>4} ({in_a}), rank in B={rb:>4} ({in_b}), "
              f"logit A={xr_a[e].item():+.6f}, B={xr_b[e].item():+.6f}, |A-B|={abs(xr_a[e].item()-xr_b[e].item()):.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
