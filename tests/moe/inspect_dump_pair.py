"""Quick visual inspector: load two paired dump tensors, print full
diff stats (cos, sign-agreement, mean, std, rel) at multiple
magnitude floors.

Pretrained-model gradients have heavy-tailed magnitude distributions:
most positions are near-zero with a few large spikes. Full-tensor
cosine is dominated by the near-zero positions where bf16 round-off
flips signs at random. Restricting to positions with non-trivial
magnitude (e.g. >= 1% of max) gives a much cleaner signal of whether
the two runs actually agree on the meaningful gradient.
"""
import sys
import torch


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = a - b
    abs_diff = diff.abs()
    cos = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()
    # Sign agreement: same sign at each position. Treat 0 as a separate
    # bucket; report fraction where sign matches among nonzero-on-both.
    a_sign = a.sign()
    b_sign = b.sign()
    both_nonzero = (a_sign != 0) & (b_sign != 0)
    if both_nonzero.any():
        sign_agree = ((a_sign == b_sign) & both_nonzero).float().sum() / both_nonzero.float().sum()
        sign_agree = sign_agree.item()
    else:
        sign_agree = float("nan")
    # Relative error: mean |a - b| / mean |b|. Robust to outliers.
    ref_scale = b.abs().mean().item()
    rel = abs_diff.mean().item() / max(ref_scale, 1e-30)
    return {
        "n": len(a),
        "cos": cos,
        "sign_agree": sign_agree,
        "mean_diff": diff.mean().item(),
        "std_diff": diff.std().item() if len(a) > 1 else 0.0,
        "max_abs": abs_diff.max().item(),
        "ref_scale": ref_scale,
        "rel_err": rel,
    }


def _print_row(label: str, s: dict) -> None:
    sign = f"{s['sign_agree']*100:.2f}%" if s['sign_agree'] == s['sign_agree'] else "  n/a "
    print(
        f"  {label:<20s} n={s['n']:>10d}  cos={s['cos']:>8.5f}  "
        f"sign_agree={sign:>8s}  "
        f"mean_diff={s['mean_diff']:>+11.3e}  std_diff={s['std_diff']:>10.3e}  "
        f"max_abs={s['max_abs']:>10.3e}  ref={s['ref_scale']:>10.3e}  "
        f"rel={s['rel_err']:>8.3e}"
    )


def main():
    if len(sys.argv) != 3:
        print("usage: inspect_dump_pair.py <tensor_a.pt> <tensor_b.pt>")
        return 1
    a = torch.load(sys.argv[1], map_location="cpu", weights_only=True).float()
    b = torch.load(sys.argv[2], map_location="cpu", weights_only=True).float()
    af = a.flatten()
    bf = b.flatten()

    print(f"shape: {tuple(a.shape)}")
    print(f"flex:  abs_mean={a.abs().mean():.3e}  max={a.abs().max():.3e}  frac_zero={(a==0).float().mean():.3f}")
    print(f"sonic: abs_mean={b.abs().mean():.3e}  max={b.abs().max():.3e}  frac_zero={(b==0).float().mean():.3f}")

    # Magnitude bucketing on the union magnitude
    mag = torch.maximum(af.abs(), bf.abs())
    max_mag = mag.max().item()
    print()
    print("position-magnitude distribution (max(|flex|, |sonic|)):")
    for thresh in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        count = int((mag >= thresh).sum())
        pct = 100 * count / len(mag)
        print(f"  >= {thresh:.0e}:  {count:>10d}  ({pct:5.2f}%)")

    print()
    print("--- diff stats at increasing magnitude floors ---")
    # Floor relative to max magnitude
    for label, frac in [("full", 0.0), ("top 10%", 0.1), ("top 1%", 0.01), ("top 0.1%", 0.001)]:
        if frac == 0.0:
            mask = torch.ones_like(mag, dtype=torch.bool)
        else:
            thresh = frac * max_mag
            mask = mag >= thresh
        if mask.sum() < 2:
            continue
        s = _stats(af[mask], bf[mask])
        _print_row(label, s)

    # Top-K positions
    topk_n = min(50, len(mag))
    topk_idx = mag.topk(topk_n).indices
    print()
    print(f"top-{topk_n} positions by max(|flex|, |sonic|):")
    s = _stats(af[topk_idx], bf[topk_idx])
    _print_row(f"top-{topk_n}", s)
    print(f"  first 5 paired:")
    for i in topk_idx[:5].tolist():
        print(f"    pos {i:>10d}: flex={af[i]:+.4e}  sonic={bf[i]:+.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
