"""Compare HF and FT logit bundles produced by hf_capture.py / ft_replay.py.

Streams the per-position diff so peak memory stays low; reports
overall + per-chunk + per-position metrics. All math on CPU.

Usage:
    python tests/multi_chunk_dense_parity/compare.py \\
        --hf tests/multi_chunk_logs/Llama-3.2-1B__hf.pt \\
        --ft tests/multi_chunk_logs/Llama-3.2-1B__ft_chunk8192.pt \\
        --out tests/multi_chunk_logs/Llama-3.2-1B__chunk8192.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="HF bundle (.pt)")
    ap.add_argument("--ft", required=True, help="FT bundle (.pt)")
    ap.add_argument("--out", required=True, help="Output stats JSON")
    ap.add_argument("--rel-thresh", type=float, default=5e-2,
                    help="Acceptance bar: rel(max|Δ|/‖HF‖) < this")
    ap.add_argument("--argmax-thresh", type=float, default=0.95,
                    help="Acceptance bar: overall argmax agreement >= this")
    ap.add_argument("--per-chunk-argmax-thresh", type=float, default=0.93,
                    help="Acceptance bar: every chunk's argmax agreement >= this. "
                         "Uniformity check is the real localization signal.")
    ap.add_argument("--per-chunk-uniformity", type=float, default=0.05,
                    help="Acceptance bar: max-min per-chunk agreement < this. "
                         "A chunk-boundary regression typically shows as the first "
                         "chunk passing while later chunks degrade.")
    args = ap.parse_args()

    hf = torch.load(args.hf, map_location="cpu", weights_only=False)
    ft = torch.load(args.ft, map_location="cpu", weights_only=False)

    hf_logits: torch.Tensor = hf["logits"]
    ft_logits: torch.Tensor = ft["logits"]
    if hf_logits.shape != ft_logits.shape:
        raise SystemExit(
            f"shape mismatch: hf={tuple(hf_logits.shape)} ft={tuple(ft_logits.shape)}"
        )
    chunk_size = ft["chunk_size"]
    num_chunks = ft["num_chunks"]
    chunk_sizes = ft.get("chunk_sizes", [chunk_size] * num_chunks)

    T, V = hf_logits.shape
    print(f"=== Compare ===  model={ft['model']}  T={T}  V={V}  chunks={num_chunks}  chunk_sizes={chunk_sizes}")

    # Pull next-token labels for per-position CE loss. CE under HF
    # logits gives a baseline; CE under FT logits should track within
    # bf16 noise. The DELTA between FT-CE and HF-CE per position is a
    # very sensitive signal: even a 1e-3 logit shift moves CE
    # noticeably for tokens near the top of the distribution.
    input_ids: torch.Tensor = hf["input_ids"] if "input_ids" in hf else ft["input_ids"]
    # next-token target at position t is input_ids[t+1]. Last position
    # has no target — mask it.
    targets = torch.empty(T, dtype=torch.long)
    targets[:-1] = input_ids[1:].long()
    targets[-1] = -100

    BLOCK = 256
    pos_max = torch.empty(T, dtype=torch.float32)
    pos_mean = torch.empty(T, dtype=torch.float32)
    pos_rel = torch.empty(T, dtype=torch.float32)
    pos_ce_hf = torch.empty(T, dtype=torch.float32)
    pos_ce_ft = torch.empty(T, dtype=torch.float32)
    hf_arg = torch.empty(T, dtype=torch.long)
    ft_arg = torch.empty(T, dtype=torch.long)
    max_abs = 0.0
    sum_abs = 0.0
    sum_count = 0
    sum_hf_norm_sq = 0.0
    for s in range(0, T, BLOCK):
        e = min(s + BLOCK, T)
        hf_b = hf_logits[s:e].float()
        ft_b = ft_logits[s:e].float()
        d = (hf_b - ft_b).abs()
        pos_max[s:e] = d.max(dim=-1).values
        pos_mean[s:e] = d.mean(dim=-1)
        norms = hf_b.norm(dim=-1)
        pos_rel[s:e] = pos_max[s:e] / norms.clamp_min(1e-12)
        max_abs = max(max_abs, float(d.max().item()))
        sum_abs += float(d.sum().item())
        sum_count += int(d.numel())
        sum_hf_norm_sq += float((hf_b * hf_b).sum().item())
        hf_arg[s:e] = hf_b.argmax(dim=-1)
        ft_arg[s:e] = ft_b.argmax(dim=-1)
        # Per-position CE under each model. Use log_softmax for numerical
        # stability; mask last-position targets via gather guard.
        tgt_b = targets[s:e].clamp(min=0)  # -100 -> 0; we'll mask after
        lp_hf = torch.log_softmax(hf_b, dim=-1)
        lp_ft = torch.log_softmax(ft_b, dim=-1)
        ce_hf = -lp_hf.gather(-1, tgt_b.unsqueeze(-1)).squeeze(-1)
        ce_ft = -lp_ft.gather(-1, tgt_b.unsqueeze(-1)).squeeze(-1)
        # Mask last position(s) where target is -100.
        bad = (targets[s:e] == -100)
        ce_hf[bad] = float("nan")
        ce_ft[bad] = float("nan")
        pos_ce_hf[s:e] = ce_hf
        pos_ce_ft[s:e] = ce_ft
    mean_abs = sum_abs / max(sum_count, 1)
    hf_norm_total = sum_hf_norm_sq ** 0.5
    rel = max_abs / max(hf_norm_total, 1e-12)
    agree = (hf_arg == ft_arg).float().mean().item()

    print(f"  shape          = {tuple(hf_logits.shape)}")
    print(f"  max|Δ|         = {max_abs:9.5f}")
    print(f"  mean|Δ|        = {mean_abs:9.5f}")
    print(f"  rel (vs ‖hf‖)  = {rel:.3e}")
    print(f"  per-position max|Δ|: max={pos_max.max().item():.4f}  mean={pos_max.mean().item():.4f}  p99={torch.quantile(pos_max, 0.99).item():.4f}")
    print(f"  per-position rel : max={pos_rel.max().item():.3e}  mean={pos_rel.mean().item():.3e}")
    print(f"  argmax agreement: {agree*100:.2f}%")

    # Per-chunk metrics from actual chunk_sizes (handles the partial
    # last chunk: ``chunk_sizes[-1]`` < chunk_size in general).
    print(f"  per-chunk:")
    seg_results = []
    seg_agree_list: list[float] = []
    cursor = 0
    for i, n in enumerate(chunk_sizes):
        s_start = cursor
        s_end = cursor + n
        seg_max = float(pos_max[s_start:s_end].max().item())
        seg_mean = float(pos_mean[s_start:s_end].mean().item())
        seg_agree = float((hf_arg[s_start:s_end] == ft_arg[s_start:s_end]).float().mean().item())
        seg_ce_hf = float(torch.nanmean(pos_ce_hf[s_start:s_end]).item())
        seg_ce_ft = float(torch.nanmean(pos_ce_ft[s_start:s_end]).item())
        seg_ce_delta = seg_ce_ft - seg_ce_hf
        print(
            f"    chunk[{i}] [{s_start:6d}, {s_end:6d}) ({n:5d} tok)  "
            f"max|Δ|={seg_max:7.4f}  mean|Δ|={seg_mean:7.5f}  "
            f"argmax_agree={seg_agree*100:5.2f}%  "
            f"CE_hf={seg_ce_hf:.4f}  CE_ft={seg_ce_ft:.4f}  ΔCE={seg_ce_delta:+.4f}"
        )
        seg_results.append({
            "chunk_idx": i, "start": s_start, "end": s_end, "size": n,
            "max_abs": seg_max, "mean_abs": seg_mean,
            "argmax_agree": seg_agree,
            "ce_hf": seg_ce_hf, "ce_ft": seg_ce_ft, "ce_delta": seg_ce_delta,
        })
        seg_agree_list.append(seg_agree)
        cursor = s_end

    # Stricter parity check: every chunk must individually clear the
    # per-chunk argmax threshold AND the spread (max-min) must be small.
    # A chunk-boundary regression typically shows up as the first chunk
    # passing while later chunks degrade — uniformity catches that
    # while overall rel and overall argmax wash it out.
    min_chunk_agree = min(seg_agree_list) if seg_agree_list else 1.0
    chunk_agree_spread = (max(seg_agree_list) - min(seg_agree_list)) if seg_agree_list else 0.0

    overall_pass = (
        (rel < args.rel_thresh) and (agree >= args.argmax_thresh)
    )
    per_chunk_pass = (
        (min_chunk_agree >= args.per_chunk_argmax_thresh)
        and (chunk_agree_spread <= args.per_chunk_uniformity)
    )
    passed = overall_pass and per_chunk_pass

    print(
        f"  per-chunk argmax: min={min_chunk_agree*100:.2f}%  "
        f"spread={chunk_agree_spread*100:.2f}%  "
        f"(thresh: min>={args.per_chunk_argmax_thresh*100:.0f}%, "
        f"spread<={args.per_chunk_uniformity*100:.0f}%)"
    )

    out = {
        "model": ft["model"],
        "shape": list(hf_logits.shape),
        "T": T, "V": V,
        "chunk_size": chunk_size, "num_chunks": num_chunks,
        "chunk_sizes": chunk_sizes,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel": rel,
        "argmax_agreement": agree,
        "per_position_max_abs": {
            "max": pos_max.max().item(),
            "mean": pos_max.mean().item(),
            "p99": torch.quantile(pos_max, 0.99).item(),
            "p999": torch.quantile(pos_max, 0.999).item(),
        },
        "per_position_rel": {
            "max": pos_rel.max().item(),
            "mean": pos_rel.mean().item(),
        },
        "per_chunk": seg_results,
        "min_chunk_argmax_agree": min_chunk_agree,
        "chunk_argmax_spread": chunk_agree_spread,
        "thresholds": {
            "rel": args.rel_thresh,
            "argmax_agreement": args.argmax_thresh,
            "per_chunk_argmax": args.per_chunk_argmax_thresh,
            "per_chunk_uniformity": args.per_chunk_uniformity,
        },
        "passed": bool(passed),
        "overall_pass": bool(overall_pass),
        "per_chunk_pass": bool(per_chunk_pass),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> {args.out}")
    print("  PASS" if passed else "  FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main()) 