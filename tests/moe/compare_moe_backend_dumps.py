"""Streaming comparison of two MoE-backend dump directories.

Reads one (run_a, run_b) tensor pair at a time, computes
cos / mean_diff / std_diff / max_abs / ref_scale, writes one CSV row,
deletes the pair if --rm is set. Bounded memory; bounded disk after
the streaming pass.

Only compares ``*_dy.pt`` and ``*_g_router.pt`` by default — the
g_up/g_down accumulators are huge and downstream of dy correctness,
so dy at every layer gives an equivalent signal at much smaller cost.

Run on the machine where the dumps live:

  python tests/scratch/compare_moe_backend_dumps.py \\
    --dir-a /home/as1669/storage/flextrain/moe_dump/flextrain \\
    --dir-b /home/as1669/storage/flextrain/moe_dump/sonicmoe \\
    --out /home/as1669/storage/flextrain/moe_dump/compare.csv

Add ``--rm`` to delete each pair after stats are computed (frees disk
progressively). Add ``--include-big`` to also compare g_up/g_down
(slow, large memory; only do this if you have headroom).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch


_BIG_NAMES = ("g_up", "g_down")
# All names worth comparing — fwd: ffn_norm_output, x_router,
# chosen_experts, router_weights, out; bwd: dy, g_router (+ optional
# g_up/g_down via --include-big).
_DEFAULT_NAMES = (
    "ffn_norm_output", "x_router", "chosen_experts", "router_weights",
    "out", "dy", "g_router",
)
COS_TOL = 0.999


def _diffstats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a_f = a.float().flatten()
    b_f = b.float().flatten()
    diff = a_f - b_f
    abs_diff = diff.abs()
    cos = torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0), dim=-1,
    ).item()
    return {
        "cos": cos,
        "mean_diff": diff.mean().item(),
        "std_diff": diff.std().item(),
        "max_abs": abs_diff.max().item(),
        "ref_scale": b_f.abs().mean().item(),
    }


def _name_from_filename(stem: str) -> str:
    # New format: <phase>_layerNNN_<name>  → split into 3 parts max
    # Legacy format: callNNN_layerNNN_<name>  → also 3 parts
    parts = stem.split("_", 2)
    return parts[-1] if len(parts) == 3 else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-a", required=True, help="dump dir for run A (e.g. flextrain)")
    ap.add_argument("--dir-b", required=True, help="dump dir for run B (e.g. sonicmoe)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--rm", action="store_true",
                    help="delete each tensor pair after stats are computed")
    ap.add_argument("--include-big", action="store_true",
                    help="also compare g_up / g_down (large tensors, slow)")
    args = ap.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    if not dir_a.is_dir() or not dir_b.is_dir():
        print(f"missing dir(s): {dir_a} {dir_b}", file=sys.stderr)
        return 2

    names_to_compare = set(_DEFAULT_NAMES)
    if args.include_big:
        names_to_compare.update(_BIG_NAMES)

    files_a = sorted(dir_a.glob("*.pt"))
    files_b_set = {p.name for p in dir_b.glob("*.pt")}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_fail = 0
    n_skip_big = 0
    failed_files: list[str] = []
    missing_in_b: list[str] = []

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "filename", "name", "cos", "mean_diff", "std_diff",
            "max_abs", "ref_scale", "status",
        ])
        for pa in files_a:
            name = _name_from_filename(pa.stem)
            if name not in names_to_compare:
                if name in _BIG_NAMES:
                    n_skip_big += 1
                continue
            if pa.name not in files_b_set:
                missing_in_b.append(pa.name)
                continue

            pb = dir_b / pa.name
            try:
                a = torch.load(pa, map_location="cpu", weights_only=True)
                b = torch.load(pb, map_location="cpu", weights_only=True)
            except Exception as e:
                print(f"load error for {pa.name}: {e}", file=sys.stderr)
                continue

            if a.shape != b.shape:
                w.writerow([
                    pa.name, name, "", "", "", "", "",
                    f"SHAPE_MISMATCH a={tuple(a.shape)} b={tuple(b.shape)}",
                ])
                n_fail += 1
                n_total += 1
                failed_files.append(pa.name)
            else:
                s = _diffstats(a, b)
                ok = s["cos"] >= COS_TOL
                if not ok:
                    n_fail += 1
                    failed_files.append(pa.name)
                w.writerow([
                    pa.name, name,
                    f"{s['cos']:.6f}",
                    f"{s['mean_diff']:.3e}",
                    f"{s['std_diff']:.3e}",
                    f"{s['max_abs']:.3e}",
                    f"{s['ref_scale']:.3e}",
                    "OK" if ok else "FAIL",
                ])
                n_total += 1

            del a, b
            if args.rm:
                pa.unlink(missing_ok=True)
                pb.unlink(missing_ok=True)

    print(f"\n--- Summary ---")
    print(f"  CSV: {out_path}")
    print(f"  total tensors compared: {n_total}")
    print(f"  failures (cos < {COS_TOL}): {n_fail}")
    if n_skip_big:
        print(f"  skipped big (g_up/g_down): {n_skip_big} (use --include-big to compare)")
    if missing_in_b:
        print(f"  in A but not B: {len(missing_in_b)} (e.g. {missing_in_b[:3]})")
    if failed_files:
        print(f"  failing files (first 10): {failed_files[:10]}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
