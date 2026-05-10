"""Side-by-side GPU memory accounting on Qwen3-30B-A3B at the failing
(29 GiB) and working (22 GiB) caps.

For each cap:
  1) Capture solver's verbose Working-Set-Log (Expected GPU Memory Usage,
     Selected Best Option components).
  2) Capture torch.cuda.memory_allocated() / max_memory_allocated() at
     three checkpoints: post-build, post-fwd, post-bwd.
  3) Run torch.cuda.memory._record_memory_history during step 1 and
     dump a snapshot to disk on OOM (or normally, if the run fits) for
     post-mortem inspection.
  4) Print a precise component breakdown:
        baseline weights  +  baseline grads  +  endpoint
      + activation buffer +  transient peak  +  PyTorch caching slop
      + CUDA context (non-PyTorch)
     Compared to the solver's est, identifying where the unaccounted
     bytes live.

Usage:
    python experiments/_mem_diag.py --cap 29   # expected to OOM
    python experiments/_mem_diag.py --cap 22   # expected to fit

The --cap value is in GiB and is passed straight as max_gpu_mem_bytes
to from_pretrained.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import torch


MODEL_PATH = os.environ.get(
    "MEM_DIAG_MODEL",
    "/home/shein/Documents/grad_school/research/flextrain/models/Qwen3-30B-A3B",
)
DATASET = "/home/shein/Documents/grad_school/research/flextrain/datasets/mathinstruct.jsonl"
HOST_CAP_GIB = int(os.environ.get("MEM_DIAG_HOST_CAP", "100"))
MODE = os.environ.get("MEM_DIAG_MODE", "lora")  # lora | full
N_STEPS = int(os.environ.get("MEM_DIAG_N_STEPS", "3"))
BATCH_TOKENS = 65_536
MAX_SEQ_LEN = 2048


def _hw_cost():
    p = Path("runs/_hw_cost_cache.json")
    from flextrain.core.save_level import HardwareCost
    if p.is_file():
        d = json.loads(p.read_text())
        return (HardwareCost(peak_tflops=d["peak_tflops"],
                             pcie_bw_gbps=d["pcie_bw_gbps"]),
                d["mem_bw_gbps"])
    from flextrain.core.hw_probe import probe_hardware
    res = probe_hardware()
    return res.hw_cost, res.mem_bw_gbps


def gpu_mem_snapshot(label):
    """Print a labelled snapshot of GPU memory state."""
    import torch
    torch.cuda.synchronize()
    free_b, total_b = torch.cuda.mem_get_info(0)
    alloc_b = torch.cuda.memory_allocated(0)
    reserved_b = torch.cuda.memory_reserved(0)
    max_alloc_b = torch.cuda.max_memory_allocated(0)
    max_reserved_b = torch.cuda.max_memory_reserved(0)
    non_pt_b = (total_b - free_b) - reserved_b
    print(
        f"[mem] {label:30s}  "
        f"alloc={alloc_b/(1<<30):6.2f}  "
        f"reserved={reserved_b/(1<<30):6.2f}  "
        f"max_alloc={max_alloc_b/(1<<30):6.2f}  "
        f"max_reserved={max_reserved_b/(1<<30):6.2f}  "
        f"free={free_b/(1<<30):6.2f}  "
        f"non_pt={non_pt_b/(1<<30):6.2f}  "
        f"total={total_b/(1<<30):6.2f}",
        flush=True,
    )
    return {
        "label": label,
        "allocated_gib": alloc_b / (1 << 30),
        "reserved_gib": reserved_b / (1 << 30),
        "max_allocated_gib": max_alloc_b / (1 << 30),
        "max_reserved_gib": max_reserved_b / (1 << 30),
        "free_gib": free_b / (1 << 30),
        "non_pytorch_gib": non_pt_b / (1 << 30),
        "total_gib": total_b / (1 << 30),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True,
                    help="GPU cap in GiB (e.g. 22 or 29)")
    args = ap.parse_args()

    print(f"\n{'='*70}\n  MEMORY DIAGNOSTIC — Qwen3-30B-A3B LoRA, cap={args.cap} GiB\n{'='*70}\n",
          flush=True)

    snap_file = Path(f"runs/_mem_diag_{args.cap}gib.pickle")

    # Initial snapshot (before any allocation).
    initial = gpu_mem_snapshot("initial")

    # Capture solver's verbose log to a string (so we can grep it later).
    solver_log_io = io.StringIO()

    import flextrain
    from flextrain.io.sources import JsonSFTTokenSource
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    hw, mem_bw = _hw_cost()
    print(f"[hw] peak_tflops={hw.peak_tflops:.1f}, pcie_bw={hw.pcie_bw_gbps:.1f} GB/s, mem_bw={mem_bw:.1f} GB/s",
          flush=True)

    # Start cuda memory history recording so we can dump a snapshot on
    # OOM (or after a successful step) for post-mortem inspection.
    torch.cuda.memory._record_memory_history(enabled="all", max_entries=200_000)

    # Build with verbose=True to capture solver's per-component breakdown.
    print("\n--- BUILD ---", flush=True)
    t0 = time.time()
    am = None
    build_err = None
    with redirect_stdout(solver_log_io), redirect_stderr(solver_log_io):
        try:
            am = flextrain.from_pretrained(
                MODEL_PATH,
                optimizer=AdamW(AdamWHyperparams(
                    lr=3e-5, beta1=0.9, beta2=0.95, weight_decay=0.0,
                )),
                max_seq_len=MAX_SEQ_LEN,
                max_global_batch_tokens=BATCH_TOKENS,
                max_gpu_mem_bytes=int(args.cap * (1 << 30)),
                max_host_mem_bytes=int(HOST_CAP_GIB * (1 << 30)),
                lora_targets=("all" if MODE == "lora" else None),
                lora_rank=16, lora_alpha=16.0,
                hw_cost=hw, mem_bw_gbps=mem_bw,
                verbose=True,
            )
        except Exception as e:
            build_err = traceback.format_exc()

    solver_log = solver_log_io.getvalue()

    # Print just the structured solver lines from the log (skip noisy
    # internal prints).
    print("\n--- SOLVER OUTPUT ---", flush=True)
    for line in solver_log.splitlines():
        if any(tag in line for tag in [
            "[Working Set Log]", "[Save Level Plan]", "[BufferManager]",
            "[from_pretrained] Engine", "Expected GPU", "Expected Host",
        ]):
            print(line, flush=True)

    if am is None:
        print("\n--- BUILD FAILED ---", flush=True)
        print(build_err, flush=True)
        torch.cuda.memory._dump_snapshot(str(snap_file))
        print(f"[snapshot] dumped to {snap_file}", flush=True)
        return

    print(f"[time] build = {time.time()-t0:.1f}s", flush=True)
    after_build = gpu_mem_snapshot("after_build")

    # Reset peaks so step-1 peak is what we measure cleanly.
    torch.cuda.reset_peak_memory_stats(0)

    # Build the data source + 1 batch.
    print("\n--- BATCH ---", flush=True)
    src = JsonSFTTokenSource(
        DATASET, tokenizer=MODEL_PATH,
        prompt_field="instruction", response_field="output",
        input_field=None, max_seq_len=MAX_SEQ_LEN, min_seq_len=32,
        loop=True,
    )
    seqs = src.get_sequences(max_token_count=BATCH_TOKENS)
    active = sum(s.active_token_count for s in seqs)
    seq_tokens = sum(len(s.tokens) for s in seqs)
    print(f"[batch] {len(seqs)} seqs, {seq_tokens} total tokens, {active} active",
          flush=True)

    # Run N_STEPS, snapshotting peaks after each. Reset peak counter
    # between steps so we get per-step deltas. Reported "after_step"
    # snapshot is taken after the LAST step (= cumulative max across
    # all steps).
    after_batch = gpu_mem_snapshot("after_batch_load")
    step_err = None
    per_step_peaks = []
    for step_idx in range(1, N_STEPS + 1):
        print(f"\n--- STEP {step_idx} ---", flush=True)
        # Refresh batch each step so the data isn't degenerate.
        seqs_i = src.get_sequences(max_token_count=BATCH_TOKENS)
        active_i = sum(s.active_token_count for s in seqs_i)
        seq_tokens_i = sum(len(s.tokens) for s in seqs_i)
        torch.cuda.reset_peak_memory_stats(0)
        t1 = time.time()
        try:
            stats = am.fwd_bwd(
                seqs_i,
                loss_scale_factor=1.0/active_i,
                total_tokens_per_step=active_i,
            )
            am.step()
            torch.cuda.synchronize()
            loss = float(stats.total_loss) / active_i
            ma = torch.cuda.max_memory_allocated(0) / (1 << 30)
            mr = torch.cuda.max_memory_reserved(0) / (1 << 30)
            per_step_peaks.append({
                "step": step_idx, "loss": loss,
                "max_alloc_gib": ma, "max_reserved_gib": mr,
                "dt_s": time.time() - t1,
                "active": active_i, "total": seq_tokens_i,
            })
            print(f"[step] step={step_idx} loss={loss:.4f} dt={time.time()-t1:.2f}s "
                  f"max_alloc={ma:.2f} max_reserved={mr:.2f}", flush=True)
        except Exception as e:
            step_err = traceback.format_exc()
            print(f"\n--- STEP {step_idx} FAILED ---", flush=True)
            print(step_err, flush=True)
            break

    after_step = gpu_mem_snapshot("after_last_step_or_oom")

    # Dump memory snapshot for offline analysis (open with
    # https://pytorch.org/memory_viz).
    torch.cuda.memory._dump_snapshot(str(snap_file))
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"\n[snapshot] dumped to {snap_file} (open at pytorch.org/memory_viz)",
          flush=True)

    # ----- Component summary -----
    print("\n--- SUMMARY: solver est vs runtime peak ---", flush=True)
    print(f"  cap                     = {args.cap:6.2f} GiB")
    print(f"  initial.non_pytorch     = {initial['non_pytorch_gib']:6.2f} GiB  (CUDA context overhead before any flextrain)")
    print(f"  after_build.allocated   = {after_build['allocated_gib']:6.2f} GiB  (resident weights+grads+opt+ring buffers)")
    print(f"  after_build.reserved    = {after_build['reserved_gib']:6.2f} GiB  (PyTorch caching pool size at build time)")
    print(f"  after_step.max_allocated= {after_step['max_allocated_gib']:6.2f} GiB  (peak live during fwd+bwd)")
    print(f"  after_step.max_reserved = {after_step['max_reserved_gib']:6.2f} GiB  (peak caching-pool incl. fragmentation)")
    if "Expected GPU Memory" in solver_log:
        for line in solver_log.splitlines():
            if "Expected GPU Memory" in line:
                print(f"  solver est              = {line.strip()}")
                break

    print(f"\n  GAP analysis:")
    print(f"    runtime peak_alloc - after_build_alloc = {after_step['max_allocated_gib'] - after_build['allocated_gib']:6.2f} GiB  "
          f"(transient fwd+bwd allocations: chunk activations, attn workspace, scatter buffers, "
          f"flash-attn internal dq_accum, fla scratch, etc.)")
    print(f"    runtime peak_reserved - peak_alloc     = {after_step['max_reserved_gib'] - after_step['max_allocated_gib']:6.2f} GiB  "
          f"(PyTorch caching-pool fragmentation overhead)")

    # Save structured snapshots so we can diff across caps.
    out = {
        "cap_gib": args.cap,
        "initial": initial,
        "after_build": after_build,
        "after_batch_load": after_batch,
        "after_step": after_step,
        "per_step_peaks": per_step_peaks,
        "step_err": step_err,
        "build_err": build_err,
        "expected_gpu_line": next(
            (l.strip() for l in solver_log.splitlines() if "Expected GPU Memory" in l),
            None,
        ),
    }
    diag_path = Path(f"runs/_mem_diag_{args.cap}gib.json")
    diag_path.write_text(json.dumps(out, indent=2))
    print(f"\n[diag] structured summary saved to {diag_path}", flush=True)


if __name__ == "__main__":
    main()
