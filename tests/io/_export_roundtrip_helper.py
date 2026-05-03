"""Helper subprocess for tests/io/test_export_roundtrip_loss.py.

Runs ONE phase (either ``orig`` or ``resumed``) end-to-end:

* ``orig``     — load model, train N steps, save_hf_full or save_hf_merged
                 depending on --mode, dump per-step losses to JSON.
* ``resumed``  — load model from the saved dir, train N steps on the
                 SAME data prefix, dump per-step losses to JSON.

The driver runs each phase in its own subprocess so all GPU memory and
pinned-host buffers are reclaimed by the OS between phases. No clever
in-process teardown needed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flextrain.api import from_pretrained
from flextrain.io.sources import JsonSFTTokenSource
from flextrain.optim.adamw import AdamW, AdamWHyperparams
from flextrain.engine.schedule import split_sequences
from flextrain.export import save_hf_full, save_hf_merged


def _lr(step: int, *, max_lr: float, final_lr: float,
        warmup_steps: int, cooldown_start: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step < cooldown_start:
        return max_lr
    pct = (step - cooldown_start) / max(1, total_steps - cooldown_start)
    pct = min(1.0, max(0.0, pct))
    return final_lr + 0.5 * (max_lr - final_lr) * (1.0 + math.cos(math.pi * pct))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["orig", "resumed"], required=True)
    ap.add_argument("--model-path", required=True,
                    help="Source HF directory (orig) or saved-out dir (resumed).")
    ap.add_argument("--mode", choices=["full", "lora"], required=True)
    ap.add_argument("--n-steps", type=int, required=True)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--lr-full", type=float, default=3e-5)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--max-gpu-gib", type=float, default=20.0)
    ap.add_argument("--max-host-gib", type=float, default=80.0)
    ap.add_argument("--moe-backend", default=None,
                    help="MoE backend for from_pretrained. Default: None "
                         "(use FT default 'flextrain'). Set to 'scattermoe' "
                         "for big MoE models that exceed working_set baseline "
                         "with the flextrain dispatcher.")
    ap.add_argument("--dataset", default="datasets/mathinstruct.jsonl")
    ap.add_argument("--save-to", default=None,
                    help="Where to write save_hf_full output (orig phase only).")
    ap.add_argument("--source-hf-dir", default=None,
                    help="Original HF dir for tokenizer/config copy "
                         "(orig phase only). Defaults to --model-path.")
    ap.add_argument("--losses-out", required=True,
                    help="JSON file to write per-step losses to.")
    args = ap.parse_args()

    print(f"[{args.phase}] model={args.model_path}  mode={args.mode}  "
          f"n_steps={args.n_steps}", flush=True)

    if args.mode == "lora":
        opt = AdamW(
            AdamWHyperparams(lr=args.lr_lora, beta1=0.9, beta2=0.95,
                             eps=1e-8, weight_decay=0.0),
            state_dtype=torch.float32,
        )
        lora_kwargs = dict(lora_targets="all",
                           lora_rank=args.lora_rank,
                           lora_alpha=args.lora_alpha)
    else:
        opt = AdamW(
            AdamWHyperparams(lr=args.lr_full, beta1=0.9, beta2=0.95,
                             eps=1e-8, weight_decay=0.0),
            state_dtype=torch.bfloat16,
        )
        lora_kwargs = {}

    extra_kwargs = {}
    if args.moe_backend:
        # Instantiate the backend object — from_pretrained takes an
        # object, not a string. Mirrors train.py's resolution.
        if args.moe_backend == "flextrain":
            from flextrain.ops.moe_backend import FlextrainMoEExpertCompute
            extra_kwargs["moe_backend"] = FlextrainMoEExpertCompute()
        elif args.moe_backend == "scattermoe":
            from flextrain.ops.moe_backend import ScatterMoEExpertCompute
            extra_kwargs["moe_backend"] = ScatterMoEExpertCompute()
        elif args.moe_backend == "sonicmoe":
            from flextrain.ops.moe_backend import SonicMoEExpertCompute
            extra_kwargs["moe_backend"] = SonicMoEExpertCompute()
        else:
            print(f"unknown --moe-backend {args.moe_backend!r}", flush=True)
            sys.exit(2)
    am = from_pretrained(
        args.model_path,
        optimizer=opt,
        max_seq_len=args.max_seq_len,
        max_global_batch_tokens=args.batch_tokens,
        max_gpu_mem_bytes=int(args.max_gpu_gib * (1 << 30)),
        max_host_mem_bytes=int(args.max_host_gib * (1 << 30)),
        load_weights=True,
        verbose=False,
        **lora_kwargs,
        **extra_kwargs,
    )

    source = JsonSFTTokenSource(
        path=args.dataset,
        tokenizer=args.model_path,
        max_seq_len=args.max_seq_len,
        loop=False,
    )

    losses: list[float] = []
    total_steps = args.n_steps
    warmup_steps = max(1, int(total_steps * 0.1))
    cooldown_start = int(total_steps * 0.8)
    max_lr_val = am.optimizer.hp.lr
    final_lr_val = max_lr_val * 0.1

    train_t0 = time.time()
    for step in range(1, total_steps + 1):
        lr_now = _lr(step, max_lr=max_lr_val, final_lr=final_lr_val,
                     warmup_steps=warmup_steps, cooldown_start=cooldown_start,
                     total_steps=total_steps)
        am.optimizer.hp = type(am.optimizer.hp)(
            **{**am.optimizer.hp.__dict__, "lr": lr_now}
        )
        seqs = source.get_sequences(max_token_count=args.batch_tokens)
        if not seqs:
            print(f"[{args.phase} step {step}] data exhausted", flush=True)
            break

        ws = am.working_set
        push_back = getattr(source, "push_back", None)
        if push_back is not None:
            for _ in range(4):
                rounds, _ = split_sequences(
                    seqs,
                    target_round_tokens=ws.target_round_tokens,
                    max_total_round_tokens=ws.max_total_round_tokens,
                    max_chunk_size=ws.max_chunk_size,
                    max_training_chunks=ws.max_training_chunks,
                    policy=am.chunk_policy,
                )
                if len(rounds) <= ws.target_num_rounds:
                    break
                spilled = rounds[-1]
                if len(spilled) >= len(seqs):
                    break
                for s in reversed(spilled):
                    push_back(s)
                seqs = seqs[: -len(spilled)]

        active = max(1, sum(getattr(s, "active_token_count", len(s)) for s in seqs))
        step_tokens = sum(len(s) for s in seqs)
        t0 = time.time()
        stats = am.fwd_bwd(
            seqs,
            loss_scale_factor=1.0 / active,
            total_tokens_per_step=step_tokens,
        )
        am.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        loss = float(stats.total_loss / active)
        losses.append(loss)
        print(f"[{args.phase} step {step:2d}] lr={lr_now:.2e} loss={loss:.6f} "
              f"tok={step_tokens} active={active} dt={dt*1000:.0f}ms",
              flush=True)

    print(f"[{args.phase}] training loop took {time.time()-train_t0:.1f}s",
          flush=True)

    # Save (orig phase only).
    if args.phase == "orig":
        if not args.save_to:
            print("ERROR: --save-to required for orig phase", flush=True)
            sys.exit(2)
        src = args.source_hf_dir or args.model_path
        print(f"[{args.phase}] saving to {args.save_to} (source={src})",
              flush=True)
        t0 = time.time()
        if args.mode == "lora":
            save_hf_merged(am, args.save_to, hf_source_dir=src)
        else:
            save_hf_full(am, args.save_to, hf_source_dir=src)
        print(f"[{args.phase}] save took {time.time()-t0:.1f}s", flush=True)

    # Dump losses for the driver to compare.
    with open(args.losses_out, "w") as f:
        json.dump({"phase": args.phase, "losses": losses}, f, indent=2)
    print(f"[{args.phase}] wrote {len(losses)} losses to {args.losses_out}",
          flush=True)


if __name__ == "__main__":
    main()
