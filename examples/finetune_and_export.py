"""End-to-end: LoRA fine-tune a model with FlexTrain, then export.

Demonstrates the three export modes:

* :func:`flextrain.export.save_hf_full` — full base weights.
* :func:`flextrain.export.save_lora_adapter` — PEFT-format adapter
  (vLLM/sGLang/PEFT compatible). Currently for Llama / Qwen3-dense.
* :func:`flextrain.export.save_hf_merged` — fold LoRA delta into the
  base, then write a full HF dir. Universal: works for every arch
  FlexTrain supports.

After running this script, the output directory contains three
sub-dirs: ``base/``, ``merged/``, ``adapter/``.

Examples
--------
Llama-3.1-8B (LoRA, all three exports)::

    python examples/finetune_and_export.py \\
        --model models/Llama-3.1-8B \\
        --n-steps 100 --rank 16 \\
        --out-dir runs/llama8b-export

Qwen3.5-9B (gated q_proj — PEFT adapter not supported, merged works)::

    python examples/finetune_and_export.py \\
        --model models/Qwen3.5-9B \\
        --n-steps 100 --rank 16 \\
        --skip-lora-adapter \\
        --out-dir runs/qwen35-9b-export
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _train_a_few_steps(am, *, n_steps: int) -> list[float]:
    from flextrain.io.sources import SyntheticTokenSource
    source = SyntheticTokenSource(
        vocab_size=int(am.dims["vocab_size"]), seq_lens=512, seed=0,
    )
    losses = []
    for step in range(n_steps):
        sequences = source.get_sequences(max_token_count=512)
        stats = am.fwd_bwd(sequences, total_tokens_per_step=512)
        am.step()
        mean_loss = stats.total_loss / max(1, stats.total_tokens)
        losses.append(mean_loss)
        if step < 5 or (step + 1) % 10 == 0 or step == n_steps - 1:
            print(f"  step {step+1}: loss={mean_loss:.4f}", flush=True)
    return losses


def main() -> int:
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--model", required=True,
                    help="Path to HF model directory.")
    ap.add_argument("--out-dir", required=True,
                    help="Output dir; gets ``base/``, ``merged/``, "
                         "``adapter/`` sub-dirs.")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=None,
                    help="LoRA alpha. Defaults to rank.")
    ap.add_argument("--max-gpu-gib", type=float, default=22.0)
    ap.add_argument("--max-host-gib", type=float, default=80.0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--skip-lora-adapter", action="store_true",
                    help="Skip ``save_lora_adapter`` (use this for archs "
                         "with gated q_proj like Qwen3.5 — those can only "
                         "use save_hf_merged).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.export import save_hf_full, save_hf_merged, save_lora_adapter

    alpha = args.alpha if args.alpha is not None else float(args.rank)

    print(f"[1/4] Loading {args.model} with LoRA-all (rank={args.rank}, "
          f"alpha={alpha})...")
    am = from_pretrained(
        args.model,
        optimizer=AdamW(AdamWHyperparams(lr=args.lr)),
        max_seq_len=args.seq_len,
        max_global_batch_tokens=args.seq_len,
        max_gpu_mem_bytes=int(args.max_gpu_gib * (1 << 30)),
        max_host_mem_bytes=int(args.max_host_gib * (1 << 30)),
        device="cuda:0",
        lora_targets="all",
        lora_rank=args.rank,
        lora_alpha=alpha,
        verbose=False,
    )
    print(f"   model has {len(am.backbone)} layers, "
          f"vocab={am.dims['vocab_size']}.")

    print(f"\n[2/4] Training {args.n_steps} synthetic-token steps...")
    losses = _train_a_few_steps(am, n_steps=args.n_steps)
    print(f"   loss: {losses[0]:.4f} -> {losses[-1]:.4f}")

    print(f"\n[3/4] Saving exports under {args.out_dir}/ ...")

    # 3a. Full base (the LoRA delta is dropped here — i.e. you get the
    # original pretrained model unchanged in this dir).
    base_dir = os.path.join(args.out_dir, "base")
    save_hf_full(am, base_dir)
    print(f"   base       -> {base_dir}/")

    # 3b. Merged: fold LoRA delta into base, write full HF dir.
    merged_dir = os.path.join(args.out_dir, "merged")
    save_hf_merged(am, merged_dir, keep_lora_after_merge=True)
    print(f"   merged     -> {merged_dir}/  "
          f"(serve as a plain base model, no LoRA support needed)")

    # 3c. PEFT-format adapter (Llama / Qwen3-dense only; gated archs
    # like Qwen3.5* fall through to merged-only).
    if not args.skip_lora_adapter:
        adapter_dir = os.path.join(args.out_dir, "adapter")
        try:
            save_lora_adapter(
                am, adapter_dir, base_model_name_or_path=args.model,
            )
            print(f"   adapter    -> {adapter_dir}/  "
                  f"(serve with vLLM --enable-lora --lora-modules "
                  f"NAME={adapter_dir})")
        except ValueError as e:
            print(f"   adapter    -> SKIPPED: {e}")

    print("\n[4/4] Done.")
    print("Next steps:")
    print(f"  vLLM:   vllm serve {os.path.join(args.out_dir, 'merged')} "
          f"--max-model-len {args.seq_len}")
    if not args.skip_lora_adapter:
        print(f"  vLLM+LoRA:  vllm serve {args.model} --enable-lora "
              f"--lora-modules my={os.path.join(args.out_dir, 'adapter')}")
    print(f"  sGLang: python -m sglang.launch_server --model-path "
          f"{os.path.join(args.out_dir, 'merged')}")
    print(f"  HF:     transformers.AutoModelForCausalLM.from_pretrained("
          f"'{os.path.join(args.out_dir, 'merged')}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
