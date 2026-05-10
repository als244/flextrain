# Verified end-to-end runs

Smoke-test record from training a handful of supported models end-to-end.

- **Workload**: 5 steps on `datasets/mathinstruct.jsonl` with the default
  `Instruction:/Response:` prompt template, `--max-seq-len 2048`,
  mean-over-active-tokens loss
  (`CrossEntropyLoss(ignore_index=-100)` convention; matches HF / PEFT).
- **Default mode**: LoRA-all at rank 16 unless the row says otherwise.
- **Generation**: greedy generation also verified — coherent output, hits
  EOS naturally.
- **Reproduce**: `python experiments/verified_runs.py run-grid --out
  runs/<dir>` shells out to `train.py` per row, captures the per-step
  log, and emits the table at `<dir>/new_table.md`.

Loss values reflect mean cross-entropy over response tokens (positions
where `targets != -100`); prior versions of this table reported a
different convention (mean over all tokens, including prompt-position
zeros) so older numbers are not directly comparable.

## RTX 5090 (31.3 GiB, 192 GiB host) — full sweep, 2026-05-10

All 9 rows re-verified at **auto memory budget** (no manual GPU/host
caps). Per-step metrics (`tok/sec`, `eff TFLOPS`, `hw TFLOPS`,
`peak alloc`, `peak resv`) read directly from `train.py`'s stdout at
step 3 — a mid-run logged data point past step-1 warmup. `peak alloc`
is `torch.cuda.max_memory_allocated()` (live peak); `peak resv` is
`torch.cuda.max_memory_reserved()` (caching-pool peak — what determines
OOM). Effective TFLOPS uses the canonical formula in
`flextrain/cli.py:_get_model_flops_per_token` (`matmul_factor = 4 if
LoRA else 6` — LoRA skips the frozen-weight wgrad — plus the causal
attention term). Hardware TFLOPS adds `recompute_flops / dt`; the gap
reflects the working-set solver's recompute trade-off.

| Model | Params | Arch | Mode | Loss curve (5 steps) | tok/sec | eff TFLOPS | hw TFLOPS | peak alloc | peak resv |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | dense | LoRA | 1.055 → 1.012 | 28,304 | 141.3 | 141.7 | 26.70 | 26.90 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | 0.865 → 0.844 | 22,649 | 107.9 | 125.3 | 25.70 | 28.60 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | 0.865 → 0.673 | 13,331 |  95.3 | 108.3 | 26.40 | 28.80 |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | 0.933 → 0.873 |  5,031 | 153.4 | 153.7 | 27.10 | 27.40 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 0.747 → 0.660 |  4,989 | 159.6 | 159.9 | 27.00 | 27.90 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | 0.744 → 0.465 |  3,244 | 155.7 | 165.9 | 26.30 | 26.60 |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | 1.014 → 0.856 |  1,518 | 156.7 | 163.6 | 27.20 | 27.50 |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | 0.900 → 0.865 |  7,763 |  96.6 | 121.5 | 25.90 | 28.70 |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 0.742 → 0.675 |  5,889 |  74.2 |  91.8 | 24.90 | 28.50 |

The Qwen3.5-9B full-FT loss curve (0.744 → 0.465) reproduces the
historical RTX 3090 reference (0.744 → 0.455) to within ≈0.01.

## RTX 3090 (24 GiB, 117 GiB host) — historical reference

Pre-2026-05 sweep. Most rows were skipped on this hardware due to
memory limits.

| Model | Params | Arch | Mode | Batch tokens | Loss curve (5 steps) |
|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | dense | LoRA | — | _not re-verified_ |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | — | _not re-verified_ |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | — | _not re-verified_ |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | — | _not re-verified_ |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 65k | 0.797 → 0.620 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | 65k | 0.744 → 0.455 |
| Qwen3.5-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | — | _not re-verified_ |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | — | _not re-verified_ |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | — | _not re-verified_ |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 65k | 0.743 → 0.541 |

Additional models supported by the existing arch loaders (require a
larger machine to actually train): Qwen3.6-35B-A3B, Qwen3.5-122B-A10B,
Qwen3.5-397B-A17B, Qwen3-Coder-30B-A3B-Instruct (no new wiring needed;
they reuse `Qwen3_5*` / `Qwen3Moe*` arch ids).
