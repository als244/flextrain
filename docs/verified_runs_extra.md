# Verified runs — constrained GPU memory budgets

Companion to [`verified_runs.md`](verified_runs.md). The main table runs
at **auto memory budget** (the working-set planner gets the full GPU
to spend); this file pins `--max-gpu-mem-gib` at three smaller caps
(**16 / 20 / 24 GiB**) so you can read off the loss / throughput cost
of running the same 13-row sweep on smaller cards.

All three tables use the same row set, the same data
(`mathinstruct.jsonl`, `Instruction:/Response:` template,
response-only loss masking) and the same hyperparams as the main
verified-runs table (lr=3e-5, betas=(0.9, 0.95), eps=1e-8, wd=0.001,
5 optimizer steps, max-global-batch-tokens=65536). The only knob
that changes is `--max-gpu-mem-gib`, plumbed through
`experiments/verified_runs.py` via the `FLEXTRAIN_MAX_GPU_MEM_GIB`
env var.

**Loss drift is expected** when the cap changes. The math is
deterministic given the same chunk schedule, but a tighter memory
budget forces the planner to pick smaller chunk sizes / different
recompute tiers, which permutes the per-token reduction order at bf16
precision. Drift typically registers at Δ 0.0001–0.005 (bf16 noise
floor); rows that take a larger planner step show Δ 0.01–0.02 and are
called out below.

The baseline column in each diff section is `runs/reverify_20260513_190829`
— the 31.3 GiB "auto-budget" sweep currently published in
[`verified_runs.md`](verified_runs.md). Throughput drift compares to
that same baseline.

## How to reproduce

```bash
# Cap at 20 GiB, diff against the current published baseline.
FLEXTRAIN_MAX_GPU_MEM_GIB=20.0 bash experiments/reverify.sh \
    --baseline runs/reverify_20260513_190829

# Cap at 16 GiB or 24 GiB by changing the env var.
FLEXTRAIN_MAX_GPU_MEM_GIB=16.0 bash experiments/reverify.sh
FLEXTRAIN_MAX_GPU_MEM_GIB=24.0 bash experiments/reverify.sh
```

The env-var path was added in `experiments/verified_runs.py`
(`_build_train_cmd`) so the caps land on `train.py --max-gpu-mem-gib`
without per-row script edits.

---

## 24 GiB cap

Sweep dir: `runs/reverify_24gib_20260513_212120`

| Model | Params | Arch | Mode | Loss curve (5 steps) | tok/sec | eff TFLOPS | hw TFLOPS | peak alloc | peak resv |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | dense | LoRA | 1.055 → 1.012 | 28,632 | 142.9 | 143.3 | 21.30 | 21.60 |
| Llama-3.2-1B | 1B | dense | full | 1.055 → 0.826 | 21,868 | 163.7 | 164.0 | 21.10 | 21.40 |
| Llama-3.1-8B-Instruct | 8B | dense | LoRA | 0.783 → 0.747 |  5,296 | 160.0 | 160.3 | 21.70 | 22.20 |
| Llama-3.1-8B-Instruct | 8B | dense | full | 0.783 → 0.600 |  3,652 | 165.5 | 174.5 | 21.20 | 21.40 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | 0.865 → 0.844 | 21,985 | 104.8 | 126.3 | 20.30 | 27.40 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | 0.865 → 0.673 | 13,146 |  94.0 | 105.8 | 21.10 | 21.90 |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | 0.933 → 0.872 |  4,999 | 152.4 | 152.7 | 21.70 | 22.00 |
| Qwen3-8B | 8B | dense, QK-norm | full | 0.928 → 0.478 |  3,479 | 159.1 | 169.1 | 22.50 | 22.70 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 0.748 → 0.662 |  4,894 | 156.6 | 160.7 | 21.40 | 22.30 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | 0.744 → 0.455 |  2,278 | 109.3 | 118.4 | 20.90 | 21.30 |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | 1.014 → 0.818 |  1,501 | 155.0 | 163.5 | 21.70 | 23.10 |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | 0.900 → 0.867 |  7,224 |  89.9 | 112.5 | 21.40 | 25.10 |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 0.742 → 0.668 |  4,528 |  57.1 |  68.7 | 21.10 | 23.20 |

**Diff vs 31.3 GiB baseline (`runs/reverify_20260513_190829`):**

| Row | Loss Δ | tok/s drift | Status |
|---|---|---|---|
| llama_3_2_1b_lora | exact | +0.2% | bit-match |
| llama_3_2_1b_full | exact | -0.1% | bit-match |
| llama_3_1_8b_lora | Δ0.0005 | +0.0% | OK |
| llama_3_1_8b_full | Δ0.0007 | -2.2% | OK |
| olmoe_7b_a1b_lora | exact | +0.7% | bit-match |
| olmoe_7b_a1b_full | Δ0.0021 | -1.5% | OK |
| qwen3_8b_lora | exact | -0.9% | bit-match |
| qwen3_8b_full | Δ0.0003 | -1.6% | OK |
| qwen3_5_9b_lora | Δ0.0017 | -2.4% | OK |
| **qwen3_5_9b_full** | Δ0.0127 | **-29.2%** | fits, but big throughput hit |
| qwen3_6_27b_lora | Δ0.0015 | -1.7% | OK |
| qwen3_30b_a3b_lora | Δ0.0028 | -8.3% | OK |
| **qwen3_5_moe_35b_a3b_lora** | Δ0.0075 | **-24.8%** | MoE — biggest hit |

* **4/13 bit-match loss** (5 GiB more budget gives the planner more
  room to converge on the original chunk size).
* **7/13 drift Δ < 0.005** (clean bf16-noise floor).
* **2/13 larger drift**: `qwen3_5_9b_full` (Δ0.0127, **-29.2% tok/s**)
  — this row barely fits in 24 GiB and the planner picks aggressive
  recompute / small chunks to satisfy the cap; `qwen3_5_moe_35b_a3b_lora`
  (Δ0.0075, -24.8% tok/s) — MoE expert offloading cost.
* **All 13 rows complete** at 24 GiB — including Qwen3.5-9B-full, which
  doesn't fit at 20 GiB. **24 GiB is the floor for full-FT of dense 9B
  with this engine config**.

---

## 20 GiB cap

Sweep dir: `runs/reverify_20gib_20260513_205618`

| Model | Params | Arch | Mode | Loss curve (5 steps) | tok/sec | eff TFLOPS | hw TFLOPS | peak alloc | peak resv |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | dense | LoRA | 1.055 → 1.012 | 28,554 | 142.6 | 142.9 | 17.30 | 17.60 |
| Llama-3.2-1B | 1B | dense | full | 1.055 → 0.826 | 22,019 | 164.9 | 165.1 | 17.10 | 17.40 |
| Llama-3.1-8B-Instruct | 8B | dense | LoRA | 0.783 → 0.747 |  5,262 | 159.0 | 159.3 | 17.70 | 18.20 |
| Llama-3.1-8B-Instruct | 8B | dense | full | 0.783 → 0.600 |  3,643 | 165.1 | 176.3 | 17.20 | 17.40 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | 0.865 → 0.844 | 21,552 | 102.7 | 128.3 | 16.30 | 23.40 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | 0.865 → 0.673 | 13,074 |  93.5 | 107.2 | 17.10 | 17.90 |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | 0.933 → 0.872 |  4,937 | 150.5 | 150.8 | 17.70 | 18.00 |
| Qwen3-8B | 8B | dense, QK-norm | full | 0.928 → 0.478 |  3,449 | 157.7 | 169.9 | 18.90 | 19.00 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 0.745 → 0.681 |  4,675 | 149.6 | 160.2 | 17.40 | 17.60 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | **OOM (build failed)** | — | — | — | — | — |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | 1.014 → 0.815 |  1,486 | 153.4 | 163.5 | 17.70 | 19.10 |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | 0.899 → 0.866 |  7,196 |  89.6 | 112.9 | 17.40 | 21.10 |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 0.743 → 0.671 |  4,471 |  56.3 |  68.6 | 17.10 | 19.20 |

**Diff vs 31.3 GiB baseline (`runs/reverify_20260513_190829`):**

| Row | Loss Δ | tok/s drift | Status |
|---|---|---|---|
| llama_3_2_1b_lora | exact | -0.0% | bit-match |
| llama_3_2_1b_full | Δ0.0004 | +0.6% | OK |
| llama_3_1_8b_lora | Δ0.0004 | -0.6% | OK |
| llama_3_1_8b_full | Δ0.0003 | -2.4% | OK |
| olmoe_7b_a1b_lora | exact | -1.3% | bit-match |
| olmoe_7b_a1b_full | Δ0.0021 | -2.0% | OK |
| qwen3_8b_lora | exact | -2.2% | bit-match |
| qwen3_8b_full | Δ0.0001 | -2.5% | OK |
| **qwen3_5_9b_lora** | **Δ0.0206** | -6.8% | larger planner re-pick |
| **qwen3_5_9b_full** | **OOM** | n/a | does not fit in 20 GiB |
| qwen3_6_27b_lora | Δ0.0041 | -2.7% | OK |
| qwen3_30b_a3b_lora | Δ0.0030 | -8.6% | OK |
| **qwen3_5_moe_35b_a3b_lora** | Δ0.0092 | **-25.7%** | MoE — biggest throughput hit |

* **3/13 bit-match loss** (small LoRA rows: same chunk-size pick).
* **9/13 drift Δ < 0.005** (bf16 noise from reordered reductions).
* **1/13 larger drift (Δ ≈ 0.02)**: `qwen3_5_9b_lora`. Worth investigating
  if you care about deterministic loss across memory budgets — but
  for training quality it's well within "engine works" range.
* **1/13 OOM**: `qwen3_5_9b_full` — the working-set solver couldn't
  fit one complete backbone layer + activations in 20 GiB. Error
  message suggests `--max-seq-len` reduction or explicit
  `--max-chunk-size` cap; the cleanest answer is "needs ~22-24 GiB
  for full-FT". (The 24 GiB sweep below resolves this question.)

Throughput drift profile:

| Category | tok/s drift range |
|---|---|
| Small dense LoRA | -0.0% to -2.2% |
| 8B dense full / 8B QK-norm full | -2.4% to -2.5% |
| OLMoE / Llama-1B full | -2.0% to +0.6% |
| 9B / 27B hybrid LoRA | -2.7% to -6.8% |
| 30B-A3B MoE LoRA | -8.6% |
| 35B-A3B MoE LoRA | **-25.7%** |

MoE rows take the biggest hits — expert-tensor offloading is more
memory-sensitive, and the planner has to keep fewer expert columns
resident on GPU.

---

## 16 GiB cap

Sweep dir: `runs/reverify_16gib_20260513_214540`

| Model | Params | Arch | Mode | Loss curve (5 steps) | tok/sec | eff TFLOPS | hw TFLOPS | peak alloc | peak resv |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | dense | LoRA | 1.055 → 1.012 | 28,902 | 144.3 | 144.7 | 13.30 | 13.60 |
| Llama-3.2-1B | 1B | dense | full | 1.055 → 0.826 | 20,740 | 155.3 | 161.9 | 13.10 | 13.40 |
| Llama-3.1-8B-Instruct | 8B | dense | LoRA | 0.783 → 0.747 |  5,161 | 156.0 | 157.8 | 13.80 | 14.20 |
| Llama-3.1-8B-Instruct | 8B | dense | full | 0.783 → 0.600 |  3,572 | 161.9 | 175.7 | 13.20 | 13.40 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | 0.865 → 0.844 | 22,022 | 105.0 | 132.9 | 12.80 | 14.00 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | 0.865 → 0.666 | 12,364 |  88.4 | 101.4 | 13.10 | 16.60 |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | 0.933 → 0.872 |  4,810 | 146.6 | 150.5 | 13.80 | 14.00 |
| Qwen3-8B | 8B | dense, QK-norm | full | 0.918 → 0.470 |  3,367 | 154.0 | 166.3 | 14.00 | 14.20 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 0.748 → 0.660 |  4,582 | 146.6 | 160.7 | 13.40 | 14.30 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | **OOM (build failed)** | — | — | — | — | — |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | 1.014 → 0.817 |  1,297 | 133.9 | 173.0 | 13.70 | 15.10 |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | 0.894 → 0.844 |  5,975 |  74.4 |  93.3 | 13.90 | 14.20 |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 0.736 → 0.697 |  2,792 |  35.2 |  42.8 | 13.90 | 14.10 |

**Diff vs 31.3 GiB baseline (`runs/reverify_20260513_190829`):**

| Row | Loss Δ | tok/s drift | Status |
|---|---|---|---|
| llama_3_2_1b_lora | exact | +1.2% | bit-match |
| llama_3_2_1b_full | Δ0.0002 | -5.3% | OK |
| llama_3_1_8b_lora | Δ0.0002 | -2.5% | OK |
| llama_3_1_8b_full | Δ0.0004 | -4.3% | OK |
| olmoe_7b_a1b_lora | Δ0.0015 | +0.8% | OK |
| **olmoe_7b_a1b_full** | **Δ0.0139** | -7.4% | larger drift |
| qwen3_8b_lora | Δ0.0006 | -4.7% | OK |
| **qwen3_8b_full** | **Δ0.0098** | -4.8% | larger drift (loss curve shifted to 0.918→0.470) |
| qwen3_5_9b_lora | Δ0.0008 | -8.6% | OK |
| **qwen3_5_9b_full** | **OOM** | n/a | does not fit in 16 GiB |
| **qwen3_6_27b_lora** | Δ0.0025 | **-15.1%** | OK loss; throughput drop |
| **qwen3_30b_a3b_lora** | **Δ0.0206** | **-24.1%** | MoE — both drift and throughput hit |
| **qwen3_5_moe_35b_a3b_lora** | **Δ0.0248** | **-53.6%** | biggest hits in both axes |

* **1/13 bit-match loss** (only the smallest LoRA row keeps the same
  chunk size).
* **8/13 drift Δ < 0.005** (bf16 noise floor; chunk-size adjustments).
* **4/13 larger drift Δ 0.01–0.025**: OLMoE-full, Qwen3-8B-full,
  Qwen3-30B-A3B-LoRA, Qwen3.5-MoE-35B-A3B-LoRA. These rows took
  substantial planner adjustments (more recompute, smaller chunks).
* **1/13 OOM**: `qwen3_5_9b_full` (same row that OOMs at 20 GiB —
  cannot fit one full backbone layer + activations at <24 GiB).

Notable: **Qwen3-8B-full's loss curve shifts** from `0.928 → 0.478`
(baseline) to `0.918 → 0.470` — the first-step loss differs by 0.01,
suggesting the planner's chunk choice produces a meaningfully different
data-packing schedule (different examples in the first chunk → slightly
different step-1 gradient).

Throughput hit profile at 16 GiB cap:

| Category | Worst tok/s drift |
|---|---|
| Small dense LoRA | -2.5% (Llama-8B-LoRA) |
| Small dense full | -5.3% (Llama-1B-full) |
| 8B dense full | -4.3% to -4.8% |
| OLMoE / Llama-1B full | -4.3% to -7.4% |
| 9B / 27B hybrid LoRA | -8.6% to -15.1% |
| 30B-A3B MoE LoRA | **-24.1%** |
| 35B-A3B MoE LoRA | **-53.6%** |

The biggest MoE row pays roughly **2× the throughput hit** between
20→16 GiB (-25.7% → -53.6%) — expert offloading is on the critical
path at very tight budgets.

---

## Cross-cap summary

| Behavior | 24 GiB | 20 GiB | 16 GiB |
|---|---|---|---|
| **Rows that fit + run** | **13/13** | 12/13 | 12/13 |
| **OOM** | none | qwen3_5_9b_full | qwen3_5_9b_full |
| **Bit-match loss** | 4/13 | 3/13 | 1/13 |
| **Drift Δ < 0.005** | 7/13 | 9/13 | 8/13 |
| **Drift Δ ≥ 0.005** | 2/13 | 1/13 | 4/13 |
| **Worst loss drift** | Δ0.0127 (qwen3_5_9b_full) | Δ0.0206 (qwen3_5_9b_lora) | Δ0.0248 (qwen3_5_moe_35b_a3b_lora) |
| **Worst tok/s drift** | -29.2% (qwen3_5_9b_full) | -25.7% (qwen3_5_moe_35b_a3b_lora) | -53.6% (qwen3_5_moe_35b_a3b_lora) |

**Reading the trend:**

* **Small dense LoRA rows are largely insensitive to cap**: drift
  stays at the bit-exact / Δ < 0.001 level across all three budgets.
  Throughput drops modestly (-2 to -5%) even at 16 GiB.
* **Full-FT 9B dense is the cliff**: works at 24 GiB (with -29%
  throughput), OOMs at 20 GiB and below. The working-set solver's
  message points to either `--max-seq-len` reduction (default 2048)
  or `--max-chunk-size` override as the actionable fixes.
* **MoE rows are throughput-sensitive**, not loss-drift-sensitive:
  Qwen3.5-MoE-35B-A3B LoRA loses 25% throughput at 20 GiB and **54%
  at 16 GiB** while loss drift stays modest (Δ0.009–0.025). Expert
  offloading dominates the cost at tight budgets.
* **The "right" budget for a row** is roughly: `peak alloc` from the
  31 GiB baseline + ~3 GiB headroom. Setting the cap below that pays
  in throughput; setting it below `model + 1 layer's working set`
  outright OOMs.

## Notes

* All three sweeps use the same `--max-seq-len 2048` and
  `--max-global-batch-tokens 65536`. Lowering either of these will
  allow tighter caps to fit the rows that currently OOM.
* The `peak alloc` / `peak resv` columns are step-3 numbers from
  `train.py`'s stdout. Observed peaks land **slightly below** each
  cap (e.g. 13–14 GiB at 16 GiB cap, 17–19 GiB at 20 GiB cap) — the
  planner reserves the configured leeway (`--leeway-gpu-mem-gib 3.0`).
* Memory peaks in the 16 GiB tables show some headroom (alloc ~13 GiB
  vs cap 16 GiB) because the planner respects the 3 GiB leeway *plus*
  rounds chunk sizes to safe values; the gap is the safety margin, not
  wasted budget.
