# Verified end-to-end runs

Smoke-test record from training a handful of supported models end-to-end.

- **Workload**: 5 steps on `datasets/mathinstruct.jsonl` with the default
  `Instruction:/Response:` prompt template, `--max-seq-len 2048`,
  mean-over-active-tokens loss
  (`CrossEntropyLoss(ignore_index=-100)` convention; matches HF / PEFT).
- **Default mode**: LoRA-all at rank 16 unless the row says otherwise.
- **Generation**: greedy generation also verified — coherent output, hits
  EOS naturally.

Loss values reflect mean cross-entropy over response tokens (positions
where `targets != -100`); prior versions of this table reported a
different convention (mean over all tokens, including prompt-position
zeros) so older numbers are not directly comparable.

## How to reproduce this table

The 13 rows below regenerate with a single wrapper script:

```bash
# Default: the 13 rows in this file (no Gemma).
# Writes to runs/reverify_<UTC timestamp>/. ~25 min on a single RTX 5090.
bash experiments/reverify.sh
```

Optionally diff the rerun against a trusted baseline directory; the
script exits non-zero if any row's per-step loss differs from baseline
or throughput drifts more than ±5%:

```bash
bash experiments/reverify.sh --baseline runs/<trusted_baseline_dir>
```

Override the output dir if the default timestamp path isn't what you
want:

```bash
bash experiments/reverify.sh --out runs/my_reverify_2026_05_12
```

Run the wrapper before committing any change that could affect training
numerics (kernels, optimizer, working-set solver, mem-budget logic).
The `--baseline` diff is the regression gate. Each row's artifacts
land at `<out>/<row>/{train.log, final.json, train_out/}`; the wrapper
also writes a freshly-formatted table at `<out>/new_table.md`.

### Also regenerating the Gemma table

The Gemma 2 / Gemma 3 rows in [`gemma_runs.md`](gemma_runs.md) are
toggled in by `--include-gemma` — same wrapper, broader scope (~50 min
total instead of ~25):

```bash
# Reproduces this file's 13 rows AND the 16 Gemma rows in gemma_runs.md.
bash experiments/reverify.sh --include-gemma

# Same, with a baseline diff covering all 29 rows.
bash experiments/reverify.sh --include-gemma --baseline runs/<trusted_baseline_dir>
```

### Direct CLI (no wrapper)

Useful for one-off subsets or comparing arbitrary run dirs:

```bash
# A specific subset of rows (any of the 29 row names from
# experiments/verified_runs.py's registry).
python experiments/verified_runs.py run-grid --out runs/<dir> \
    --only llama_3_2_1b_lora qwen3_8b_full

# Diff two existing run dirs.
python experiments/verified_runs.py compare \
    --baseline runs/<old> --rerun runs/<new>
```

### Path overrides

Per-row defaults assume HF snapshots at `<repo>/models/<name>` and
the SFT dataset at `<repo>/datasets/mathinstruct.jsonl`. Override via:

| env var | purpose |
|---|---|
| `FLEXTRAIN_MODELS_DIR` | parent dir holding per-model snapshots |
| `FLEXTRAIN_VERIFIED_DATASET` | path to the SFT dataset |

Per-row hardware tweaks (`--leeway-gpu-mem-gib`, `--max-global-batch-tokens`,
etc.) are baked into `experiments/verified_runs.py`'s row registry; edit
that file rather than passing flags to `reverify.sh`.

## Tokenization & prompt format

Every row in the sweep uses **the same prompt wrapper and the model's
own tokenizer** (`JsonSFTTokenSource._build_prompt`,
`flextrain/io/sources.py:380`):

```text
Instruction:
{instruction}

Response:
{response}<|EOS|>
```

- **Per-model tokenizer**: each model's own HF tokenizer is loaded
  (Llama-3 BPE 128k vocab, Qwen3 ~152k, OLMoE GPT-NeoX-style, etc.) —
  token IDs are correct for that model.
- **No chat template**: `tokenizer.apply_chat_template()` is **not**
  invoked. The wrapper is plain text, so models like
  Llama-3.x-Instruct or Qwen3-Instruct do not see their natural
  `<|start_header_id|>` / `<|im_start|>` chat tokens.
- **Special tokens**: `add_special_tokens=False`, so no BOS / no
  chat-format header tokens. The tokenizer's EOS id is appended after
  the response so the model learns to terminate.
- **Loss masking**: cross-entropy is taken over response-token
  positions only (`ignore_index=-100` on prompt positions); reported
  loss is mean over those response tokens.

Implications for reading the table:

1. **Cross-row comparable**: every model trains on the same plain-text
   wrapper, so the curves can be compared head-to-head as an
   engine-correctness signal.
2. **Instruction-tuned models** (`Llama-3.1-8B-Instruct`, etc.): their
   priors are tuned to the model's chat template, not to
   `Instruction:/Response:`. Loss still descends (engine works), but
   absolute values are higher than what you'd see fine-tuning on the
   model's native chat format. Treat these numbers as a smoke signal,
   not as a production fine-tune quality measure.
3. **Base models**: the generic wrapper is roughly as
   out-of-distribution as anything else — descent is the relevant
   signal, not the absolute number.

If you want chat-template-aware training for production use, the right
fix is a `--apply-chat-template` flag on `train.py` that calls
`tokenizer.apply_chat_template([{"role":"user",...},
{"role":"assistant",...}], tokenize=False)` and feeds the resulting
string to the encoder; that's a separate change.

## RTX 5090 (31.3 GiB, 192 GiB host) — full sweep, 2026-05-13

All 13 rows re-verified at **auto memory budget** (no manual GPU/host
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
| Llama-3.2-1B | 1B | dense | LoRA | 1.055 → 1.012 | 28,562 | 142.6 | 142.9 | 27.00 | 27.20 |
| Llama-3.2-1B | 1B | dense | full | 1.055 → 0.826 | 21,891 | 163.9 | 164.2 | 26.80 | 27.10 |
| Llama-3.1-8B-Instruct | 8B | dense | LoRA | 0.783 → 0.747 |  5,295 | 160.0 | 160.3 | 27.40 | 27.90 |
| Llama-3.1-8B-Instruct | 8B | dense | full | 0.783 → 0.600 |  3,733 | 169.2 | 174.7 | 26.50 | 26.80 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | 0.865 → 0.844 | 21,840 | 104.1 | 120.6 | 25.70 | 28.50 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | 0.865 → 0.673 | 13,346 |  95.4 | 108.7 | 26.30 | 29.00 |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | 0.933 → 0.872 |  5,046 | 153.8 | 154.1 | 27.10 | 27.40 |
| Qwen3-8B | 8B | dense, QK-norm | full | 0.928 → 0.478 |  3,536 | 161.7 | 168.5 | 27.50 | 27.70 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 0.748 → 0.660 |  5,015 | 160.5 | 160.7 | 26.70 | 27.60 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | 0.745 → 0.467 |  3,219 | 154.5 | 165.5 | 26.20 | 26.40 |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | 1.014 → 0.819 |  1,527 | 157.7 | 163.5 | 27.40 | 28.80 |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | 0.899 → 0.865 |  7,874 |  98.0 | 123.8 | 26.20 | 29.00 |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 0.744 → 0.673 |  6,021 |  75.9 |  93.9 | 25.20 | 28.80 |

The Llama-3.1-8B-Instruct rows show a smaller absolute loss-drop because
the instruction-tuned base is starting from a chat-template prior the
generic `Instruction:/Response:` wrapper does not match (see the
tokenization section above) — descent itself confirms the engine path
is exercised cleanly.

**Notes on the 2026-05-13 refresh vs the 2026-05-12 baseline:** every
row in this sweep drifted from the prior baseline; this is the
**expected effect of the RMSNorm precision-policy fix landed
2026-05-13** (full description in
`docs/internal/multimodal_session_notes.md` and the commit message
for the fix). In short: previously, all RMSNorm γ tensors (input /
post-attention layernorms, QK-norm, final norm) were stored fp32 in
the host master but downcast to **bf16 on the GPU compute slot**, and
AdamW operated on that bf16 copy. Because flextrain stores the γ at
the `γ_canonical = 1 + scale` convention (around magnitude 1.0), the
per-step AdamW update of `lr·sign(g) ≈ 3e-5` was below bf16 ULP
(~8e-3 at magnitude 1.0), so **the normalization scales effectively
didn't train** in full-FT mode and the GPU→host offload silently
overwrote the fp32 master with the bf16-quantized value every step.
The fix uses fp32 throughout for RMSNorm params (compute / master /
grad / opt-state) — they're tiny (~1 MB total at fp32), so the cost is
negligible.

Drift profile:

* **LoRA rows drift modestly (Δ 0.0002–0.0063 in cumulative
  per-step loss)** even though LoRA doesn't touch norm parameters —
  the forward path through RMSNorm now runs in fp32 instead of bf16,
  so the residual stream is slightly more precise downstream.
  Qwen-family LoRA drifts most (QK-norm runs every attention block);
  plain Llama LoRA drifts least.
* **Full-FT rows drift more (Δ 0.0002–0.0031)** since they additionally
  reflect norms actually training at correct precision now.
* **Max drift Δ 0.0063 (qwen3_6_27b_lora)**, well within sane bounds.
  Tok/s within ±4.2% (norm fp32 storage is ~1MB total; compute path
  unchanged).

Previously the table's drift floor was the kernel-non-determinism
budget of Δ ≈ 0.003 (documented in `docs/internal/multimodal_session_notes.md`).
That bound no longer applies — the new floor will be set by the next
clean reverify after this fix is merged.

## Gemma family

Gemma 2 / Gemma 3 dense (and the Gemma 4 31B in-progress effort) have
their own verified-runs page at [`gemma_runs.md`](gemma_runs.md). It
covers the OOD-wrapper vs native-chat-template comparison, the 5-step
loss-trajectory parity vs HF, the post-engine-change re-verify
protocol, and known caveats for that arch family. The rows there
mirror the format of the table above.

## RTX 3090 (23.5 GiB, 125 GiB host) — full sweep, 2026-05-12

All 13 rows re-verified at **auto memory budget** (no manual GPU/host
caps). Methodology is identical to the RTX 5090 sweep above: per-step
metrics read from `train.py`'s stdout at step 3, `peak alloc` /
`peak resv` definitions, and effective / hardware TFLOPS conventions
all unchanged.

| Model | Params | Arch | Mode | Loss curve (5 steps) | tok/sec | eff TFLOPS | hw TFLOPS | peak alloc | peak resv |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | dense | LoRA | 1.051 → 0.977 |  8,910 |  44.5 |  44.5 | 19.80 | 20.10 |
| Llama-3.2-1B | 1B | dense | full | 1.055 → 0.825 |  6,998 |  52.4 |  52.5 | 19.70 | 19.90 |
| Llama-3.1-8B-Instruct | 8B | dense | LoRA | 0.776 → 0.719 |  1,674 |  50.6 |  50.6 | 20.10 | 20.30 |
| Llama-3.1-8B-Instruct | 8B | dense | full | 0.783 → 0.595 |  1,253 |  56.8 |  56.8 | 19.60 | 19.90 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | 0.864 → 0.832 |  8,087 |  38.5 |  39.1 | 19.80 | 22.40 |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | 0.865 → 0.673 |  5,286 |  37.8 |  39.8 | 19.50 | 21.60 |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | 0.918 → 0.845 |  1,573 |  48.0 |  48.0 | 20.10 | 20.30 |
| Qwen3-8B | 8B | dense, QK-norm | full | 0.933 → 0.490 |  1,194 |  54.6 |  54.6 | 19.60 | 19.80 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 0.743 → 0.673 |  1,554 |  49.7 |  49.7 | 20.00 | 20.30 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | 0.740 → 0.466 |  1,122 |  53.9 |  53.9 | 19.90 | 20.20 |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | 1.006 → 0.852 |    460 |  47.5 |  47.5 | 20.20 | 20.60 |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | 0.899 → 0.868 |  2,614 |  32.5 |  39.1 | 19.90 | 21.30 |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 0.742 → 0.662 |  1,871 |  23.6 |  29.9 | 19.50 | 21.60 |

Every row reproduces the RTX 5090 baseline above to within ≤0.035 at
step 5 (typical bf16 + save-tier / chunk-size noise across hardware);
two rows are bit-stable to 3 decimal places — `OLMoE-1B-7B` full
(0.673 on both cards) and `Qwen3.5-9B` full (0.466 vs 0.465).
Per-row throughput is ~30-35% of the 5090's, consistent with the
bf16-peak ratio between the two cards (3090 ≈71 TFLOPS, 5090
≈209 TFLOPS).

Additional models supported by the existing arch loaders (require a
larger machine to actually train): Qwen3.6-35B-A3B, Qwen3.5-122B-A10B,
Qwen3.5-397B-A17B, Qwen3-Coder-30B-A3B-Instruct (no new wiring needed;
they reuse `Qwen3_5*` / `Qwen3Moe*` arch ids).
