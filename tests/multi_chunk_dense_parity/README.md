# Multi-chunk parity harness

Verifies FT's forward path is correct when one sequence spans multiple
chunks. Used as

* the regression baseline for the dense-attn KV-context-window machinery
  (`flextrain/engine/active_model.py:_update_fwd_context`), and
* the regression test for the linear-attn cross-chunk-state work
  (Item 3c — `LinearAttnStateBank` + FLA `initial_state` plumbing).

## Layout

* `hf_capture.py`  – HF forward → save logits to `.pt` bundle
* `ft_replay.py`   – Load bundle → FT multi-chunk fwd → save FT logits
* `compare.py`     – Stream-diff HF vs FT, write JSON stats
* `run_e2e.py`     – Driver: spawns the three above as subprocesses

Splitting fwd-then-compare across processes keeps GPU peak at one
model's footprint at a time. Important on a 24 GB card: HF and FT
logit tensors at 32k tokens × 248k vocab are 8–15 GiB each.

## Fixture

`tests/fixtures/long_real_sample.txt` — record 81 from LongBench-v2,
a long-form prose passage. Tokenizes between 28k and 36k tokens for
the Llama-3, Qwen3, and Qwen3.5 tokenizers, so the same fixture
exercises all three model families at the multi-chunk threshold.

## Running

The fastest path: use the e2e driver.

```bash
# Activate flextrain env (or set LD_LIBRARY_PATH for cu12 shim)
PY=/home/shein/miniconda3/envs/flextrain/bin/python

# Default: runs Llama-3.2-1B, Qwen3-1.7B, Qwen3.5-2B at chunk=8192
$PY tests/multi_chunk_dense_parity/run_e2e.py

# Single model:
$PY tests/multi_chunk_dense_parity/run_e2e.py --model models/Qwen3.5-2B

# Keep the .pt bundles around for re-running compare later:
$PY tests/multi_chunk_dense_parity/run_e2e.py --keep-bundles
```

The driver auto-cleans `.pt` bundles by default since each is 7–15 GiB.
Per-model JSON stats land at `tests/multi_chunk_logs/<model>__chunk<N>.json`.

## Acceptance bar (compare.py)

A run passes iff ALL of the following hold:

* `rel = max|Δ|/‖HF‖` < `5e-2` (overall logit drift)
* overall `argmax_agreement` ≥ `0.95`
* per-chunk `argmax_agreement` ≥ `0.93` for every chunk
* per-chunk `argmax_agreement` spread (max − min) ≤ `0.05`

The per-chunk uniformity check is the real localization signal. A
chunk-boundary regression typically shows up as the first chunk
passing (no prior context) while continuation chunks degrade by 10%+
argmax — overall metrics wash that out.

The harness also reports per-chunk **CE delta** (FT next-token CE vs
HF next-token CE on the same input). `ΔCE > 0.01` on continuation
chunks is a clear bug signal even when argmax stays nearly aligned,
because CE is sensitive to logit shifts that don't cross the argmax
threshold.

## Stage 3a baselines (RTX 3090, master @ 2026-04-30)

Dense-attention only: cross-chunk parity is correct as expected.

| Model         | T     | rel       | argmax_agree | per-chunk argmax (4 chunks)              | result |
|---------------|-------|-----------|--------------|------------------------------------------|--------|
| Llama-3.2-1B  | 31530 | 1.80e-05  | 97.94%       | 98.23 / 97.85 / 97.81 / 97.84            | **PASS** |
| Qwen3-1.7B    | 31629 | 2.15e-05  | 97.79%       | 97.97 / 98.17 / 97.44 / 97.53            | **PASS** |

## Stage 3b baseline (failing — to-be-fixed by Stage 3c)

Hybrid (linear-attn + dense): chunk 0 (no prior context) is correct,
continuation chunks degrade.

| Model         | T     | rel      | argmax_agree | per-chunk argmax              | per-chunk ΔCE                | result |
|---------------|-------|----------|--------------|-------------------------------|------------------------------|--------|
| Qwen3.5-2B    | 32000 | 7.97e-05 | 89.39%       | 98.38 / 87.99 / 85.78 / 84.99 | -0.001 / +0.035 / +0.041 / +0.039 | **FAIL** |

The 13.4% spread between chunk 0 and chunk 3 is the smoking gun.
ΔCE jumping from −0.001 (chunk 0) to +0.04 (chunks 1–3) confirms
linear-attn recurrent state is being thrown away at chunk boundaries.

## Stage 3d (after fix lands)

Re-run the full e2e driver. Expected:

* Llama-3.2-1B → PASS unchanged (no linear-attn layers)
* Qwen3-1.7B   → PASS unchanged (no linear-attn layers)
* Qwen3.5-2B   → flips to PASS, with per-chunk argmax >= 95% uniformly

Anything weaker on Qwen3.5-2B means the cross-chunk state plumbing
is incomplete or the bank lifecycle is wrong.

## TODO (future stages)

* **Backward parity**: same harness pattern but compare FT vs HF
  gradients (`hf_capture` saves grads, `ft_replay` runs `_backward_pass`,
  `compare` diffs `dW`). Needed because forward parity does not
  guarantee bwd parity — FLA's `dh0/dht` symmetry has to land for
  multi-chunk bwd to work.

* **Multi-step parity**: 3–5 step training loop with the same long
  sequence; compare loss curves. The most stringent test — catches
  any drift in optimizer-state interaction with cross-chunk activations.
