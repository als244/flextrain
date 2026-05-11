#!/bin/bash
# Re-run the Gemma 2 / Gemma 3 verified-runs rows and diff against the
# committed baseline. Intended as the "regression test for big engine
# changes" — after any commit that touches the engine, run this and
# confirm the loss curves are bit-identical and the throughput is
# within ±5%.
#
# Usage:
#   bash experiments/reverify_gemma.sh
#
# Honors the same FLEXTRAIN_MODELS_DIR / FLEXTRAIN_VERIFIED_DATASET
# env overrides as experiments/verified_runs.py. Exits 0 if every row
# in the rerun matches the baseline within tolerance, non-zero otherwise.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Baseline = the dir whose numbers are in docs/verified_runs.md.
# Rerun goes to a sibling dir so we never overwrite the trusted baseline.
BASELINE_DIR="${BASELINE_DIR:-runs/verified_gemma}"
RERUN_DIR="${RERUN_DIR:-runs/verified_gemma_rerun}"

# The 16 Gemma rows + 1 non-Gemma smoke test row. The non-Gemma row
# (Llama-3.2-1B LoRA) catches regressions in the shared-code paths
# (api.py, head.py, embed.py, hf_weights.py) — Gemma rows alone
# wouldn't catch a backwards-compat break for Llama / Qwen / etc.
ROWS=(
    # Smoke: one non-Gemma row to catch shared-code regressions.
    llama_3_2_1b_lora
    # Gemma 2.
    gemma2_2b_lora
    gemma2_2b_full
    gemma2_2b_lora_chat
    gemma2_2b_full_chat
    # Gemma 3.
    gemma3_1b_lora
    gemma3_1b_full
    gemma3_1b_lora_chat
    gemma3_1b_full_chat
    gemma3_4b_lora
    gemma3_4b_full
    gemma3_4b_lora_chat
    gemma3_4b_full_chat
    gemma3_12b_lora
    gemma3_12b_full
    gemma3_12b_lora_chat
    gemma3_12b_full_chat
)

PY="${PY:-./run_with_env.sh python}"

echo "[reverify] baseline = $BASELINE_DIR"
echo "[reverify] rerun    = $RERUN_DIR"
echo "[reverify] rows     = ${#ROWS[@]}"
echo

# run-grid uses one subprocess per row (with memory drain in between),
# so a single row failing doesn't take the whole grid down. Outputs
# land in $RERUN_DIR/<row>/{train.log,final.json}.
$PY experiments/verified_runs.py run-grid \
    --out "$RERUN_DIR" \
    --only "${ROWS[@]}"

echo
echo "[reverify] diff vs baseline..."
$PY experiments/verified_runs.py compare \
    --baseline "$BASELINE_DIR" \
    --rerun "$RERUN_DIR"
