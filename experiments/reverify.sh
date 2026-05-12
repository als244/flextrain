#!/bin/bash
# Reproduce the docs/verified_runs.md table end-to-end.
#
# Shells out to ``verified_runs.py run-grid`` once per row, capturing
# per-step metrics from train.py's stdout. Writes per-row final.json +
# train.log to ``$RERUN_DIR/<row>/`` and a fresh table to
# ``$RERUN_DIR/new_table.md``. Optionally diffs the rerun against a
# trusted baseline directory.
#
# Use this before committing any change that could touch training
# numerics (kernels, optimizer, working-set solver, mem-budget logic).
#
# Usage
# -----
#   bash experiments/reverify.sh
#   bash experiments/reverify.sh --include-gemma
#   bash experiments/reverify.sh --baseline runs/verified_gemma_rerun2
#   bash experiments/reverify.sh --out runs/my_rerun
#   bash experiments/reverify.sh --include-gemma --baseline runs/verified_gemma_rerun2
#
# Flags
# -----
#   --include-gemma     Also run the 16 Gemma 2 / Gemma 3 rows from
#                       docs/gemma_runs.md. Off by default. Adds
#                       ~25 minutes (4B / 12B full FT are the long ones).
#   --baseline <dir>    After the rerun, diff against this baseline dir
#                       via ``verified_runs.py compare``. Reports any
#                       per-row loss change or throughput drift > ±5%.
#                       Skip this flag to run without comparison.
#   --out <dir>         Where to write the rerun output. Default:
#                       runs/reverify_<UTC timestamp>/.
#
# Environment overrides
# ---------------------
#   FLEXTRAIN_MODELS_DIR        Override the per-row HF-snapshot dir.
#                                (default: <repo>/models)
#   FLEXTRAIN_VERIFIED_DATASET  Override the dataset path.
#                                (default: <repo>/datasets/mathinstruct.jsonl)
#   PY                          Override the python wrapper.
#                                (default: ./run_with_env.sh python)
#
# Exit status
# -----------
#   0  every row finished cleanly. If --baseline was passed: every row
#      matches loss bit-exactly AND throughput within ±5%.
#   non-zero  any row failed to run, or any baseline diff fired.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ----------------------------------------------------------------------
# Parse args.
# ----------------------------------------------------------------------

INCLUDE_GEMMA=0
BASELINE_DIR=""
RERUN_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-gemma) INCLUDE_GEMMA=1; shift ;;
        --baseline)      BASELINE_DIR="$2"; shift 2 ;;
        --out)           RERUN_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed 's/^# \?//;$d'
            exit 0 ;;
        *) echo "[reverify] unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "$RERUN_DIR" ]]; then
    RERUN_DIR="runs/reverify_$(date -u +%Y%m%d_%H%M%S)"
fi

# ----------------------------------------------------------------------
# Row sets. Keep in sync with docs/verified_runs.md (non-Gemma table)
# and docs/gemma_runs.md (Gemma rows).
# ----------------------------------------------------------------------

# 13 rows that produce docs/verified_runs.md. Excludes Gemma; Gemma is
# in its own page and toggled in via --include-gemma below.
ROWS=(
    llama_3_2_1b_lora
    llama_3_2_1b_full
    llama_3_1_8b_lora
    llama_3_1_8b_full
    olmoe_7b_a1b_lora
    olmoe_7b_a1b_full
    qwen3_8b_lora
    qwen3_8b_full
    qwen3_5_9b_lora
    qwen3_5_9b_full
    qwen3_6_27b_lora
    qwen3_30b_a3b_lora
    qwen3_5_moe_35b_a3b_lora
)

# 16 Gemma 2 / Gemma 3 rows. Source of truth: docs/gemma_runs.md.
GEMMA_ROWS=(
    gemma2_2b_lora
    gemma2_2b_full
    gemma2_2b_lora_chat
    gemma2_2b_full_chat
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

if [[ $INCLUDE_GEMMA -eq 1 ]]; then
    ROWS+=("${GEMMA_ROWS[@]}")
fi

# ----------------------------------------------------------------------
# Run the grid + (optional) diff against the baseline.
# ----------------------------------------------------------------------

PY="${PY:-./run_with_env.sh python}"

echo "[reverify] rerun       = $RERUN_DIR"
echo "[reverify] rows        = ${#ROWS[@]} (include-gemma=$INCLUDE_GEMMA)"
if [[ -n "$BASELINE_DIR" ]]; then
    echo "[reverify] baseline    = $BASELINE_DIR"
fi
echo

$PY experiments/verified_runs.py run-grid \
    --out "$RERUN_DIR" \
    --only "${ROWS[@]}"

if [[ -n "$BASELINE_DIR" ]]; then
    echo
    echo "[reverify] diff vs $BASELINE_DIR ..."
    $PY experiments/verified_runs.py compare \
        --baseline "$BASELINE_DIR" \
        --rerun "$RERUN_DIR"
fi
