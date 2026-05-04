#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: baseline/scripts/run_in_backend_env.sh BACKEND [run_baseline.py args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND="$1"
shift

ENV_DIR="${BASELINE_DIR}/envs/${BACKEND}"
if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
  echo "Missing backend env: ${ENV_DIR}" >&2
  echo "Create it with: ${BASELINE_DIR}/scripts/install_backend.sh --backend ${BACKEND}" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"

# Pre-flight: surface driver/torch CUDA mismatches before the backend
# allocates a model. ``BASELINE_SKIP_CUDA_CHECK=1`` opts out (useful in
# CI where the GPU isn't available at install time but is at run time).
# ``BASELINE_CUDA_CHECK_WARN_ONLY=1`` downgrades a hard fail to a warning.
if [[ "${BASELINE_SKIP_CUDA_CHECK:-0}" != "1" ]]; then
  CUDA_CHECK_ARGS=()
  if [[ "${BASELINE_CUDA_CHECK_WARN_ONLY:-0}" == "1" ]]; then
    CUDA_CHECK_ARGS+=(--warn-only)
  fi
  python "${BASELINE_DIR}/scripts/check_cuda_compat.py" "${CUDA_CHECK_ARGS[@]}"
fi

exec python "${BASELINE_DIR}/run_baseline.py" --backend "${BACKEND}" "$@"
