#!/usr/bin/env bash
# Activate the conda env that owns the requested backend, run a CUDA
# preflight check, then exec run_baseline.py inside that env.
#
# Backend -> env mapping (mirrors install_backend.sh):
#   megatron                       -> baseline_megatron
#   trl_deepspeed, trl_fsdp,       -> baseline_core
#   deepspeed_arctic, megatrain,
#   torchtitan
#
# Override with BASELINE_CORE_ENV / BASELINE_MEGATRON_ENV env vars if
# you renamed the conda envs.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: baseline/scripts/run_in_backend_env.sh BACKEND [run_baseline.py args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND="$1"
shift

CORE_ENV="${BASELINE_CORE_ENV:-baseline_core}"
MEGATRON_ENV="${BASELINE_MEGATRON_ENV:-baseline_megatron}"

case "${BACKEND}" in
  megatron) ENV_NAME="${MEGATRON_ENV}" ;;
  trl_deepspeed|trl_fsdp|deepspeed_arctic|megatrain|torchtitan) ENV_NAME="${CORE_ENV}" ;;
  *)
    echo "Unknown backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

# Source conda's shell hook so `conda activate` works inside this script.
init_conda() {
  local conda_bin
  if [[ -n "${CONDA_EXE:-}" ]] && [[ -x "${CONDA_EXE}" ]]; then
    conda_bin="${CONDA_EXE}"
  elif command -v conda >/dev/null 2>&1; then
    conda_bin="$(command -v conda)"
  elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    conda_bin="${HOME}/miniconda3/bin/conda"
  elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
    conda_bin="${HOME}/anaconda3/bin/conda"
  else
    echo "error: conda not found on PATH and not at \$CONDA_EXE / ~/miniconda3 / ~/anaconda3" >&2
    echo "       install Miniconda first, or set CONDA_EXE to your conda binary." >&2
    exit 2
  fi
  # shellcheck source=/dev/null
  eval "$("${conda_bin}" shell.bash hook)"
}

init_conda

if ! conda env list | awk 'NR>2 {print $1}' | grep -qx "${ENV_NAME}"; then
  echo "error: conda env '${ENV_NAME}' not found" >&2
  echo "       run: ${BASELINE_DIR}/scripts/install_backend.sh --backend ${BACKEND}" >&2
  exit 2
fi

conda activate "${ENV_NAME}"

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
