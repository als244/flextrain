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
exec python "${BASELINE_DIR}/run_baseline.py" --backend "${BACKEND}" "$@"
