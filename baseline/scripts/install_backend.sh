#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  baseline/scripts/install_backend.sh --backend BACKEND [options]

Backends:
  megatrain, torchtitan, trl_deepspeed, deepspeed_arctic, megatron

Options:
  --env-dir PATH          Virtualenv path. Default: baseline/envs/BACKEND
  --python PYTHON         Python executable for venv creation. Default: python3
  --torch-index-url URL   PyTorch wheel index. Default: https://download.pytorch.org/whl/cu126
                           Use "auto" to select from the local CUDA version.
  --torch-packages LIST   Torch packages to install before backend deps. Default: "torch torchvision torchaudio"
  --skip-torch            Do not install torch; useful for pre-provisioned envs.
  --flash MODE            Flash wheel install mode: both, fa2, fa3, none. Default: both
  --flash-version VERSION Require an exact FlashAttention wheel version.
  --linear-attention MODE Install flash-linear-attention in every env; also install
                          Qwen causal-conv1d deps where useful. auto, strict, none.
                          Default: auto
  --causal-conv1d-torch-tag TAG
                          Override causal-conv1d wheel Torch tag, e.g. torch2.10.
  --recreate              Remove the env dir before creating it.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${BASELINE_DIR}/.." && pwd)"
VENDOR_DIR="${BASELINE_DIR}/vendor"

MEGATRAIN_REPO="${MEGATRAIN_REPO:-https://github.com/DLYuanGod/MegaTrain.git}"
MEGATRAIN_REF="${MEGATRAIN_REF:-919520f07182c0d4d3e9765b2f915702e71c11a4}"
TORCHTITAN_REPO="${TORCHTITAN_REPO:-https://github.com/pytorch/torchtitan.git}"
TORCHTITAN_REF="${TORCHTITAN_REF:-}"

BACKEND=""
ENV_DIR=""
PYTHON_BIN="${PYTHON:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch torchvision torchaudio}"
SKIP_TORCH=0
FLASH_MODE="${FLASH_MODE:-both}"
FLASH_VERSION="${FLASH_VERSION:-}"
LINEAR_ATTENTION_MODE="${LINEAR_ATTENTION_MODE:-auto}"
CAUSAL_CONV1D_TORCH_TAG="${CAUSAL_CONV1D_TORCH_TAG:-}"
RECREATE=0

ensure_git_checkout() {
  local name="$1"
  local repo="$2"
  local ref="$3"
  local dest="$4"

  if [[ -d "${dest}/.git" ]]; then
    echo "Using existing ${name} checkout: ${dest}"
    return
  fi

  mkdir -p "$(dirname "${dest}")"
  rm -rf "${dest}"
  echo "Cloning ${name} from ${repo} into ${dest}"
  git clone --filter=blob:none "${repo}" "${dest}"
  if [[ -n "${ref}" ]]; then
    git -C "${dest}" fetch --depth 1 origin "${ref}" || true
    git -C "${dest}" checkout --detach "${ref}"
  fi
}

apply_git_patch_if_needed() {
  local checkout="$1"
  local patch="$2"

  if [[ ! -f "${patch}" ]]; then
    return
  fi
  if git -C "${checkout}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
    echo "Patch already applied: ${patch}"
  elif git -C "${checkout}" apply --check "${patch}" >/dev/null 2>&1; then
    git -C "${checkout}" apply "${patch}"
    echo "Applied patch: ${patch}"
  else
    echo "warning: could not apply patch ${patch}; continuing with checkout as-is" >&2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --env-dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --torch-packages)
      TORCH_PACKAGES="$2"
      shift 2
      ;;
    --skip-torch)
      SKIP_TORCH=1
      shift
      ;;
    --flash)
      FLASH_MODE="$2"
      shift 2
      ;;
    --flash-version)
      FLASH_VERSION="$2"
      shift 2
      ;;
    --linear-attention)
      LINEAR_ATTENTION_MODE="$2"
      shift 2
      ;;
    --causal-conv1d-torch-tag)
      CAUSAL_CONV1D_TORCH_TAG="$2"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${BACKEND}" ]]; then
  echo "--backend is required" >&2
  usage >&2
  exit 2
fi

case "${BACKEND}" in
  megatrain|torchtitan|trl_deepspeed|deepspeed_arctic|megatron) ;;
  *)
    echo "Unknown backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

if [[ -z "${ENV_DIR}" ]]; then
  ENV_DIR="${BASELINE_DIR}/envs/${BACKEND}"
fi

if [[ "${RECREATE}" == "1" ]]; then
  rm -rf "${ENV_DIR}"
fi

"${PYTHON_BIN}" -m venv "${ENV_DIR}"
# shellcheck source=/dev/null
source "${ENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel packaging

if [[ "${TORCH_INDEX_URL}" == "auto" ]]; then
  TORCH_INDEX_URL="$(python "${BASELINE_DIR}/scripts/detect_cuda.py" --format index-url)"
  echo "Detected local CUDA; using ${TORCH_INDEX_URL}"
fi

if [[ "${SKIP_TORCH}" == "0" ]]; then
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    python -m pip install --index-url "${TORCH_INDEX_URL}" ${TORCH_PACKAGES}
  else
    python -m pip install ${TORCH_PACKAGES}
  fi
fi

REQ_FILE="${BASELINE_DIR}/requirements/${BACKEND}.txt"
if [[ -f "${REQ_FILE}" ]]; then
  python -m pip install -r "${REQ_FILE}"
fi

case "${BACKEND}" in
  megatrain)
    MEGATRAIN_ROOT="${BASELINE_DIR}/MegaTrain"
    if [[ ! -f "${MEGATRAIN_ROOT}/setup.py" ]]; then
      MEGATRAIN_ROOT="${VENDOR_DIR}/MegaTrain"
      ensure_git_checkout "MegaTrain" "${MEGATRAIN_REPO}" "${MEGATRAIN_REF}" "${MEGATRAIN_ROOT}"
    fi
    apply_git_patch_if_needed "${MEGATRAIN_ROOT}" "${BASELINE_DIR}/patches/megatrain_cpu_master_mask.patch"
    python -m pip install -e "${MEGATRAIN_ROOT}"
    ;;
  torchtitan)
    TORCHTITAN_ROOT="${BASELINE_DIR}/TorchTitan"
    if [[ ! -d "${TORCHTITAN_ROOT}/torchtitan" ]]; then
      TORCHTITAN_ROOT="${VENDOR_DIR}/TorchTitan"
      if [[ ! -d "${TORCHTITAN_ROOT}/torchtitan" ]]; then
        ensure_git_checkout "TorchTitan" "${TORCHTITAN_REPO}" "${TORCHTITAN_REF}" "${TORCHTITAN_ROOT}"
      fi
    fi
    if [[ ! -d "${TORCHTITAN_ROOT}/torchtitan" ]]; then
      TORCHTITAN_ROOT="${REPO_ROOT}/orig/baseline/torchtitan"
    fi
    if [[ -d "${TORCHTITAN_ROOT}/torchtitan" ]]; then
      python -m pip install -e "${TORCHTITAN_ROOT}"
    else
      echo "warning: no TorchTitan checkout found; place one in baseline/TorchTitan or orig/baseline/torchtitan" >&2
    fi
    ;;
esac

case "${FLASH_MODE}" in
  both)
    FLASH_ARGS=(--package both --optional)
    if [[ -n "${FLASH_VERSION}" ]]; then
      FLASH_ARGS+=(--version "${FLASH_VERSION}")
    fi
    python "${BASELINE_DIR}/scripts/install_flash_attention_wheels.py" "${FLASH_ARGS[@]}"
    ;;
  fa2)
    FLASH_ARGS=(--package flash_attn --optional)
    if [[ -n "${FLASH_VERSION}" ]]; then
      FLASH_ARGS+=(--version "${FLASH_VERSION}")
    fi
    python "${BASELINE_DIR}/scripts/install_flash_attention_wheels.py" "${FLASH_ARGS[@]}"
    ;;
  fa3)
    FLASH_ARGS=(--package flash_attn_3 --optional)
    if [[ -n "${FLASH_VERSION}" ]]; then
      FLASH_ARGS+=(--version "${FLASH_VERSION}")
    fi
    python "${BASELINE_DIR}/scripts/install_flash_attention_wheels.py" "${FLASH_ARGS[@]}"
    ;;
  none)
    ;;
  *)
    echo "Unknown --flash mode: ${FLASH_MODE}" >&2
    exit 2
    ;;
esac

case "${LINEAR_ATTENTION_MODE}" in
  auto|strict|none) ;;
  *)
    echo "Unknown --linear-attention mode: ${LINEAR_ATTENTION_MODE}" >&2
    exit 2
    ;;
esac

if [[ "${LINEAR_ATTENTION_MODE}" != "none" ]]; then
  python -m pip install flash-linear-attention
  case "${BACKEND}" in
    megatrain|trl_deepspeed|deepspeed_arctic)
      CAUSAL_ARGS=()
      if [[ -n "${CAUSAL_CONV1D_TORCH_TAG}" ]]; then
        CAUSAL_ARGS+=(--torch-tag "${CAUSAL_CONV1D_TORCH_TAG}")
      fi
      if [[ "${LINEAR_ATTENTION_MODE}" == "auto" ]]; then
        CAUSAL_ARGS+=(--optional)
      fi
      python "${BASELINE_DIR}/scripts/install_causal_conv1d_wheel.py" "${CAUSAL_ARGS[@]}"
      if ! python -c "import causal_conv1d" >/dev/null 2>&1; then
        if python - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.__version__.startswith("2.11.") and str(torch.version.cuda).startswith("13.") else 1)
PY
        then
          FALLBACK_ARGS=(--torch-tag torch2.10)
          if [[ "${LINEAR_ATTENTION_MODE}" == "auto" ]]; then
            FALLBACK_ARGS+=(--optional)
          fi
          python "${BASELINE_DIR}/scripts/install_causal_conv1d_wheel.py" "${FALLBACK_ARGS[@]}"
        fi
      fi
      ;;
  esac
fi

cat <<EOF

Installed ${BACKEND} environment:
  ${ENV_DIR}

Activate it with:
  source ${ENV_DIR}/bin/activate

Run this backend with:
  python ${BASELINE_DIR}/run_baseline.py --backend ${BACKEND} ...
EOF
