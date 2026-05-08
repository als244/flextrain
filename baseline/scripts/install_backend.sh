#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  baseline/scripts/install_backend.sh --backend BACKEND [options]

Backends and their target conda env (visible in `conda env list`):
  trl_deepspeed, trl_fsdp, deepspeed_arctic, megatrain, torchtitan
                                                  -> baseline_core
  megatron                                        -> baseline_megatron

The five "core" backends share one conda env because their pip deps
are mutually compatible. Megatron is split out because
transformer-engine pins torch tightly and historically conflicts with
the deeper deps of the HF backends.

Options:
  --env-name NAME         Conda env name. Default: baseline_core for all
                          backends except megatron, which targets
                          baseline_megatron.
  --python-version VER    Python version for `conda create`. Default: 3.12
                          (only used when the env is created from scratch).
  --torch-index-url URL   PyTorch wheel index. Default: auto (detect
                          from local CUDA via baseline/scripts/detect_cuda.py).
                          Pass an explicit URL like
                          https://download.pytorch.org/whl/cu130 to pin.
  --torch-packages LIST   Torch packages to install before backend deps.
                          Default: "torch torchvision torchaudio"
  --skip-torch            Do not install torch; useful for pre-provisioned envs.
  --flash MODE            Flash wheel install mode: both, fa2, fa3, none. Default: both
  --flash-version VERSION Require an exact FlashAttention wheel version.
  --linear-attention MODE Install flash-linear-attention; also install
                          Qwen causal-conv1d wheels where useful.
                          auto, strict, none. Default: auto
  --causal-conv1d-torch-tag TAG
                          Pin a specific causal-conv1d wheel torch tag
                          (disables minor-version probing).
  --recreate              Drop the conda env (`conda env remove`) before
                          recreating it.
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
ENV_NAME=""
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-auto}"
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

# Source conda's shell hook so `conda activate` works inside this script.
# Looks at $CONDA_EXE first (set by conda activate), then PATH, then a
# couple of common install locations.
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

env_exists() {
  conda env list | awk 'NR>2 {print $1}' | grep -qx "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
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
  megatrain|torchtitan|trl_deepspeed|deepspeed_arctic|megatron|trl_fsdp) ;;
  *)
    echo "Unknown backend: ${BACKEND}" >&2
    exit 2
    ;;
esac

# =============================================================
# Decide which conda env owns this backend.
# =============================================================
# All backends except megatron share one env (baseline_core) because
# their pip deps are mutually compatible. Megatron is split into its
# own env (baseline_megatron) because transformer-engine pins torch
# tightly and historically conflicts with the HF backends' deeper
# deps (deepspeed, kernels, tvm-ffi, etc).
#
# Override either with --env-name NAME (or set the
# BASELINE_CORE_ENV / BASELINE_MEGATRON_ENV env vars).
if [[ -z "${ENV_NAME}" ]]; then
  case "${BACKEND}" in
    megatron) ENV_NAME="${BASELINE_MEGATRON_ENV:-baseline_megatron}" ;;
    *)        ENV_NAME="${BASELINE_CORE_ENV:-baseline_core}" ;;
  esac
fi

# Map env -> consolidated requirements file. The five core backends
# share baseline_core.txt; megatron has its own.
case "${ENV_NAME}" in
  baseline_core)     REQ_FILE="${BASELINE_DIR}/requirements/baseline_core.txt" ;;
  baseline_megatron) REQ_FILE="${BASELINE_DIR}/requirements/baseline_megatron.txt" ;;
  *)                 REQ_FILE="${BASELINE_DIR}/requirements/${ENV_NAME}.txt" ;;
esac

# =============================================================
# Banner: surface the env decision before any install work.
# =============================================================
cat <<BANNER

================================================================
[install_backend] backend  : ${BACKEND}
                  conda env: ${ENV_NAME}    <-- visible in \`conda env list\`
                  reqs file: $(realpath --relative-to="${REPO_ROOT}" "${REQ_FILE}" 2>/dev/null || echo "${REQ_FILE}")
================================================================

BANNER

# =============================================================
# Create or reuse the conda env.
# =============================================================
init_conda

if [[ "${RECREATE}" == "1" ]] && env_exists "${ENV_NAME}"; then
  echo "[install_backend] --recreate: removing existing conda env '${ENV_NAME}'"
  conda env remove -n "${ENV_NAME}" -y
fi

if env_exists "${ENV_NAME}"; then
  echo "[install_backend] reusing existing conda env '${ENV_NAME}' (no create)"
else
  echo "[install_backend] creating new conda env '${ENV_NAME}' with python=${PYTHON_VERSION}"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"
echo "[install_backend] activated conda env '${ENV_NAME}' at ${CONDA_PREFIX}"

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

if [[ -f "${REQ_FILE}" ]]; then
  python -m pip install -r "${REQ_FILE}"
else
  echo "warning: no requirements file at ${REQ_FILE}; skipping pip install -r" >&2
fi

# Editable installs of vendor checkouts (megatrain + torchtitan).
# Only run for the core env since megatron has its own world.
if [[ "${ENV_NAME}" == "baseline_core" ]]; then
  if [[ "${BACKEND}" == "megatrain" ]] || [[ -d "${BASELINE_DIR}/MegaTrain" ]] || [[ -d "${VENDOR_DIR}/MegaTrain" ]]; then
    MEGATRAIN_ROOT="${BASELINE_DIR}/MegaTrain"
    if [[ ! -f "${MEGATRAIN_ROOT}/setup.py" ]]; then
      MEGATRAIN_ROOT="${VENDOR_DIR}/MegaTrain"
      if [[ ! -f "${MEGATRAIN_ROOT}/setup.py" ]]; then
        ensure_git_checkout "MegaTrain" "${MEGATRAIN_REPO}" "${MEGATRAIN_REF}" "${MEGATRAIN_ROOT}"
      fi
    fi
    python -m pip install -e "${MEGATRAIN_ROOT}"
  fi
  if [[ "${BACKEND}" == "torchtitan" ]] || [[ -d "${BASELINE_DIR}/TorchTitan/torchtitan" ]] || [[ -d "${VENDOR_DIR}/TorchTitan/torchtitan" ]]; then
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
      echo "warning: no TorchTitan checkout found; the torchtitan backend will not be importable in ${ENV_NAME}" >&2
    fi
  fi
fi

case "${FLASH_MODE}" in
  both)
    FLASH_ARGS=(--package both --optional)
    if [[ -n "${FLASH_VERSION}" ]]; then
      FLASH_ARGS+=(--version "${FLASH_VERSION}")
    fi
    python "${BASELINE_DIR}/../helpers/install_flash_attn_wheels.py" "${FLASH_ARGS[@]}"
    ;;
  fa2)
    FLASH_ARGS=(--package flash_attn --optional)
    if [[ -n "${FLASH_VERSION}" ]]; then
      FLASH_ARGS+=(--version "${FLASH_VERSION}")
    fi
    python "${BASELINE_DIR}/../helpers/install_flash_attn_wheels.py" "${FLASH_ARGS[@]}"
    ;;
  fa3)
    FLASH_ARGS=(--package flash_attn_3 --optional)
    if [[ -n "${FLASH_VERSION}" ]]; then
      FLASH_ARGS+=(--version "${FLASH_VERSION}")
    fi
    python "${BASELINE_DIR}/../helpers/install_flash_attn_wheels.py" "${FLASH_ARGS[@]}"
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
  # causal-conv1d only matters for HF / Qwen-hybrid paths (not megatron).
  if [[ "${ENV_NAME}" == "baseline_core" ]]; then
    # The wheel installer probes the detected torch tag first and then
    # walks back through earlier torch minors (default 2) to find a
    # prebuilt wheel. Override with --causal-conv1d-torch-tag only if
    # you need to pin a specific known-good wheel.
    CAUSAL_ARGS=()
    if [[ -n "${CAUSAL_CONV1D_TORCH_TAG}" ]]; then
      CAUSAL_ARGS+=(--torch-tag "${CAUSAL_CONV1D_TORCH_TAG}")
    fi
    if [[ "${LINEAR_ATTENTION_MODE}" == "auto" ]]; then
      CAUSAL_ARGS+=(--optional)
    fi
    python "${BASELINE_DIR}/../helpers/install_causal_conv1d_wheel.py" "${CAUSAL_ARGS[@]}"
  fi
fi

# Pre-fetch HF kernel-hub kernels (sonic-moe today) into the local
# HF cache. The compute node may have no internet, in which case the
# kernels' lazy-load path falls back to whatever it finds in
# ~/.cache/huggingface/. Skip when the caller is already offline (the
# fetch would just fail) or when the user explicitly opts out via
# BASELINE_SKIP_KERNEL_PREFETCH=1. Errors are non-fatal; the runtime
# fallback for missing kernels is to use --moe-kernel-backend hf.
if [[ "${ENV_NAME}" == "baseline_core" ]] && \
   [[ "${HF_HUB_OFFLINE:-0}" != "1" ]] && \
   [[ "${BASELINE_SKIP_KERNEL_PREFETCH:-0}" != "1" ]]; then
  python "${BASELINE_DIR}/scripts/prefetch_kernels.py" || \
    echo "warning: kernel prefetch failed; sonic-moe path may need internet at first run" >&2
fi

cat <<EOF

Installed backend "${BACKEND}" into conda env: ${ENV_NAME}
Visible in: conda env list

Activate it with:
  conda activate ${ENV_NAME}

Run this backend with:
  conda activate ${ENV_NAME}
  python ${BASELINE_DIR}/run_baseline.py --backend ${BACKEND} ...

Or use the dispatcher (auto-activates the right env per backend):
  ${BASELINE_DIR}/scripts/run_in_backend_env.sh ${BACKEND} ...
EOF
