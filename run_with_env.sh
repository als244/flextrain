#!/bin/bash
# Wrapper that activates the flextrain env's CUDA shim and dispatches
# whatever python command is passed. Used by the multi-chunk harnesses
# so the agent doesn't have to repeat the LD_LIBRARY_PATH dance every
# invocation.
#
# Usage: ./run_with_env.sh python tests/multi_chunk_dense_parity.py --model ...
set -euo pipefail
export CONDA_PREFIX="${CONDA_PREFIX:-/home/shein/miniconda3/envs/flextrain}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
# Replace bare 'python' with the env's python.
if [ "$1" == "python" ]; then
    shift
    exec "${CONDA_PREFIX}/bin/python" "$@"
fi
exec "$@"
