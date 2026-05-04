# 128K Max-Memory Baseline Runbook

This guide runs synthetic-token Llama 3-family training at 128K sequence length across the current baseline backends, with the most memory-saving knobs exposed by each backend. It assumes you want logs under one run folder and a CSV of per-step token throughput.

Use a Llama checkpoint whose config actually supports 128K context, such as Llama 3.1/3.3. Original Llama 3 8B checkpoints are usually 8K context unless the config has rope scaling or `max_position_embeddings >= 131072`.

## 0. Set Run Variables

Run from the repo root. The first line takes you there from anywhere inside the
checkout; the rest derive from `$PWD` so the same block works on any machine
without edits.

```bash
cd "$(git rev-parse --show-toplevel)"

export MODEL_PATH="$PWD/models/Llama-3.1-8B"
export SEQ_LEN=131072
export NUM_GPUS="$(nvidia-smi -L | wc -l)"
export NUM_STEPS=5
# Python used to seed each backend env (`python3 -m venv`). The default works
# on most machines; override only if you need a specific interpreter (e.g. a
# conda one). `command -v python3` resolves whatever `python3` is on PATH.
export PYTHON_FOR_ENVS="${PYTHON_FOR_ENVS:-$(command -v python3)}"
export PIP_CACHE_DIR="$PWD/baseline/.pip-cache"
export RUN_ROOT="$PWD/baseline/runs/llama3_128k_maxmem_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
```

Sanity-check the model context metadata:

```bash
python - <<'PY'
import json, os
c = json.load(open(os.path.join(os.environ["MODEL_PATH"], "config.json")))
print("model_type=", c.get("model_type"))
print("max_position_embeddings=", c.get("max_position_embeddings"))
print("rope_scaling=", c.get("rope_scaling"))
print("vocab_size=", c.get("vocab_size"))
PY
```

## 1. Install Independent Backend Envs

These commands create isolated envs under `baseline/envs/<backend>` and do not install or modify the main FlexTrain environment. CUDA 13.1 machines should use the cu130 torch wheel index; `--torch-index-url auto` detects that.

```bash
for backend in megatrain trl_deepspeed deepspeed_arctic torchtitan megatron; do
  baseline/scripts/install_backend.sh \
    --backend "$backend" \
    --python "$PYTHON_FOR_ENVS" \
    --torch-index-url auto \
    --flash fa2 \
    --flash-version 2.8.3
done
```

Notes:

- `--torch-index-url auto` picks the right wheel index for the local CUDA (cu130 for CUDA 13.x, cu126 for CUDA 12.6+, etc).
- The FlashAttention resolver requests an exact wheel for the env's torch + CUDA + Python + ABI; if you don't pin `--flash-version`, it picks the latest compatible one.
- `flash-linear-attention` is installed in every backend env.
- `causal-conv1d` is only relevant for HF/Qwen-hybrid paths. The installer first tries the exact `torch{X}.{Y}` wheel for the env's torch, then falls back to adjacent torch minors automatically (no `--causal-conv1d-torch-tag` override required). Pass `--causal-conv1d-torch-tag torch2.10` only if you need to pin a specific known-good wheel.

## 2. Run Dry-Runs First

This catches bad command generation before allocating model memory:

```bash
for backend in megatrain trl_deepspeed deepspeed_arctic torchtitan megatron; do
  baseline/scripts/run_in_backend_env.sh "$backend" \
    --model-path "$MODEL_PATH" \
    --seq-length "$SEQ_LEN" \
    --num-gpus "$NUM_GPUS" \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --num-steps 1 \
    --attn-implementation flash_attention_2 \
    --dry-run \
    --output-dir "$RUN_ROOT/dry_${backend}"
done
```

## 3. Run Max-Memory-Savings Configs

These commands prioritize fitting the run over speed. Start with `NUM_STEPS=1` if you are debugging a new machine, then increase it after the first successful pass.

### MegaTrain

MegaTrain is already a CPU-master/offload-style backend. Its vendor default is `checkpoint_interval=4`, meaning "checkpoint every N layers"; this is not a boolean full-recompute switch. Smaller intervals create more forward checkpoints but shorter recompute blocks, while larger intervals create fewer forward checkpoints but larger recompute caches. Use `4` for the default baseline, then sweep `1,2,4,8` if you need the tightest fit on a specific model/GPU.

```bash
baseline/scripts/run_in_backend_env.sh megatrain \
  --model-path "$MODEL_PATH" \
  --seq-length "$SEQ_LEN" \
  --num-gpus "$NUM_GPUS" \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-steps "$NUM_STEPS" \
  --attn-implementation flash_attention_2 \
  --activation-checkpoint-interval 4 \
  --num-grad-slabs 12 \
  --backend-extra-arg=--optimizer \
  --backend-extra-arg=deepspeed_cpu_adam \
  --output-dir "$RUN_ROOT/megatrain"
```

### TRL + DeepSpeed

This is the mainstream HF trainer path with ZeRO-3, CPU param/optimizer offload, full activation checkpointing, activation offload when TRL exposes it, FA2, and Liger when available.

```bash
baseline/scripts/run_in_backend_env.sh trl_deepspeed \
  --model-path "$MODEL_PATH" \
  --seq-length "$SEQ_LEN" \
  --num-gpus "$NUM_GPUS" \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-steps "$NUM_STEPS" \
  --attn-implementation flash_attention_2 \
  --zero-stage 3 \
  --param-offload cpu \
  --optimizer-offload cpu \
  --activation-checkpointing full \
  --activation-offload cpu \
  --liger-kernel auto \
  --output-dir "$RUN_ROOT/trl_deepspeed"
```

### DeepSpeed Arctic / ALST

For 128K, use sequence parallelism when running on multiple GPUs. If `NUM_GPUS=1`, set `--sequence-parallel-size 1`.

```bash
baseline/scripts/run_in_backend_env.sh deepspeed_arctic \
  --model-path "$MODEL_PATH" \
  --seq-length "$SEQ_LEN" \
  --num-gpus "$NUM_GPUS" \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-steps "$NUM_STEPS" \
  --attn-implementation flash_attention_2 \
  --zero-stage 3 \
  --param-offload cpu \
  --optimizer-offload cpu \
  --activation-checkpointing full \
  --activation-offload cpu \
  --sequence-parallel-size "$NUM_GPUS" \
  --tiled-loss-shards 16 \
  --tiled-mlp \
  --output-dir "$RUN_ROOT/deepspeed_arctic"
```

### TorchTitan

TorchTitan currently only works for model families in its registry. Llama 3 8B is covered; Qwen3.5/Qwen3.5-MoE are not. This run uses full activation checkpointing, activation CPU offload, and FSDP CPU offload.

```bash
baseline/scripts/run_in_backend_env.sh torchtitan \
  --model-path "$MODEL_PATH" \
  --seq-length "$SEQ_LEN" \
  --num-gpus "$NUM_GPUS" \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-steps "$NUM_STEPS" \
  --activation-checkpointing full \
  --activation-offload cpu \
  --param-offload cpu \
  --optimizer-offload cpu \
  --fsdp-shard-degree "$NUM_GPUS" \
  --fsdp-replicate-degree 1 \
  --output-dir "$RUN_ROOT/torchtitan"
```

### Megatron

This path uses generated model dimensions from the HF config, full recompute, TE CPU offload for activations/weights, and optimizer CPU offload.

```bash
baseline/scripts/run_in_backend_env.sh megatron \
  --model-path "$MODEL_PATH" \
  --seq-length "$SEQ_LEN" \
  --num-gpus "$NUM_GPUS" \
  --micro-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --num-steps "$NUM_STEPS" \
  --activation-checkpointing full \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --activation-offload cpu \
  --optimizer-offload cpu \
  --output-dir "$RUN_ROOT/megatron"
```

## 4. Watch Logs

Each backend writes:

- `launch_plan.json`
- generated backend configs, if any
- `run.log`

Examples:

```bash
tail -f "$RUN_ROOT/deepspeed_arctic/run.log"
rg "tokens_per_s|Throughput|tok/s|OOM|out of memory|Traceback" "$RUN_ROOT" -g run.log
```

## 5. Extract Per-Step Throughput

After the runs finish:

```bash
python baseline/scripts/extract_step_throughput.py "$RUN_ROOT"/*/run.log \
  > "$RUN_ROOT/throughput.csv"

column -s, -t < "$RUN_ROOT/throughput.csv" | less -S
```

The CSV has:

```text
backend,step,loss,step_time_s,tokens_per_s,log,line
```

## 6. Interpreting Failures

- If a backend OOMs at 128K, keep the failed `run.log`; that is still a useful maximum-context result.
- If Llama 3 fails on position length, verify that you are using a 128K-context Llama 3.1/3.3-style checkpoint, not an original 8K Llama 3 config.
- If TRL/DeepSpeed fails before training, inspect the generated DeepSpeed JSON in that backend's output directory and the top of `run.log`.
- If TorchTitan cannot infer a config, it means the model is outside the current TorchTitan registry. That is expected for Qwen3.5 and Qwen3.5-MoE.
- The proposed `hf_fsdp` backend is not implemented yet. Once added, it should be the standard HF+FSDP2 baseline for model families TorchTitan does not support.
