# FlexTrain v2 — Start Here

The full design doc is in [PLAN.md](PLAN.md). This file names the order of work and where we are.

## What's being built

A production-like refactor of the AdaWS training engine currently in `orig/`. The engine semantics (paper §3) are preserved 1:1; the **layer / activation-slot / save-level seam** is rebuilt so that:

- Layers *declare* their activation tensors once (names, shape-as-fn-of-num-tokens, dtype, save-level tier).
- The engine derives `make_slot` / `size` / `send_home` / `fetch_home` automatically — no more four-parallel-code-paths per layer.
- Dense and MoE layers stop duplicating 70%+ of their attention/norm/residual logic via composable `nn/blocks/`.
- HF safetensors load/save works for Llama3, Qwen3, OLMoE, etc. (HF is config + weight source only; FlexTrain owns the compute path — patching `AutoModelForCausalLM` was considered and rejected; see PLAN.md §"Why not patch HF modules").

## Order of work

The execution plan in PLAN.md has 11 steps. We're doing them in this order:

### Phase 1 — Core abstractions (IN PROGRESS)

First, prove out the new API on ~300 LOC before porting the rest. No torch.nn compute yet, just types and contracts.

1. **`flextrain/core/activation_schema.py`** — `ActivationField`, `ActivationSchema`, `ActivationSlot`. The load-bearing abstraction.
2. **`flextrain/core/layer.py`** — `Layer` / `InputLayer` / `OutputLayer` Protocols; `ParamSpec`, `TensorSpec`, `ComputeCost`, `LayerContext`, `ChunkMeta`.
3. **`flextrain/core/save_level.py`** — `SaveLevel`, `SaveLevelPlan`, `build_dp_tables`. Wraps the existing C DP solver with per-layer `max_tier` + padding.
4. **A smoke test** that instantiates a dummy `DenseBlock`-like object, exercises the schema at each tier, and builds a DP table on it. Confirms the API holds together before we port real compute.

**Review gate.** Once Phase 1 compiles + smoke-tests, we pause. If the abstractions feel right, we move to Phase 2. If not, we revise — changes at this stage are cheap; changes after we've ported 8K LOC of `orig/` are not.

### Phase 2 — Blocks + first architecture

5. `flextrain/nn/blocks/` — norm, attention, ffn_dense, ffn_moe, rope ported from `orig/awsm_transformer/{dense_layer,moe_layer}.py`.
6. `flextrain/nn/layers/llama.py` — first architecture, template for the others.
7. `flextrain/nn/embed.py`, `flextrain/nn/head.py` — `InputLayer` / `OutputLayer` ports.

### Phase 3 — Engine port

8. `flextrain/engine/` — `active_model.py` split into orchestrator + `buffers.py` + `streams.py` + `schedule.py`. All slot-internal knowledge moves engine-side.
9. `flextrain/core/working_set.py` — port `orig/working_set.py` with a dataclass return type.
10. `flextrain/optim/` — AdamW / Muon wrappers that derive opt_state from `ParamSpec`.

### Phase 4 — I/O

11. `flextrain/io/hf_config.py` + `io/hf_weights.py` + `io/arch/<family>.py` — HF safetensors in/out.
12. `flextrain/io/checkpoint.py` — native resume format.
13. `flextrain/io/tokenstream.py` + `hf_stream.py` + `shard_stream.py` — data.

### Phase 5 — Additional architectures (iterate)

14. Qwen3-dense → OLMoE → Qwen3-MoE → Mixtral → DeepSeek-MoE → Mistral → GPT-OSS → Gemma → Phi → GLM → InternLM → Yi → Baichuan2 → Qwen3-Next. Each: one `nn/layers/<arch>.py` + one `io/arch/<arch>.py` weight-map + one `hf_config` adapter entry + one HF logit-parity smoke test.

### Phase 6 — Entrypoint

15. `cli.py` + `config.py` — `python -m flextrain train <yaml>`.

## What's explicitly NOT being built

- Multi-GPU / ZeRO / pipeline parallelism.
- Vision-language models (Qwen-VL, InternVL, LLaVA, etc.).
- Patching `AutoModelForCausalLM.forward` (would cap throughput, see PLAN.md).
- W&B / typer CLI / large test harness.
- Dashboard port (stays in `orig/` for now).

## Current status

**Phase 1, step 1 starting.** See todo list.
