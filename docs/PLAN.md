# Plan: FlexTrain v2 — Production-Like AdaWS Training Engine

## Context

`orig/` is a research prototype of **AdaWS** (Adaptive Working Set Strategy) — a single-GPU Transformer training engine from a NeurIPS paper that co-optimizes data sizing, training-state offload, and activation offload/recompute under tight GPU memory budgets. The core engine works and is algorithmically correct. Pain points:

- **The "layer" abstraction is loose.** `TransformerLayer` (dense, 1076 LOC) and `TransformerMoELayer` (MoE, 1637 LOC) duplicate ~70% of their attention/norm/residual logic. There is no base class — the scheduler duck-types ten methods and two attributes.
- **Activation slots are dicts that four parallel code paths must keep in lockstep** ([dense_layer.py:745](orig/awsm_transformer/dense_layer.py#L745) `make_act_slot`, [:837](orig/awsm_transformer/dense_layer.py#L837) `get_act_slot_size`, [:918](orig/awsm_transformer/dense_layer.py#L918) `send_activations_home`, [:952](orig/awsm_transformer/dense_layer.py#L952) `fetch_activations`, plus per-level FLOP accounting in `get_fwd_flops`). Adding an activation tensor means touching 5+ locations, all of which must agree on names, shapes, dtypes, offsets, and save-level tiers. The MoE file has an explicit TODO: *"clean this up and have systematic way of handling act slots!!!!"*
- **Embed and head are special-cased outside the main loop** ([active_model.py:1233](orig/active_model.py#L1233), [:1383](orig/active_model.py#L1383)) with slightly-different interfaces.
- **Not wired to real workflows.** Random init only, custom `torch.save` checkpoint format per layer, no HuggingFace support, no checkpoint resume. `train.py` uses hardcoded globals; `bench_train.py` has argparse but is a benchmark driver. Data pipeline (`fineweb.py`, `sequence_pool.py`) consumes pre-tokenized `.bin` shards only.

**User's answered scope** (out-of-band):
- **Layout**: single pip-installable `flextrain/` package.
- **Weights**: load HF safetensors, save/resume native FlexTrain checkpoints, export back to HF.
- **Data**: support both HF `datasets` + tokenizers AND the existing pre-tokenized `.bin` shards.
- **Engine scope**: preserve single-GPU AdaWS functionality 1:1, refactor the Layer / ActivationSlot / SaveLevel seam.
- **Architecture breadth**: match MegaTrain-style coverage (Llama 2/3/3.1/3.2, Qwen2/2.5/3/3-Next, Mistral, Mixtral, DeepSeek, Phi-3/4, Gemma 2/3, GLM-4/4.5, InternLM, Yi, Baichuan2, OLMoE, GPT-OSS) — but, critically, **not by wrapping `AutoModelForCausalLM`'s forward**. MegaTrain does that and pays a 60% throughput ceiling (paper §3.4, "MegaTrain … limiting throughput to 60% vs. no recomputation") because HF forward passes can't do tensor-level save-level policy. FlexTrain owns the compute path.
- **Explicitly out of scope**: distributed / multi-GPU, production hygiene extras (W&B, typer CLI, extensive test infra).

## Goals

1. Preserve AdaWS algorithmic semantics — schedule, data sizing (§3.2), memory partitioning (§3.3), save-level DP (§3.4) — with no throughput regression.
2. Collapse dense/MoE duplication via composable blocks; layers declare their activation tensors **once** and the engine derives make/size/send/fetch.
3. Make `EmbedLayer` and `OutputHead` first-class under the same contract (with minor protocol variants).
4. Load HF safetensors checkpoints for Llama3-8B, Qwen3, OLMoE; save/resume FlexTrain checkpoints; export to HF format.
5. Plug in HF `datasets` + `AutoTokenizer`, keep existing `.bin` shard loader behind a common `TokenStream` protocol.

## Target layout (`flextrain/` package)

```
flextrain/
├── core/                         # Engine primitives — no torch.nn dependence
│   ├── activation_schema.py      # ActivationField, ActivationSchema, ActivationSlot
│   ├── layer.py                  # Layer / InputLayer / OutputLayer Protocols; ParamSpec, ComputeCost, LayerContext
│   ├── save_level.py             # SaveLevel, SaveLevelPlan, build_dp_tables
│   └── working_set.py            # determine_working_set_config (from orig/working_set.py, cleaned)
├── engine/
│   ├── active_model.py           # Orchestrator (slimmed from orig/active_model.py — no layer-internal knowledge)
│   ├── buffers.py                # GPU/host buffer managers (params, grads, activation ring, transitions)
│   ├── streams.py                # CUDA stream + event bookkeeping (extracted from ActiveModel)
│   └── schedule.py               # Per-(chunk, layer) pre/post-action dispatch (Table 2 of paper)
├── nn/
│   ├── blocks/                   # Composable shared pieces
│   │   ├── norm.py               # NormBlock (RMSNorm fwd/bwd + ActivationField declarations)
│   │   ├── attention.py          # AttentionBlock (GQA + RoPE + flash-attn bindings)
│   │   ├── ffn_dense.py          # DenseFFN (SwiGLU)
│   │   └── ffn_moe.py            # MoEFFN (router + scatter/gather; per-expert views are internal)
│   ├── layers/
│   │   ├── dense.py              # DenseTransformerBlock = attn_norm + attention + ffn_norm + DenseFFN
│   │   └── moe.py                # MoETransformerBlock   = attn_norm + attention + ffn_norm + MoEFFN
│   ├── embed.py                  # TokenEmbedLayer : InputLayer
│   └── head.py                   # LMHead : OutputLayer (chunked CE loss, fwd+bwd fused)
├── ops/                          # Kept verbatim from orig/awsm_transformer/ops — not the refactor target
├── io/
│   ├── hf_weights.py             # Load safetensors; per-architecture name mapping (Llama3, Qwen3, OLMoE); export back
│   ├── checkpoint.py             # Native FlexTrain checkpoint (safetensors + meta json)
│   ├── tokenstream.py            # TokenStream protocol: yields pre-tokenized chunks
│   ├── hf_stream.py              # HF datasets + AutoTokenizer implementation
│   └── shard_stream.py           # Existing .bin shard reader (ported from orig/fineweb.py + sequence_pool.py)
├── optim/
│   ├── adamw.py                  # Wraps awsm_adamw_step; derives opt_state from ParamSpec
│   └── muon.py                   # Wraps awsm_muon_step
├── hw/
│   ├── env.py                    # Cleaned hardware_env.py (bench_matmul/bench_transfer kept inside)
│   └── query_memory.py           # Ported as-is
├── cli.py                        # Single entrypoint: `python -m flextrain train <config.yaml>`
└── config.py                     # Dataclass configs: ModelConfig, TrainConfig, HardwareConfig
```

Deliberately omitted: `dashboard/` (keep in orig until requested), multi-GPU hooks, W&B, tests beyond a numerical-parity smoke test.

## Key abstractions (the core refactor)

### `ActivationField` / `ActivationSchema` / `ActivationSlot`

An `ActivationField` declaratively describes one activation tensor:

```python
@dataclass(frozen=True)
class ActivationField:
    name: str
    shape_fn: Callable[[int, Mapping[str,int]], tuple[int,...]]  # (num_tokens, dims) -> shape
    dtype: torch.dtype
    tier: int                # 0 = always saved; N = saved only at level >= N
    offload: bool = True     # if False, device-only, never in home slot
    persist: bool = True     # if False, engine-owned scratch (e.g. MoE x_up outer buffer)
    token_axis: int = 0      # which axis is num_tokens (1 for softmax_lse)
```

An `ActivationSchema` is a tuple of `ActivationField`s plus `max_tier`. From the schema, the engine **derives automatically**:
- `home_size_bytes(num_tokens, dims, level)` → replaces `get_act_slot_size`.
- Host-buffer slicing → replaces the buffer-offset-arithmetic branch of `make_act_slot`.
- `send_home(slot, level)` / `fetch_home(slot, level)` → replaces hand-maintained `save_level_mapping` dicts.
- Chunk-view re-slicing (including the `softmax_lse` transpose case) via `token_axis`.

`ActivationSlot` is a typed view (`slot.x_inp`, `slot.has("xq")`). Forward-recompute checks `slot.has(name)` instead of `key in dict` introspection.

**MoE resolution**: the outer `x_up` buffer is declared with shape `(num_tokens * top_k, 2 * expert_dim)` — deterministic in `num_tokens`, so it fits the schema. Per-expert `dict[expert_id, Tensor]` views are built inside `MoEFFN.forward` at runtime using `expert_counts` (which itself is declared `offload=False`). Router metadata (`x_router`, `chosen_experts`, `scattered_router_weights`) is declared `offload=False` — device scratch only, never home.

### `Layer` Protocol + composable `Block` pieces

```python
@runtime_checkable
class Layer(Protocol):
    layer_id: int
    schema: ActivationSchema
    param_spec: ParamSpec
    def forward(self, x, chunk_meta, weights, slot, ctx) -> torch.Tensor: ...
    def forward_recompute(self, slot, chunk_meta, weights, ctx) -> None: ...
    def backward(self, dx, chunk_meta, weights, grads, slot, ctx) -> torch.Tensor: ...
    def compute_cost(self, chunk_meta) -> ComputeCost: ...
```

What's **no longer a layer concern** (moves to engine):
`make_act_slot`, `get_act_slot_size`, `send_activations_home`, `fetch_activations`, `get_fwd_flops` per-level dict, `fetch_weights`, `create`/`load`/`save`/`step` (all derived from `ParamSpec` + optimizer), chunk-view re-slicing, ephemeral workspace (`LayerContext.scratch(shape, dtype)`).

`DenseTransformerBlock` becomes composition:
```python
class DenseTransformerBlock:
    def __init__(self, layer_id, dims, hp, optimizer):
        self.attn_norm = NormBlock("attn_norm")
        self.attn      = AttentionBlock(dims, hp)   # declares xk/xv(tier0), attn_result/lse(tier1), xq/xo(tier2)
        self.ffn_norm  = NormBlock("ffn_norm")
        self.ffn       = DenseFFN(dims)             # declares x1/x3 at tier=3
        self.schema    = ActivationSchema.concat([b.fields for b in self._blocks], max_tier=3)
        self.param_spec = ParamSpec.merge([b.param_spec for b in self._blocks])
```

`MoETransformerBlock` is identical except `self.ffn = MoEFFN(dims)`. The 1637-vs-1076 LOC delta collapses to one swapped block.

### `SaveLevel` — per-layer tiers, padded DP array

Keep the 4-level enum semantics used by the paper and C DP solver, but make `max_tier` per-layer and pad the DP `(T, k)` array to `max_k_across_layers` with `-inf` values. This avoids [active_model.py:567](orig/active_model.py#L567)'s assumption that all layers share `max_saved_activations_level`. `build_dp_tables(layers, chunks, hw) → DPTables` wraps the current inline computation; solver interface unchanged. Special `-1` "GPU-resident, no home slot" becomes `SaveLevel(-1)`, not a dict-value sentinel.

### `InputLayer` / `OutputLayer` Protocols (embed / head)

Two extra Protocols share `schema` + `param_spec` + `compute_cost` with `Layer` but differ on the compute method:
- `InputLayer.forward(chunk_meta, weights, ctx) -> Tensor` (takes token_ids, not x; no activation slot).
- `OutputLayer.forward_backward(x, labels, chunk_meta, weights, grads, ctx, loss_scale) -> (Tensor, LossStats)` — fuses the chunked-CE loss that never materializes full `(tokens, vocab)` logits ([head.py:110](orig/awsm_transformer/head.py#L110)).

Engine loop becomes: `embed → for layer in backbone → head`, with `create`/`load`/`save`/`step` uniformly derived from `param_spec`.

## Multi-architecture strategy (Llama / Qwen / MoE / GPT-OSS / hybrids)

**Decision: own the forward path, treat HF as config + weight source only.** Patching `AutoModelForCausalLM.forward` looks attractive (free menagerie) but breaks on the first AdaWS requirement — per-(chunk, layer) tensor-level save-level selection requires us to call specific ops in a specific order and decide at runtime which intermediates to materialize. HF's `LlamaDecoderLayer.forward` doesn't expose those hooks; we'd end up reimplementing it anyway. MegaTrain proves this out: they use HF forward unmodified and hit a 60% throughput ceiling because of it.

Instead, architectures share most of their compute — GQA attention + RoPE + RMSNorm + SwiGLU FFN is Llama, Qwen-dense, Mistral, Gemma (with one norm variant), InternLM, Yi, Baichuan2, Phi-3. We express architectures as **compositions of shared blocks**, with ~3 axes of variation:

### Block variation axes

- **Attention**: standard GQA (Llama3/Qwen-dense/Mistral/OLMoE), sliding-window GQA (Mistral, GPT-OSS), hybrid linear + full attention (Qwen3-Next). Implemented as `AttentionBlock` variants sharing the same `schema` fields (`xk/xv/attn_result/lse/xq/xo`) with per-instance `window_size`, linear-variant subclass adds its own state fields.
- **FFN**: `DenseFFN` (SwiGLU, Llama-family), `GLUGatedFFN` (Gemma uses `gate_proj(x) * up_proj(x)` with GELU), `MoEFFN` (OLMoE, Qwen3-MoE, Mixtral, DeepSeek-MoE — differ only in top-k, num-experts, router type, shared-expert-count), `HybridMoE` (GPT-OSS has shared expert + routed, similar to DeepSeekV3).
- **Norm**: RMSNorm (Llama/Qwen/Mistral), LayerNorm (older variants), QK-norm (Qwen3, Gemma3 — extra `q_norm` / `k_norm` before RoPE, adds `xq_norm_rstd`, `xk_norm_rstd` fields at tier=0).

### Concrete class hierarchy (in `flextrain/nn/`)

```
nn/blocks/
├── attention.py       GQAAttentionBlock (base)
│                      ├── SlidingWindowAttentionBlock (Mistral, GPT-OSS layers w/ SWA)
│                      └── LinearAttentionBlock (Qwen3-Next linear layers)
├── norm.py            RMSNormBlock, LayerNormBlock, GemmaRMSNormBlock (+1 quirk)
├── ffn_dense.py       SwiGLUFFN, GeluGatedFFN (Gemma), GatedFFN (generic)
├── ffn_moe.py         MoEFFN (parametric on top_k, n_experts, n_shared, router_type)
└── rope.py            RoPE variants: standard, YARN (Llama3.1 long-ctx), M-RoPE (Qwen2-VL),
                       NoPE (some Phi variants)

nn/layers/
├── llama.py           LlamaBlock: GQAAttn + RMSNorm + SwiGLUFFN + rope-yarn-or-standard
├── qwen.py            QwenDenseBlock (QK-norm variant), QwenMoEBlock
├── mistral.py         MistralBlock (SWA + SwiGLUFFN)
├── mixtral.py         MixtralBlock (SWA + MoEFFN)
├── gemma.py           GemmaBlock (post-norm quirk, GELU-gated FFN, norm scaling)
├── deepseek.py        DeepSeekMoEBlock (shared experts, MLA attention optional)
├── olmoe.py           OLMoEBlock (~= QwenMoEBlock minus QK-norm)
├── phi.py             Phi3Block, Phi4Block
├── gpt_oss.py         GptOssBlock (hybrid dense+MoE layers, SWA subset)
├── qwen_next.py       Qwen3NextBlock (linear + full attention layer types)
└── glm.py / internlm.py / yi.py / baichuan.py  (mostly Llama-clones + renames)
```

Each architecture block is 30–80 LOC of composition — no duplicated fwd/bwd kernels. Heterogeneous models (GPT-OSS dense+MoE alternation, Qwen3-Next linear+full) use **mixed-layer-type backbones**: `backbone = [LayerAT_0, LayerBT_1, LayerAT_2, ...]`. The engine already treats `self.model_layers` as a dict of per-layer objects ([active_model.py:1261](orig/active_model.py#L1261)), so mixing types Just Works — but only after the per-layer `max_tier` + DP-padding change in section C above. That's the main reason the `SaveLevel` design went per-layer instead of global.

### HF integration: config + weights, not compute

Two small, focused surfaces:

1. **`io/hf_config.py`** — `hf_config_to_flextrain(hf_config: PretrainedConfig) → ModelConfig`. One per architecture family (~15 adapters covering the MegaTrain list). Dispatches on `hf_config.model_type`. Outputs the same dataclass that `model_dims.json` entries parse into — so you can either hand-write a FlexTrain config or derive one from any HF checkpoint.

2. **`io/hf_weights.py`** — `load_hf_safetensors(hf_path: str, model: ActiveModel) → None` and `export_to_hf(model, out_dir)`. Per-architecture **name mapping**: for each FlexTrain `ParamSpec` field, the HF tensor name to copy from (e.g. `w_q` ← `model.layers.{i}.self_attn.q_proj.weight`, transpose). These mapping tables are the only architecture-specific I/O code — each is ~20 lines.

```python
# flextrain/io/arch/llama.py
LLAMA_WEIGHT_MAP = {
    "w_attn_norm":    "model.layers.{i}.input_layernorm.weight",
    "w_q":            ("model.layers.{i}.self_attn.q_proj.weight", "T"),   # transpose
    "w_k":            ("model.layers.{i}.self_attn.k_proj.weight", "T"),
    "w_v":            ("model.layers.{i}.self_attn.v_proj.weight", "T"),
    "w_o":            ("model.layers.{i}.self_attn.o_proj.weight", "T"),
    "w_ffn_norm":     "model.layers.{i}.post_attention_layernorm.weight",
    "w_1":            ("model.layers.{i}.mlp.gate_proj.weight", "T"),
    "w_2":            ("model.layers.{i}.mlp.down_proj.weight", "T"),
    "w_3":            ("model.layers.{i}.mlp.up_proj.weight", "T"),
}
EMBED_MAP = {"w_tok_embeddings": "model.embed_tokens.weight"}
HEAD_MAP  = {"w_final_norm":     "model.norm.weight",
             "w_head_proj":      ("lm_head.weight", "T")}
```

`AutoModelForCausalLM` is used for the **tokenizer** (via `AutoTokenizer.from_pretrained`) and for **reading `config.json`** via `AutoConfig` — not for instantiating torch modules. We pay tokenizer/config cost from HF and gain our own compute path.

### Why not patch HF modules

We considered the "monkey-patch `LlamaDecoderLayer.forward`" route. Three blockers:
1. **Forward signature mismatch.** HF layer forwards take `hidden_states`, return `(hidden_states, past_key_value, attention_output)`. Our `forward` takes a typed `ActivationSlot` and `LayerContext`, writes into slot tensors, and returns the residual-added output. We'd have to rewrite forward regardless.
2. **No hook for per-tensor save-level decisions.** HF uses `torch.autograd.Function` + `checkpoint()` — all-or-nothing block-level. AdaWS needs per-(Q/K/V/attn_out/x1/x3) tier control.
3. **KV cache semantics fight our context window.** HF expects a `past_key_value` object; our `KVContextWindow` is a chunk-sweeping ring. Different objects, different lifetimes.

Net: the right split is **FlexTrain owns nn compute, HF owns config/tokenizer/weight storage.**

### Architecture coverage: implementation order

1. **Llama3** (8B, 70B) — simplest GQA + RMSNorm + SwiGLU. First arch, sets the composition template.
2. **Qwen3-dense** (0.6B..32B) — adds QK-norm. One extra `NormBlock` pair per attention, `.fields` gain `xq_norm_rstd`, `xk_norm_rstd` at tier=0.
3. **OLMoE** (7Bx1B) — first MoE; validates `MoEFFN` with router metadata declared `offload=False` and `x_up` at tier=3.
4. **Qwen3-MoE** (30Bx3B, 235B) — MoE + QK-norm composition.
5. **Mixtral** / **DeepSeek-MoE** — shared experts, alternate routing.
6. **Mistral** / **GPT-OSS** — sliding-window attention variant.
7. **Gemma** / **Phi** / **GLM** / **InternLM** / **Yi** / **Baichuan2** — mostly Llama-clones + small norm/scaling quirks.
8. **Qwen3-Next** — heterogeneous linear+full attention layers.

Each step is ~1 day: one `nn/layers/<arch>.py` (30–80 LOC), one `io/arch/<arch>.py` weight map (20–40 LOC), one HF-config adapter entry (~15 LOC), one numerical-parity smoke test (see Verification).

### Vision-language models

**Explicitly out of scope for v1.** MegaTrain claims Qwen-VL / InternVL / LLaVA / Gemma 3 VL etc., but VLMs add a vision tower (ViT, MLP projector), variable-length image sequences, cross-modal attention, and M-RoPE. That's a whole separate design axis on top of AdaWS. Defer to v2.

## I/O layer (HF + checkpoints + data)

- **`io/hf_weights.py`** + **`io/arch/<family>.py`**: `load_hf_safetensors(model_path, flextrain_model) → None` iterates safetensors shards and copies into pinned host tensors allocated via `ParamSpec`, driven by a per-architecture HF-name → FlexTrain-field map (see *Multi-architecture strategy* above). `export_to_hf(model, out_dir)` does the reverse. Mapping tables are one ~20-line module per arch family; `AutoConfig` + `AutoTokenizer` are used only for config/tokenizer parsing, not for module instantiation.
- **`io/checkpoint.py`**: native format = `model.safetensors` (params) + `optimizer.safetensors` (Adam m/v or Muon state) + `meta.json` (step, config hash, tokenizer ref). Resume rebuilds `ActiveModel` from config, then copies into the same pinned buffers.
- **`io/tokenstream.py`**: one Protocol — `iter_sequences() → Iterator[Sequence]`. `HFTokenStream` wraps `datasets.load_dataset` + `AutoTokenizer` with background prefetch. `ShardTokenStream` ports `sequence_pool.py` verbatim behind the same interface. The engine consumes `Sequence` objects exactly as today.

## Execution plan (implementation order)

1. **`core/activation_schema.py`** — field/schema/slot types. Unit test: round-trip a dense-block schema at each tier, byte-count check.
2. **`core/layer.py`** + **`core/save_level.py`** — Protocols, `ParamSpec.derive_grad_spec()`, `ComputeCost.sum()`, DP table builder. Unit test: build DP tables for a 4-layer, 2-chunk, heterogeneous-tier model.
3. **`nn/blocks/`** — port norm (RMS + LayerNorm + QK-norm) / attention (GQA + SWA + linear) / dense-FFN (SwiGLU + GELU-gated) / MoE-FFN from `dense_layer.py` and `moe_layer.py`, using the blocks + schema contract. Shared GEMM fusions (e.g. O-proj + residual) stay inside `AttentionBlock`.
4. **`nn/layers/llama.py`** — first architecture: `LlamaBlock` as block composition. This is the template for every subsequent architecture.
5. **`nn/embed.py`**, **`nn/head.py`** — port via `InputLayer`, `OutputLayer` protocols; derive create/load/save from `ParamSpec`.
6. **`engine/`** — port `active_model.py` in four modules (`active_model.py` orchestrator, `buffers.py`, `streams.py`, `schedule.py`). The `forward`/`backward` chunk-loops stay; call sites to `layer.make_act_slot` / `layer.send_activations_home` / `layer.get_act_slot_size` all become engine-side methods on `BufferManager`. DP-table builder in `core/save_level.py` pads to `max_k_across_layers` so heterogeneous backbones work from day one.
7. **`core/working_set.py`** — drop-in port of `orig/working_set.py` with dataclass return type instead of dict.
8. **`optim/`** — wrap existing `awsm_adamw_step` / `awsm_muon_step`; derive opt_state from `ParamSpec`. Collapse the 9 unrolled `awsm_adamw_step` calls per layer ([dense_layer.py:344](orig/awsm_transformer/dense_layer.py#L344)) into a loop.
9. **`io/`** — HF safetensors loader + native checkpoint format + tokenstreams (HF `datasets` + ported `.bin` shard reader). `AutoConfig` / `AutoTokenizer` only, no `AutoModelForCausalLM` instantiation.
10. **Architectures (iterate)** — Llama3 ✅ (step 4). Then Qwen3-dense → OLMoE → Qwen3-MoE → Mixtral → DeepSeek-MoE → Mistral → GPT-OSS → Gemma/Phi/GLM/InternLM/Yi/Baichuan2 → Qwen3-Next. Each: one `nn/layers/<arch>.py`, one `io/arch/<arch>.py` weight map, one HF-config adapter entry, one numerical-parity smoke test.
11. **`cli.py`** + **`config.py`** — one entrypoint: `python -m flextrain train <yaml>` loads `ModelConfig` (hand-written or derived from HF `config.json` via `hf_config_to_flextrain`), `TrainConfig`, `HardwareConfig` and runs the same loop that `orig/train.py` ran.

## Verification

- **Numerical parity smoke test**: pick a small config (`nanogpt_124M` from `model_dims.json`), random-init, fix RNG, run 5 steps on both `orig/active_model.py` and the new `flextrain.engine.ActiveModel`, compare losses bitwise (allowing bf16 tolerance). Lives at `tests/test_parity.py`.
- **AdaWS invariants test**: on a Llama3-8B-scale dummy config, confirm the save-level DP output for a known (compute_times, transfer_durations) input matches the current `transmission_scheduler.solve()` call byte-for-byte.
- **HF round-trip per architecture**: for each arch added in step 10, load the smallest public safetensors checkpoint (e.g. Llama3-8B, Qwen3-0.6B, OLMoE-7Bx1B, etc.) → run 0 steps → export → diff against source safetensors (identity modulo dtype cast). Catches weight-map typos. Also: one-forward-pass logit parity against HF `AutoModelForCausalLM.forward` at fp32 tolerance for the first token of a fixed prompt — catches semantic bugs (transpose, norm-order, RoPE base, QK-norm placement).
- **End-to-end training**: run `python -m flextrain train configs/llama3_8b_hf_fineweb.yaml` on H100 with 40GiB budget and 256GiB host for 100 steps; confirm throughput is within 10% of the `bench_train.py` orig number for the same config (tracked in the final paragraph of §4.1 of the paper).
- **Long-context spot check**: `llama3_8B` at seq_len=64K with host-budget=256GiB; confirm `recomputation_frac → 0` as in Figure 5 of the paper.

## Critical files (today) the new code must mirror behaviorally

- [orig/active_model.py:531–665](orig/active_model.py#L531) — `determine_saved_levels` (DP input assembly).
- [orig/active_model.py:1162–1632](orig/active_model.py#L1162) — `fwd_bwd` orchestration loop (forward + backward + step).
- [orig/working_set.py:215](orig/working_set.py#L215) — `determine_working_set_config` (TR/TC/NP/NG/NA derivation).
- [orig/awsm_transformer/dense_layer.py:23–1077](orig/awsm_transformer/dense_layer.py#L23) — reference implementation of a dense block (fwd, recompute, bwd, act slot, flops).
- [orig/awsm_transformer/moe_layer.py](orig/awsm_transformer/moe_layer.py) — MoE reference; especially the per-expert `x_up` slicing, to validate the `persist=False` + runtime-views pattern.
- [orig/awsm_transformer/embed.py](orig/awsm_transformer/embed.py), [orig/awsm_transformer/head.py](orig/awsm_transformer/head.py) — embed/head reference including HF-safetensors target shapes.
- [orig/transmission_scheduler_pkg/transmission_scheduler/__init__.py](orig/transmission_scheduler_pkg/transmission_scheduler/__init__.py) — DP solver signature; kept verbatim as dependency (we still `pip install` this C extension).
- [orig/model_dims.json](orig/model_dims.json) — 10 model configs; becomes `flextrain/configs/models/*.yaml` (or loaded as-is).

## Known tradeoffs / what this does NOT do

- **HF `AutoModelForCausalLM` is not used for compute.** We load config + weights + tokenizer from HF, then run our own forward/backward. Cost: a new HF architecture requires a FlexTrain `nn/layers/<arch>.py` (30–80 LOC) + weight-map (~20 LOC) + config adapter (~15 LOC). Benefit: we get the per-tensor save-level policy that gives AdaWS its throughput win — something you structurally cannot get from patching HF forward passes (MegaTrain's 60% ceiling is the demonstration).
- **VLMs (Qwen-VL / InternVL / LLaVA / etc.) are deferred.** Vision towers + variable-length image sequences + M-RoPE + cross-modal attention add a second design axis on top of AdaWS. Revisit after the text-only menagerie is stable.
- **No multi-GPU / ZeRO / pipeline parallelism.** Paper's §5 says single-GPU is the right focus; adding distributed is a v3.
- **No W&B / typer / extensive test harness.** User explicitly deprioritized.
- **Loss of hand-tuned per-layer micro-optimizations.** Specifically, the fused `dispatcher.matmul(A, B, C=X, D=act_slot["xo"], beta=1.0)` that folds the attention residual into the O-proj GEMM must be recovered *inside* `AttentionBlock.forward` — block composition happens at init, not at each forward, so no perf penalty but the fusion code moves.
- **No `forward_recompute` "magic"**. Layers still have to explicitly branch on `slot.has("x1")` etc. We gain type-checking and lose implicit dict-lookup behavior — net positive for maintainability.
- **Dashboard stays in orig/** until explicitly ported.
- **`model_dims.json` format** may be reshaped into dataclass `ModelConfig`, but keep a loader for the JSON so existing model entries work.
