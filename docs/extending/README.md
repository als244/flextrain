# Extending FlexTrain to a new model

These docs cover everything needed to add a new transformer
architecture to FlexTrain — what to write, where it goes, and how
it fits into the training engine.

## What FlexTrain composes

```
Block      algorithmic unit (attention, FFN, MoE, RMSNorm, RoPE, ...)
  └─→ Layer      one decoder layer (composes blocks; engine sees this)
        └─→ Arch+Builder    HF-loadable model family (config + weight map + builder fn)
              └─→ Model     ActiveModel = embed + backbone + head + optimizer
```

Each level has a contract; the engine assumes those contracts hold.
The four `*_contract.md` files spell each one out exactly.

## Where to start

Pick the doc that matches what you're trying to do:

| If you want to... | Read |
|---|---|
| Get the runtime mental model first | [`flow.md`](flow.md) |
| Naming, class shape, tier picks, memory + compute contracts | [`best_practices.md`](best_practices.md) |
| Walk the **full ladder** — write a new block, new layer, AND new arch | [`tutorial.md`](tutorial.md) |
| **Compose existing blocks/layers** into a new arch (most archs land here) | [`tutorial_phi3.md`](tutorial_phi3.md) |
| Pick which existing blocks to reuse | [`block_contract.md`](block_contract.md) |
| Look up exact `forward` / `backward_dgrad` / `compute_cost` signatures | [`layer_contract.md`](layer_contract.md) |
| Find what's in `ChunkMeta` / `LayerContext` / `ActivationSlot` | [`chunk_contract.md`](chunk_contract.md) |
| Wire the arch into `from_pretrained` | [`model_contract.md`](model_contract.md) |

If you're new to the codebase: read `flow.md` for the mental
model, skim `best_practices.md` for the conventions and
user-responsibility contracts, then walk `tutorial.md` for the
full procedure. `tutorial_phi3.md` covers the common case where
you don't need a new block or layer.

## Contents

* [`flow.md`](flow.md) — one training step end-to-end; object
  lifecycles; engine assumptions you must not break.
* [`best_practices.md`](best_practices.md) — naming and class-shape
  conventions; how to choose activation tiers and what to declare
  as a field vs. scratch; the user-responsibility contracts the
  working-set planner and DP solver depend on; symptom → cause
  table for debugging new blocks/layers.
* [`tutorial.md`](tutorial.md) — full ladder walkthrough. Adds a
  parallel-residual + vanilla-MLP arch ("ParArch", structurally
  GPT-NeoX/Pythia/GPT-J shaped). Covers writing `MLPBlock` (new
  block), `ParArchBlock` (new layer composing it), the arch /
  builder / hook / wire, and the 4-test pyramid. Use this when
  your arch's per-layer math doesn't match any existing layer.
* [`tutorial_phi3.md`](tutorial_phi3.md) — composability case.
  Adds Phi-3 by reusing `LlamaBlock` and writing only the arch +
  `post_load_hook` (HF packs Q/K/V into one tensor; FT wants them
  separate). Use this when your arch is structurally Llama / Qwen /
  OLMoE / etc. under different HF tensor names — most new archs
  in practice.
* [`block_contract.md`](block_contract.md) — Block convention plus
  the in-tree block catalog (attention, FFN, MoE, norm, linear-attn,
  RoPE) with real signatures.
* [`layer_contract.md`](layer_contract.md) — `Layer` Protocol,
  `InputLayer`, `OutputLayer`, `ActivationField` / `ActivationSchema`
  / save tiers, `ParamSpec` / `TensorSpec`, `BackwardIntermediates`.
* [`chunk_contract.md`](chunk_contract.md) — runtime values handed
  in on every protocol call: `ChunkMeta`, `LayerContext`,
  `ActivationSlot`. The `chunk.extra` vs `slot.aux` distinction.
* [`model_contract.md`](model_contract.md) — `ArchSpec`,
  `BuildContext`, `register_block_builder`, `from_pretrained`
  dispatch, `ActiveModel` runtime API.

## Related docs (outside `extending/`)

* [`../weights.md`](../weights.md) — HF safetensors I/O, `Transform`
  enum, `post_load_hook` for MoE expert stacking, Q/K halved →
  pair-interleave permutation. The model_contract page links here
  for weight-table specifics.
* [`../architectures.md`](../architectures.md) — list of currently
  supported HF arch IDs and which layer class implements each.
* [`../working_set.md`](../working_set.md) — how the planner picks
  chunk size, GPU layer counts, save tiers given your memory
  budget.
* [`../dtypes.md`](../dtypes.md) — compute / master / grad / opt-
  state dtype roles and recommended combinations.
* [`../lora.md`](../lora.md) — `LoRAWrapperLayer` API, MoE per-
  expert adapters, HF PEFT parity.

## Currently supported architectures

`flextrain/io/arch/` ships with: Llama 2 / 3 / 3.1+, Mistral, Qwen2,
Qwen3 (dense + MoE), Qwen3.5 (dense + MoE) / Qwen3.6, Qwen3-Next,
OLMoE. See [`../architectures.md`](../architectures.md) for the
full table with layer classes and HF arch IDs.
