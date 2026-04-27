# Implementing a new block / layer / model

FlexTrain is composable at four levels, each one re-using the work
done at the level below:

* **Block** — an algorithmic unit that owns a portion of the layer's
  activations + parameters (e.g. `GQAAttentionBlock`,
  `SwiGLUFFN`, `MoESwiGLUFFN`, `RMSNormBlock`, `GatedDeltaNetBlock`).
  A block declares its `fields()` (activation schema entries) and its
  `param_spec()` and exposes `fwd` / `bwd` methods.
* **Layer** — composes blocks into a transformer "decoder layer"
  (e.g. `LlamaBlock`, `Qwen3DenseBlock`, `OLMoEBlock`,
  `Qwen3MoEBlock`, `Qwen3NextLinearLayer`). A layer implements the
  `Layer` protocol (`forward` / `forward_recompute` / `backward` /
  `compute_cost`) by orchestrating its blocks.
* **Arch + block builder** (under `flextrain/io/arch/<name>.py`) —
  declares the HF tensor-name map (`ArchSpec`), the
  `hf_config_to_flextrain` and `..._hyperparams` translators, and a
  block builder that the high-level loader uses to instantiate one
  layer from `(layer_idx, BuildContext)`.
* **Model** — the engine glues an `embed: InputLayer`, a
  `backbone: list[Layer]`, and a `head: OutputLayer` together. End
  users get the engine via `flextrain.from_pretrained(model_path,
  ...)`; that just calls the registered block builder once per layer
  and wires up `ActiveModel`.

This page walks the contracts. The layer and block code paths support
heterogeneous backbones — a single model can mix Llama-style layers and
MoE layers and Qwen3-Next linear-attention layers in any order, as long
as each layer's `schema` fits in the engine's GPU activation buffer.

## Roadmap for adding a new architecture

A complete walkthrough for adding e.g. a hypothetical `MyArch` model:

1. **Block** — if the architecture is mostly conventional (GQA + SwiGLU
   + RMSNorm), reuse the existing blocks under
   `flextrain/nn/blocks/`. If it has a novel attention or FFN,
   implement a new block following the contracts in this doc.
2. **Layer** — add `flextrain/nn/layers/myarch.py` defining
   `MyArchBlockConfig` (a frozen dataclass) and `MyArchBlock` (the
   layer class). Mirror `flextrain/nn/layers/llama.py` for the
   simplest case.
3. **Arch + builder** — add `flextrain/io/arch/myarch.py` that:
   * Declares an `ArchSpec` and calls `register_arch(...)` on import,
     mapping FT tensor names ↔ HF tensor names.
   * Implements `hf_config_to_flextrain(hf_config)` and
     `hf_config_to_hyperparams(hf_config)`.
   * (Optional) Implements `post_load_permute(am, hf_config, dims,
     hyperparams)` for Q/K-style weight fixups.
   * Implements `_myarch_block_builder(layer_idx, ctx) -> Layer` that
     reads `ctx.dims`, `ctx.hyperparams`, `ctx.compute_dtype`, ...,
     instantiates `MyArchBlock`, and (if `ctx.lora_targets`) wraps it
     in `LoRAWrapperLayer`.
   * Calls `register_block_builder(("MyArchForCausalLM",), _builder)`.
4. **Wire it in** — add `from .io.arch import myarch as _myarch  #
   noqa: F401` to `flextrain/__init__.py` so the registration runs at
   import time. Now `flextrain.from_pretrained("models/MyArch-1B")`
   just works for that family.
5. **Tests** — write a small math-parity test (block-level fwd+bwd
   vs autograd reference), then a step-0 loss-parity test against HF
   transformers (see `tests/test_lora_8b_diagnostics.py` for the
   pattern).

See `flextrain/io/arch/llama.py` for the canonical reference.

---

## The Layer protocol

```python
class Layer(Protocol):
    layer_id: int
    schema: ActivationSchema
    param_spec: ParamSpec

    def forward(
        self, x, chunk: ChunkMeta, weights, slot: ActivationSlot,
        ctx: LayerContext,
    ) -> torch.Tensor: ...

    def forward_recompute(
        self, slot, chunk, weights, ctx,
    ) -> None: ...

    def backward(
        self, dx, chunk, weights, grads, slot, ctx,
    ) -> torch.Tensor: ...

    def compute_cost(self, chunk) -> ComputeCost: ...
```

### `forward(x, chunk, weights, slot, ctx)`

Inputs:
* `x` — `(num_tokens, d_model)` residual stream input.
* `chunk` — `ChunkMeta` with sequence-packing info (`q_seq_lens`,
  `q_seq_offsets`, `seq_positions`, `total_q`, `total_k`,
  `prior_seq_lens_host`, `extra` — a mutable dict for engine-allocated
  per-chunk scratch like MoE token-index mappings).
* `weights` — `dict[str, Tensor]` keyed by your `param_spec` names
  (`w_q`, `w_attn_norm`, etc). All on-device.
* `slot` — `ActivationSlot` with pre-allocated tensors for every field
  in your `schema`. Layers MUST `slot.x_inp.copy_(x)` if their schema
  declares an `x_inp` field, write all required-tier-0 activations,
  and may write higher-tier activations if the slot's level allows
  (`slot.level >= field.tier`).
* `ctx` — `LayerContext` with `scratch(shape, dtype)` for ephemeral
  workspace, `kv_cache`, `stream`, optional `secondary_stream`, and
  `total_tokens_per_step` for MoE aux-loss scaling.

Returns the residual-added output `(num_tokens, d_model)`.

### `forward_recompute(slot, chunk, weights, ctx)`

Called during backward when `slot.level < schema.max_tier` —
fields with `tier > slot.level` were not saved at forward time and
must be reconstructed before `backward` is called.

Layers check `slot.has("xq")` (etc.) to decide what to recompute.
Save level semantics are documented in [docs/working_set.md](working_set.md).

### `backward(dx, chunk, weights, grads, slot, ctx)`

Inputs: upstream gradient `dx` (same shape as forward's return),
chunk meta, weights, mutable `grads` dict (engine-zeroed once per
training step), the `slot` with all activations now valid (post-recompute
if needed), and `ctx`.

Returns: `dx` for the preceding layer.
Side effect: accumulates weight grads into `grads[g_<param_name>]`
in-place. Naming convention: `g_<param_name without leading w_>`,
matching `state_key` / `flextrain.optim.base`.

### `compute_cost(chunk)`

Returns a `ComputeCost` with FLOPs per save-tier. Consumed by the
save-level solver. A reasonable default is to estimate fwd FLOPs +
recompute-on-bwd FLOPs at each tier (lower tier = more recompute).

## Activation schema contract

A field is one piece of saved-or-recomputable state:

```python
ActivationField(
    name="xq",
    shape_fn=lambda n, d: (n, d["n_heads"], d["head_dim"]),
    dtype=torch.bfloat16,
    tier=2,        # higher = saved at higher save levels only
    token_axis=0,  # which dim is num_tokens (the engine narrows this axis
                   # to chunk size at runtime). None = doesn't scale with T.
)
```

Save tier conventions (loose):
* **Tier 0** — always saved. Tiny, recompute-cheap, but recomputation
  is wasteful (e.g. RMSNorm's rstd, MoE expert counts, router weights).
* **Tier 1** — flash-attn-equivalent state (`attn_result`, `softmax_lse`).
  Mid-sized; recomputable at moderate cost.
* **Tier 2** — large pre-projection tensors (`xq`, `xo`).
* **Tier 3** — largest intermediates that fall out naturally during fwd
  recomputation (`x1`, `x3`, `x_up`).

The save-level solver picks the lowest tier that fits the working-set
budget while balancing the recompute FLOPs you reported in
`compute_cost`. You declare tiers; you don't choose save level.

### `slot.aux` — per-call mutable stash

Sometimes a fwd helper produces something a bwd helper needs that doesn't
fit naturally in the schema (e.g. a fully-recomputed RMSNorm output that
you just produced and want to reuse). Stash it on `slot.aux["..."]`.
The engine clears `aux` between layers, so it's strictly intra-layer.

### `chunk.extra` — per-chunk scratch

For per-chunk scratch tensors that span multiple layers (e.g. MoE token
index mappings shared across the layer's fwd and bwd), the engine
allocates these into `chunk.meta.extra[<layer_id>]`. Declare your need
via a `MoEChunkConfig` attribute on the layer; the engine probes and
allocates accordingly. See `flextrain/engine/active_model.py` for
the allocation site.

## ParamSpec contract

```python
ParamSpec(tensors=(
    TensorSpec(
        name="w_q",
        shape_fn=lambda d: (d["d_model"], d["attn_dim"]),
        compute_dtype=torch.bfloat16,
        master_dtype=torch.bfloat16,   # default = compute_dtype
        grad_dtype=torch.bfloat16,     # default = compute_dtype
        opt_state_dtype=torch.float32, # default = fp32
        optimizer="muon",              # default = None (auto-infer)
    ),
    # ...
))
```

* `name` — must start with `w_` for parameters managed by the
  optimizer. Special-purpose layers may use other prefixes.
* `shape_fn(dims) -> tuple[int, ...]` — the shape, computed from a
  `dims: dict[str, int]` map. Standard dim names: `d_model`,
  `n_heads`, `n_kv_heads`, `head_dim`, `attn_dim`, `kv_dim`,
  `expert_dim`, `vocab_size`, `num_experts`, `top_k`. Add custom
  ones (your dims map controls everything).
* Per-role dtypes (compute / master / grad / opt_state) are honored
  per-tensor by the engine. See [docs/dtypes.md](dtypes.md).
* `optimizer` — hint for hybrid optimizers like
  `HybridMuonAdamW`. `None` triggers auto-classification (2-D / 3-D
  → Muon, others → AdamW).

## Block contract (algorithmic unit)

Blocks are not part of the engine protocol — they're a convention. A
block conventionally has:

```python
class MyBlock:
    def __init__(self, cfg: MyBlockConfig): ...
    def fields(self) -> tuple[ActivationField, ...]: ...
    def param_spec(self) -> ParamSpec: ...
    def fwd(self, x, weights, slot, ctx) -> torch.Tensor: ...
    def bwd(self, dy, weights, grads, slot, ctx) -> torch.Tensor: ...
    def compute_cost(self, chunk, *, max_tier) -> ComputeCost: ...
```

Layers compose blocks by:
1. Declaring all blocks in `__init__`.
2. Concatenating `fields()` and merging `param_spec()` for the
   layer's schema and param_spec.
3. Calling `block.fwd / bwd` from the layer's `forward / backward`
   in the right order, with the right slice of weights / grads /
   slot.

## InputLayer (embedding)

Two methods: `forward(token_ids, chunk, weights, ctx) -> Tensor`
and `backward(dx, token_ids, chunk, weights, grads, ctx)`. No
`slot` (no activations to save other than the input tokens, which
the chunk already owns).

## OutputLayer (LM head)

Single method: `forward_loss_backward(x, targets, ...) -> (loss, dx)`
that fuses the head projection, cross-entropy loss, and the head's
backward in one step. This avoids materializing the full
`(num_tokens, vocab_size)` logits tensor.

## A worked example: a "MyLayer" that's Llama with an extra residual norm

```python
from flextrain.core.activation_schema import ActivationField, ActivationSchema, concat_fields
from flextrain.core.layer import ParamSpec
from flextrain.nn.blocks import GQAAttentionBlock, GQAAttentionConfig, RMSNormBlock, SwiGLUConfig, SwiGLUFFN

class MyLayerConfig:
    d_model = 2048
    # ... etc

class MyLayer:
    def __init__(self, layer_id, cfg):
        self.layer_id = layer_id
        self.cfg = cfg
        self._dims = {"d_model": cfg.d_model, ...}

        self.attn_norm = RMSNormBlock(prefix="attn_norm", eps=cfg.rms_norm_eps)
        self.attn = GQAAttentionBlock(GQAAttentionConfig(d_model=cfg.d_model, ...))
        self.extra_norm = RMSNormBlock(prefix="extra_norm", eps=cfg.rms_norm_eps)
        self.ffn_norm = RMSNormBlock(prefix="ffn_norm", eps=cfg.rms_norm_eps)
        self.ffn = SwiGLUFFN(SwiGLUConfig(d_model=cfg.d_model, ...))

        x_inp = ActivationField("x_inp", lambda n, d: (n, d["d_model"]),
                                cfg.compute_dtype, tier=0)
        self.schema = ActivationSchema(
            fields=concat_fields([
                self.attn_norm.fields(),
                (x_inp,),
                self.attn.fields(),
                self.extra_norm.fields(),
                self.ffn_norm.fields(),
                self.ffn.fields(),
            ]),
            max_tier=3,
        )
        self.param_spec = ParamSpec.merge([
            self.attn_norm.param_spec(),
            self.attn.param_spec(),
            self.extra_norm.param_spec(),
            self.ffn_norm.param_spec(),
            self.ffn.param_spec(),
        ])

    def forward(self, x, chunk, weights, slot, ctx):
        slot.x_inp.copy_(x)
        x_temp = ctx.scratch(x.shape, x.dtype)
        h = self.attn_norm.fwd(x, weights, slot.attn_norm_rstd, output=x_temp)
        a = self.attn.fwd(x, h, chunk, weights, slot, ctx)
        # extra norm step:
        h2 = self.extra_norm.fwd(a.view(-1, self.cfg.d_model), weights,
                                 slot.extra_norm_rstd, output=x_temp)
        h3 = self.ffn_norm.fwd(h2, weights, slot.ffn_norm_rstd, output=x_temp)
        return self.ffn.fwd(h3, weights, a, out_tensor=x, slot=slot, ctx=ctx)
    # backward / forward_recompute / compute_cost — mirror the LlamaBlock
    # pattern, calling each block's bwd / fwd_from_rstd / compute_cost.
```

## Testing your new block / layer

1. **Block parity vs naive PyTorch.** Build a small case with random
   weights, run `block.fwd` on FlexTrain side and an equivalent forward
   in pure PyTorch on the reference side; assert max |Δ| < a few × bf16
   noise (1e-3 to 1e-2 typical).
2. **Bwd parity vs autograd.** Same block, scoped autograd reference,
   compare `block.bwd` outputs against `torch.autograd.backward`. See
   `tests/test_gated_deltanet_bwd.py` and
   `tests/test_moe_routing_kernel.py` as templates.
3. **End-to-end engine parity.** Build a small model with the new layer,
   compare loss curves to a naive PyTorch implementation of the same
   architecture for a few SGD steps. See
   `tests/test_olmoe_engine_parity.py` as a template.
4. **Save-level invariance.** Run the same model under save_level=0 and
   save_level=max; loss curves must be bit-identical (engine guarantee).
   See `tests/test_save_level_parity.py`.
5. **Real HF weights.** Once block-level parity passes, write an
   `ArchSpec` (see [weights.md](weights.md)), load real HF weights,
   compare step-0 loss to HF transformers. The Q/K halved→pair RoPE
   permutation lives in `tests/test_llama32_1b_parity.py` if your arch
   needs it.

## Common pitfalls

* **Not copying `x` into `slot.x_inp`** — RMSNorm bwd needs the pre-norm
  input. If you don't save it, the slot's lookup will fail at bwd.
* **Wrong save-tier for an activation** — too low and you silently
  recompute every time; too high and you OOM. Match the orig blocks for
  reference.
* **Forgetting `chunk.extra` in MoE blocks** — the engine has no idea
  to allocate per-chunk MoE scratch unless you set
  `self.moe_chunk_config = MoEChunkConfig(...)` on the layer.
* **bf16 master + fp32 norm grad** — RMSNorm weights are tiny and
  benefit from fp32 master + fp32 grad accumulator. Set
  `param_master_dtype=torch.float32, param_grad_dtype=torch.float32`
  on RMSNormBlock; the engine handles per-role dtypes correctly.
* **Naming `g_*` keys with `w_*` prefix retained** — the engine strips
  the leading `w_` to construct grad keys (`w_q` → `g_q`). If you have
  a non-`w_`-prefixed parameter, the engine uses `g_<name>` directly.
