# Model contract

How a `Layer` becomes loadable from a HuggingFace checkpoint and
trainable end-to-end. This page documents the seam between an arch
module (`flextrain/io/arch/<name>.py`) and the engine's
`ActiveModel`.

For the weight-name mapping itself (`ArchSpec`, `Transform`,
`post_load_hook` for MoE expert stacking, RoPE permutations), see
[`../weights.md`](../weights.md). This page covers what builds the
layers and what the runtime API looks like once they're built.

## Three things an arch module must register

Per family (`flextrain/io/arch/myarch.py`):

1. **`ArchSpec` via `register_arch(...)`** — the HF tensor-name map.
   Covered in [`../weights.md`](../weights.md).
2. **`hf_config_to_flextrain(hf_config)`** — produces the `dims` map
   (d_model, n_heads, n_kv_heads, head_dim, vocab_size, num_experts,
   ...). The builder reads from this.
3. **A block-builder via `register_block_builder(...)`** — turns
   `(layer_idx, BuildContext)` into one configured `Layer`.

A typical arch module also has:

* **`hf_config_to_hyperparams(hf_config)`** — per-layer hyperparams
  (rope_base, rms_norm_eps, sliding window, load_balance_coef,
  routing_mode, ...). Read by the builder.
* **`post_load_permute(am, hf_config, dims, hyperparams)`** —
  optional. Q/K halved → pair-interleave permutation, QK-norm
  reshuffle, etc. Called by `from_pretrained` after weights load.

Look at `flextrain/io/arch/llama.py` as the canonical reference.

## `BuildContext` — what the builder receives

```python
@dataclass(frozen=True)
class BuildContext:
    hf_config: Mapping[str, Any]
    dims: Mapping[str, Any]
    hyperparams: Mapping[str, Any]
    compute_dtype: torch.dtype
    master_dtype: torch.dtype
    grad_dtype: torch.dtype
    norm_grad_dtype: torch.dtype
    lora_targets: object | None = None
    lora_rank: int = 16
    lora_alpha: float = 16.0
    lora_adapter_compute_dtype: torch.dtype | None = None
    lora_adapter_master_dtype: torch.dtype | None = None
    lora_adapter_grad_dtype: torch.dtype | None = None
```

Fields:

* `hf_config` — raw HF `config.json` dict.
* `dims` — produced by `hf_config_to_flextrain`. The builder reads
  it for shapes (`d_model`, `n_heads`, …).
* `hyperparams` — produced by `hf_config_to_hyperparams`. Per-layer
  arch knobs (rope_base, rms_norm_eps, sliding window, …).
* `compute_dtype` / `master_dtype` / `grad_dtype` — per-role dtype
  overrides applied uniformly to all backbone layers (default
  bf16 / bf16 / bf16). The block is free to override norms to
  fp32 internally.
* `norm_grad_dtype` — separate override for RMSNorm / 1-D weights
  (default fp32 — the bytes are negligible and fp32 grads avoid
  round-to-zero on small-LR updates).
* `lora_targets` / `lora_rank` / `lora_alpha` / `lora_adapter_*_dtype`
  — if `lora_targets` is non-empty, the builder MUST wrap the base
  layer in `LoRAWrapperLayer` with the given targets / rank / alpha
  / adapter-dtype overrides.

## The block-builder function

```python
BlockBuilder = Callable[[int, BuildContext], object]

def _myarch_block_builder(layer_idx: int, ctx: BuildContext) -> Layer:
    cfg = MyArchBlockConfig(
        d_model=ctx.dims["d_model"],
        n_heads=ctx.dims["n_heads"],
        # ... pull whatever you need from ctx.dims / ctx.hyperparams ...
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
    )
    layer = MyArchBlock(layer_id=layer_idx, cfg=cfg)
    if ctx.lora_targets:
        layer = LoRAWrapperLayer(
            layer,
            targets=ctx.lora_targets,
            rank=ctx.lora_rank,
            alpha=ctx.lora_alpha,
            adapter_compute_dtype=ctx.lora_adapter_compute_dtype,
            adapter_master_dtype=ctx.lora_adapter_master_dtype,
            adapter_grad_dtype=ctx.lora_adapter_grad_dtype,
        )
    return layer

register_block_builder(("MyArchForCausalLM",), _myarch_block_builder)
```

The builder runs `n_layers` times (once per backbone layer). The
returned object MUST satisfy the [`Layer`](layer_contract.md#the-layer-protocol)
Protocol — either a raw block (`MyArchBlock`) or a `LoRAWrapperLayer`
wrapping one.

## How `from_pretrained` finds your arch module

`flextrain.from_pretrained(model_path, ...)`:

1. Reads `config.json` to find `architectures: ["MyArchForCausalLM"]`.
2. Maps that arch_id to a module name. The default rule strips
   `ForCausalLM` / `Model` and snake-cases:
   `Qwen3MoeForCausalLM` → `flextrain.io.arch.qwen3_moe`.
3. If the default mapping is wrong for your arch, add an explicit
   entry to `_ARCH_MODULE_OVERRIDES` in `flextrain/api.py`:

   ```python
   _ARCH_MODULE_OVERRIDES: dict[str, str] = {
       # ...
       "MyArchForCausalLM": "myarch",
   }
   ```

4. Imports that module — the side-effect imports run
   `register_arch` and `register_block_builder`.
5. Looks up the registered `BlockBuilder`, instantiates each
   backbone layer via `_block_builder(layer_idx, BuildContext)`,
   builds the embed and head, and assembles an `ActiveModel`.
6. Calls `am.load_hf(model_path, ...)` to load HF safetensors and
   runs the arch's `post_load_permute` if registered.

For step 4 to fire automatically, your arch module must be reachable
via the standard import: add `from . import myarch  # noqa: F401`
to `flextrain/io/arch/__init__.py`.

## `ActiveModel` runtime API

What `from_pretrained` returns. Public methods used in normal
training loops:

```python
am = from_pretrained(...)

# One training step.
am.fwd_bwd(batch)        # forward + loss + backward (no opt step)
am.step()                # optimizer step + LR schedule advance

# Weight I/O.
am.load_hf(path)         # load HF safetensors into am.buffers
am.save_hf(out_dir,      # save FlexTrain-native HF-arch'd checkpoint
           arch=None,
           out_filename="model.safetensors")
```

`am.fwd_bwd(batch)` runs the full per-step schedule: data → embed →
backbone (all layers, all chunks, fwd) → head → backbone (reverse,
bwd, with `forward_recompute` on offloaded chunks) → embed bwd. It
accumulates grads into the internal buffers; it does NOT step the
optimizer.

`am.step()` runs the optimizer (AdamW / Muon / HybridMuonAdamW per
the spec passed to `from_pretrained`), advances the LR schedule,
zeros grads, and returns the new `step_num`.

### Exporting

For HF-compatible export (full / merged / LoRA-adapter), use the
helpers in `flextrain.export`:

```python
from flextrain.export import save_hf_full, save_hf_merged, save_lora_adapter

save_hf_full(am, "out/", arch_id="LlamaForCausalLM")     # full FT checkpoint
save_hf_merged(am, "out/", arch_id="LlamaForCausalLM")   # base + LoRA merged
save_lora_adapter(am, "out/")                            # PEFT-style adapter
```

See [`../export.md`](../export.md) for the full export contract and
correctness validation.

## Putting it all together

A complete working arch lives in three files:

```
flextrain/nn/blocks/<your_blocks>.py    -- if novel; else reuse existing
flextrain/nn/layers/myarch.py           -- MyArchBlock + MyArchBlockConfig
flextrain/io/arch/myarch.py             -- ArchSpec + hf_config_to_* + builder + register_*
```

Plus one line in `flextrain/io/arch/__init__.py` to make the
side-effect imports run, and (optionally) one line in
`_ARCH_MODULE_OVERRIDES` if the arch_id doesn't auto-snake-case.

The [`tutorial.md`](tutorial.md) walks all of this end-to-end with
runnable code.
