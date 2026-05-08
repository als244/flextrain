# Tutorial — adding Phi-3 (composability case)

This is the **composability** path: your new arch's per-layer math
matches an existing layer family, so you only write the arch +
builder + post-load hook. Most new archs in practice land here.

If your arch needs a new block (different attention or FFN
algorithm) or a new layer (different per-layer composition like
parallel residual), follow [`tutorial.md`](tutorial.md) instead —
it walks the full ladder block → layer → model.

Phi-3 is a useful target because it's:

* a **real model** people actually want to train;
* **architecturally Llama-shaped** (pre-norm GQA / MHA + SwiGLU +
  RMSNorm + full RoPE), so `LlamaBlock` works as-is;
* but stores its Q/K/V projection as one packed `qkv_proj.weight`
  tensor and its gate/up projection as one packed `gate_up_proj.weight`
  — meaning we DO have to write a `post_load_hook` to unpack them.

That weight-unpacking pattern is the most common reason a new arch
needs more than just an `ArchSpec` table. Once you've done it once,
the rest of the in-tree archs make sense.

If you haven't read [`flow.md`](flow.md) yet, read it first — the
mental model below assumes you know the per-step traversal and the
slot/chunk lifecycle.

## Phi-3-mini-4k at a glance

```python
# from HF config.json
{
  "architectures": ["Phi3ForCausalLM"],
  "hidden_size": 3072,
  "num_attention_heads": 32,
  "num_key_value_heads": 32,         # MHA: n_kv == n_heads
  "intermediate_size": 8192,
  "vocab_size": 32064,
  "num_hidden_layers": 32,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,             # NB: not 500_000 like Llama-3
  "tie_word_embeddings": false,
}
```

HF tensor names (per layer):

```
model.layers.{i}.input_layernorm.weight              -- 1:1 → w_attn_norm
model.layers.{i}.self_attn.qkv_proj.weight           -- packed [Q,K,V] → w_q,w_k,w_v
model.layers.{i}.self_attn.o_proj.weight             -- 1:1 → w_o (with TRANSPOSE)
model.layers.{i}.post_attention_layernorm.weight     -- 1:1 → w_ffn_norm
model.layers.{i}.mlp.gate_up_proj.weight             -- packed [gate,up] → w_1,w_3
model.layers.{i}.mlp.down_proj.weight                -- 1:1 → w_2 (with TRANSPOSE)
```

Plus the usual `model.embed_tokens.weight`, `model.norm.weight`,
`lm_head.weight` at the model scope.

## Step 1 — Pick the layer class (reuse LlamaBlock)

Phi-3's per-layer compute is exactly Llama's:

```
x = x + W_O · attn(x_q, x_k, x_v)        where x_{q,k,v} = W_{q,k,v} · rmsnorm(x)
x = x + W_2 · (silu(W_1·rmsnorm(x)) ⊙ (W_3·rmsnorm(x)))
```

`flextrain/nn/layers/llama.py` already does this. We don't need a
new layer file — the builder will instantiate `LlamaBlock` directly.

When DO you need a new layer? When the per-layer math actually
differs from any existing layer:

* Dual-residual norms (Gemma 2/3 — pre + post norms each side).
* Per-head QK-norm (Qwen3 dense — `Qwen3DenseBlock`).
* Hybrid linear+full attention (Qwen3-Next, Qwen3.5 —
  `Qwen3NextLinearLayer` + `Qwen3NextFullLayer`).
* Novel attention or FFN (e.g. MLA, attention-sink, Mamba).

For Phi-3, none of those apply. Move on.

## Step 2 — Write the ArchSpec + post_load_hook

Create `flextrain/io/arch/phi3.py`. Three concerns:

1. The 1:1 weight-name table for tensors HF stores in their natural
   FlexTrain shape.
2. The packed QKV / gate-up tensors that need a post-load hook to
   slice and copy into the right FlexTrain slots.
3. The Q/K halved → pair-interleave permutation for FlexTrain's RoPE
   kernel. Same as Llama.

```python
# flextrain/io/arch/phi3.py
"""Phi-3 family HF <-> FlexTrain mapping.

Architecturally: Llama-style decoder (pre-norm GQA + SwiGLU + RMSNorm
+ full RoPE). Wire-format quirks vs. Llama:

* qkv_proj is packed: HF stores Q, K, V concatenated in the OUT dim
  of one tensor. Hook splits + transposes into FlexTrain's separate
  w_q, w_k, w_v slots.
* gate_up_proj is packed: HF stores [gate, up] concatenated in the
  OUT dim. Hook splits + transposes into w_1 (gate) and w_3 (up).
* Standard halved → pair-interleave Q/K permutation (same as Llama).
"""
from __future__ import annotations
import os
from typing import Any, Mapping
import torch

from flextrain.api import BuildContext, register_block_builder
from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


def _phi3_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Unpack qkv_proj → w_q,w_k,w_v and gate_up_proj → w_1,w_3.

    Reads the two packed HF tensors per layer directly from the
    safetensors shards (since they have no WeightMapEntry rows).
    HF stores nn.Linear weight as (out_features, in_features), so
    we slice along dim 0 for the packed dimension and transpose
    before copying into FlexTrain's (in, out) layout.
    """
    from safetensors import safe_open

    # Determine shapes from layer 0's pre-allocated FT tensors.
    sample_w_q = dest[("layer_0", "w_q")]    # (d_model, attn_dim)
    sample_w_k = dest[("layer_0", "w_k")]    # (d_model, kv_dim)
    d_model, attn_dim = sample_w_q.shape
    _, kv_dim = sample_w_k.shape

    sample_w_1 = dest[("layer_0", "w_1")]    # (d_model, expert_dim)
    _, expert_dim = sample_w_1.shape

    # Resolve which shard each tensor lives in (single or multi-file).
    idx_path = os.path.join(hf_path, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        import json
        with open(idx_path) as f:
            file_index: dict[str, str] = json.load(f)["weight_map"]
    else:
        file_index = {}    # single shard: every tensor in model.safetensors

    open_files: dict[str, Any] = {}
    def _get(name: str) -> torch.Tensor:
        shard = file_index.get(name, "model.safetensors")
        if shard not in open_files:
            open_files[shard] = safe_open(
                os.path.join(hf_path, shard), framework="pt"
            )
        return open_files[shard].get_tensor(name)

    try:
        for L in range(num_layers):
            # qkv_proj is (q_out + k_out + v_out, d_model)
            qkv = _get(f"model.layers.{L}.self_attn.qkv_proj.weight")
            q = qkv[:attn_dim, :]                                  # (attn, d)
            k = qkv[attn_dim : attn_dim + kv_dim, :]               # (kv,  d)
            v = qkv[attn_dim + kv_dim :, :]                        # (kv,  d)
            dest[(f"layer_{L}", "w_q")].copy_(q.t().contiguous())  # (d, attn)
            dest[(f"layer_{L}", "w_k")].copy_(k.t().contiguous())  # (d, kv)
            dest[(f"layer_{L}", "w_v")].copy_(v.t().contiguous())  # (d, kv)

            # gate_up_proj is (2 * expert_dim, d_model), packed [gate, up].
            gate_up = _get(f"model.layers.{L}.mlp.gate_up_proj.weight")
            gate = gate_up[:expert_dim, :]
            up = gate_up[expert_dim:, :]
            dest[(f"layer_{L}", "w_1")].copy_(gate.t().contiguous())
            dest[(f"layer_{L}", "w_3")].copy_(up.t().contiguous())
    finally:
        for f in open_files.values():
            f.__exit__(None, None, None)


PHI3_ARCH = ArchSpec(
    hf_arch_ids=("Phi3ForCausalLM",),
    embed=(
        WeightMapEntry(
            flextrain_name="w_tok_embeddings",
            hf_name="model.embed_tokens.weight",
            transform=Transform.NONE,
        ),
    ),
    head=(
        WeightMapEntry(
            flextrain_name="w_final_norm",
            hf_name="model.norm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_head_proj",
            hf_name="lm_head.weight",
            transform=Transform.TRANSPOSE,
        ),
    ),
    layer=(
        WeightMapEntry(
            "w_attn_norm",
            "model.layers.{i}.input_layernorm.weight",
            Transform.NONE,
        ),
        WeightMapEntry(
            "w_ffn_norm",
            "model.layers.{i}.post_attention_layernorm.weight",
            Transform.NONE,
        ),
        WeightMapEntry(
            "w_o",
            "model.layers.{i}.self_attn.o_proj.weight",
            Transform.TRANSPOSE,
        ),
        WeightMapEntry(
            "w_2",
            "model.layers.{i}.mlp.down_proj.weight",
            Transform.TRANSPOSE,
        ),
        # No WeightMapEntry rows for w_q / w_k / w_v / w_1 / w_3 —
        # they're populated by _phi3_post_load_hook above.
    ),
    post_load_hook=_phi3_post_load_hook,
)
register_arch(PHI3_ARCH)
```

Note `strict=False` is required at load time because `w_q`, `w_k`,
`w_v`, `w_1`, `w_3` aren't covered by `WeightMapEntry` rows —
`from_pretrained` defaults to non-strict, so callers don't need to
opt in.

For the full Transform / WeightMapEntry / post_load_hook reference,
see [`../weights.md`](../weights.md). For an alternate hook example
(MoE expert stacking), look at `flextrain/io/arch/olmoe.py`.

## Step 3 — `hf_config_to_flextrain` + `hf_config_to_hyperparams`

Same module, two small functions:

```python
def hf_config_to_flextrain(hf_config: Mapping[str, Any]) -> dict[str, int]:
    n_heads = int(hf_config["num_attention_heads"])
    n_kv_heads = int(hf_config.get("num_key_value_heads", n_heads))
    # Phi-3 sets head_dim explicitly (96 for mini, not hidden/n_heads).
    head_dim = int(hf_config.get(
        "head_dim", hf_config["hidden_size"] // n_heads
    ))
    return {
        "d_model": int(hf_config["hidden_size"]),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "attn_dim": n_heads * head_dim,
        "kv_dim": n_kv_heads * head_dim,
        "expert_dim": int(hf_config["intermediate_size"]),
        "vocab_size": int(hf_config["vocab_size"]),
        "n_layers": int(hf_config["num_hidden_layers"]),
    }


def hf_config_to_hyperparams(hf_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rms_norm_eps": float(hf_config.get("rms_norm_eps", 1e-5)),
        "rope_base": float(hf_config.get("rope_theta", 10000.0)),
        # Phi-3-medium-128k uses LongRoPE (rope_scaling type "longrope").
        # Pass through; LlamaBlockConfig will fall back to vanilla RoPE if
        # FlexTrain doesn't yet support the scaling variant.
        "rope_scaling": hf_config.get("rope_scaling"),
    }
```

`head_dim=96` is the one easy-to-miss detail. Phi-3 sets it
explicitly because `hidden_size / n_heads = 3072 / 32 = 96` happens
to coincide here, but the convention is to read `config.head_dim`
and fall back only if it's absent.

## Step 4 — Block builder

```python
def _phi3_block_builder(layer_idx: int, ctx: BuildContext):
    hp = ctx.hyperparams
    cfg = LlamaBlockConfig(
        d_model=ctx.dims["d_model"],
        n_heads=ctx.dims["n_heads"],
        n_kv_heads=ctx.dims["n_kv_heads"],
        head_dim=ctx.dims["head_dim"],
        expert_dim=ctx.dims["expert_dim"],
        rms_norm_eps=hp["rms_norm_eps"],
        rope_base=hp["rope_base"],
        rope_scaling=hp.get("rope_scaling"),
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )
    layer = LlamaBlock(layer_id=layer_idx, cfg=cfg)
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


register_block_builder(("Phi3ForCausalLM",), _phi3_block_builder)
```

This is the minimum block-builder. Notice it's almost identical to
`flextrain/io/arch/llama.py`'s — once you reuse a layer, the builder
is mostly boilerplate.

## Step 5 — Q/K halved → pair-interleave permutation

FlexTrain's RoPE Triton kernel uses pair-interleave (pairs `x[2i]`
with `x[2i+1]`); HF stores Q/K so RoPE pairs `x[:half]` with
`x[half:]`. After load, you must permute the OUT dim of `w_q` and
`w_k` so the LoRA delta and the base agree on coordinate ordering.

This is identical to Llama's. Add a `post_load_permute` to the same
arch module:

```python
def post_load_permute(am, hf_config, dims, hyperparams):
    """Permute Phi-3 Q/K halved-split → pair-interleave for FT's RoPE
    kernel. Same as Llama."""
    head_dim = int(dims["head_dim"])
    n_heads = int(dims["n_heads"])
    n_kv = int(dims["n_kv_heads"])
    n_layers = int(dims["n_layers"])
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    def _halved_to_pair(dim: int, head_dim: int) -> torch.Tensor:
        half = head_dim // 2
        out = torch.empty(dim, dtype=torch.int64)
        for h in range(dim // head_dim):
            base = h * head_dim
            for i in range(half):
                out[base + 2 * i] = base + i
                out[base + 2 * i + 1] = base + half + i
        return out

    q_perm = _halved_to_pair(attn_dim, head_dim)
    k_perm = _halved_to_pair(kv_dim, head_dim)
    for L in range(n_layers):
        host = am.buffers.host_params[L]
        host["w_q"].copy_(host["w_q"][:, q_perm])
        host["w_k"].copy_(host["w_k"][:, k_perm])
```

`from_pretrained` calls this automatically after `am.load_hf` returns,
if it's defined as a module-level function in the arch module.

## Step 6 — Wire it in

One line in `flextrain/io/arch/__init__.py`:

```python
from . import phi3  # noqa: F401
```

`Phi3ForCausalLM` snake-cases to `phi3` cleanly, so no
`_ARCH_MODULE_OVERRIDES` entry is needed in `flextrain/api.py`.

Now this just works:

```python
from flextrain import from_pretrained
from flextrain.optim.adamw import AdamW, AdamWHyperparams

am = from_pretrained(
    "models/Phi-3-mini-4k-instruct",
    optimizer=AdamW(AdamWHyperparams(lr=3e-5)),
    max_seq_len=1024,
    max_global_batch_tokens=1024,
    max_gpu_mem_bytes=24 << 30,
    max_host_mem_bytes=110 << 30,
)
```

## Step 7 — Test it

Four progressively-stronger checks. Don't skip ahead — each catches
a different class of bug.

### 7a. Step-0 logit + per-token CE diff vs HuggingFace

Run `tests/test_arch_parity.py` against the new arch:

```bash
python tests/test_arch_parity.py Phi-3-mini-4k-instruct
```

Add a model entry to the script's models list mirroring an existing
arch (e.g., `Llama-3.2-1B`). The harness runs both stacks (FlexTrain
and `transformers`) on the same MathInstruct prompts, then compares:

* Step-0 logit max|Δ| (forward floor — should be at the bf16 noise
  floor, ~0.5 on a small model).
* Step-0 per-token CE max|Δ|.
* Loss curve over N training steps (FT vs HF).

This catches: weight-name mapping bugs, post_load_hook slice errors,
RoPE convention mismatches, missing post-load permutation, fp32/bf16
mismatch with HF defaults.

### 7b. LoRA-mode parity vs HF PEFT

Run `tests/test_arch_lora_e2e.py`:

```bash
python tests/test_arch_lora_e2e.py --arch phi-3-mini-4k --mode lora
```

Add an arch entry mirroring `qwen3-1.7b`. This subprocess-isolates a
FT vs PEFT run with identical LoRA inits and asserts loss-curve
agreement within the configured tolerance.

This catches: LoRA wrapper integration, save-tier-dependent numerical
drift (i.e., `forward_recompute` not producing identical output to
`forward`).

### 7c. Block-level math parity (optional, useful for debugging)

If 7a/7b show a bigger gap than expected, drop down to
block-level. Build `LlamaBlock` with random weights, run forward
on a known input, compare against a hand-rolled reference forward
in plain PyTorch (no FlexTrain). The pattern lives in
`tests/moe/test_full_layer_parity.py` for MoE; the dense analog is
straightforward to adapt.

This catches: a bug in your config translation (wrong head_dim,
wrong rope_base) that integration tests would also catch but
slowly.

### 7d. HF export round-trip (only if you plan to export)

If you'll export Phi-3 trained weights back to HF format, run
`tests/io/test_export_roundtrip_loss.py` after registering a
pre-export hook that re-packs `w_q,w_k,w_v` → `qkv_proj` and
`w_1,w_3` → `gate_up_proj` (the inverse of `_phi3_post_load_hook`).
See [`../weights.md`](../weights.md) and [`../export.md`](../export.md)
for the pre-export hook contract.

This catches: silently-corrupt exports — the resumed loss must be
strictly lower than the original loss on the same data prefix.

## What was specific to Phi-3 vs. universal

Universal — applies to every new arch:

* The 4-stage roadmap (layer | arch+hook | builder | wire).
* The Q/K halved → pair-interleave permutation (any arch using HF's
  RoPE convention).
* `from_pretrained` arch_id → module name auto-snake-casing; drop
  into `_ARCH_MODULE_OVERRIDES` in `flextrain/api.py` only when it
  doesn't.
* The 4-test pyramid (forward parity → LoRA loss parity → block
  math → export round-trip).

Phi-3-specific — the realistic wrinkle of THIS arch:

* `qkv_proj` and `gate_up_proj` weight packing → handled by
  `_phi3_post_load_hook`.

For a different new arch, the wrinkles will differ. Examples from
in-tree archs you can read for analog patterns:

| Wrinkle | See |
|---|---|
| MoE expert stacking (per-expert HF tensors → 3-D FT tensor) | `flextrain/io/arch/olmoe.py` |
| RMSNorm `γ - 1` storage (Gemma 2/3) | `flextrain/io/arch/gemma2.py` |
| Per-head QK-norm permutation (OLMoE, Qwen3) | `flextrain/io/arch/olmoe.py` (post_load_permute) |
| Hybrid linear+full attention layers | `flextrain/io/arch/qwen3_5.py` |
| Linear-attention bundled projections (FLA `in_proj_qkv`) | `flextrain/io/arch/qwen3_next.py` |

## Common pitfalls

* **Forgetting the post-load hook unpacks AFTER WeightMapEntry rows
  copy.** If you accidentally add a WeightMapEntry for `qkv_proj`,
  the loader will try to copy it as-is into the FT tensor — which
  has wrong shape. Leave the packed tensors out of `WeightMapEntry`
  and let the hook own them entirely.
* **Loading without `strict=False`.** When the hook owns Q/K/V/gate/
  up, the standard load can't find a 1:1 entry for them — `strict=
  True` would raise. `from_pretrained` defaults to `strict=False`,
  but if you call `am.load_hf` directly, pass `strict=False`.
* **Wrong slice arithmetic for GQA Phi-3.** Phi-3-medium has
  `n_kv_heads=10 < n_heads=40`. The hook slices Q at `[:n_heads * head_dim]`
  and K at `[n_heads*head_dim : (n_heads+n_kv_heads)*head_dim]` —
  unequal thirds, not `[:H,H:2H,2H:3H]`. The code above gets this
  right by using `attn_dim` and `kv_dim` from the sample tensors;
  don't hardcode `H * 3`.
* **Skipping the Q/K pair-interleave permutation.** Without it the
  step-0 logit max|Δ| is ~0.5–2.0× the bf16 noise floor and grows
  with depth. Almost always a sign the permute didn't run. If
  `post_load_permute` is defined as a top-level function in the arch
  module, `from_pretrained` calls it automatically.
* **bf16 master + fp32 norm grad** — RMSNorm weights are 1-D so
  the fp32-vs-bf16 byte cost is negligible. The convention in
  every in-tree arch is fp32 norm master + fp32 norm grad on the
  intuition that fp32 may help precision on small-LR updates of
  small tensors; this hasn't been measured rigorously, so treat it
  as a low-cost convention rather than a correctness requirement.
  `BuildContext.norm_grad_dtype` defaults to fp32 for you.
* **Stashing tensors on `self`** — only `self.layer_id`,
  `self.schema`, `self.param_spec`, and config / submodule
  references survive between fwd and bwd. Use `slot.aux` for
  intra-layer ferrying, `chunk.extra` for cross-block-within-chunk
  state. See [`flow.md`](flow.md#engine-assumptions-dont-break-these).
