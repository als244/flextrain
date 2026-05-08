# Weight I/O

FlexTrain reads and writes HuggingFace `safetensors` checkpoints. The
mapping between HF tensor names and FlexTrain parameter names is
declared per-architecture as an [`ArchSpec`](../flextrain/io/hf_weights.py)
and registered at import time.

## Loading HF weights

```python
am.load_hf("/path/to/Llama-3.2-1B", strict=True)
```

Behavior:

1. Reads `config.json` to find `architectures: ["LlamaForCausalLM"]`.
2. Looks up the registered `ArchSpec` for that ID
   (`flextrain.io.arch.llama` registers `LLAMA_ARCH`).
3. Iterates each scope (`embed`, `layer_{i}` for backbone, `head`) and
   each `WeightMapEntry` within. For each, reads the HF tensor by name,
   applies any declared `Transform`, and writes into FlexTrain's host
   parameter buffer.
4. If the arch has a `post_load_hook`, calls it once with
   `(hf_path, dest, num_layers)`. Used by MoE archs to stack per-expert
   HF tensors into FlexTrain's `(E, d, 2F)` / `(E, F, d)` layouts.
5. Any model-specific weight permutations the user wants (e.g. Llama Q/K
   halved → pair-interleave for our RoPE kernel) are applied by the
   caller AFTER `load_hf` returns, by editing
   `am.buffers.host_params[i][name]` in place.

`strict=True` raises if any expected HF tensor is missing. `strict=False`
is needed for tied-embedding models (Llama-3.2 small variants —
`lm_head.weight` is missing, copied from `embed_tokens.weight`) and for
MoE archs where `w_up` / `w_down` are populated by the post-load hook
rather than a direct entry.

## Saving HF weights

```python
am.save_hf("/path/to/output", arch_id="LlamaForCausalLM")
```

Inverts the load: writes `config.json` (FlexTrain reconstructs minimal
keys; you can add more) plus `model.safetensors`. For multi-shard
output, the size threshold is configurable via the CLI.

## ArchSpec — declaring a new architecture

For a new model family that mostly follows Llama / Qwen3 conventions:

```python
# flextrain/io/arch/myarch.py

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


MY_ARCH = ArchSpec(
    hf_arch_ids=("MyModelForCausalLM",),
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
            transform=Transform.TRANSPOSE,   # HF stores (out, in); we want (in, out)
        ),
    ),
    layer=(
        WeightMapEntry(
            flextrain_name="w_attn_norm",
            hf_name="model.layers.{i}.input_layernorm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_q",
            hf_name="model.layers.{i}.self_attn.q_proj.weight",
            transform=Transform.TRANSPOSE,
        ),
        # ...
    ),
)
register_arch(MY_ARCH)
```

Then add `from . import myarch` to
`flextrain/io/arch/__init__.py` so it gets registered on import.

### Transform reference

* `Transform.NONE` — copy as-is. Use for 1-D weights (RMSNorm γ),
  bias vectors, embedding tables.
* `Transform.TRANSPOSE` — swap the last two dims. HF stores
  `nn.Linear` weights as `(out_features, in_features)` (matching its
  `x @ W.T` semantics); FlexTrain wants `(in_features, out_features)`
  so the engine can do `x @ W` directly.

## post_load_hook — for MoE expert stacking

MoE architectures store per-expert tensors:

```
model.layers.{L}.mlp.experts.0.gate_proj.weight   # (F, d)
model.layers.{L}.mlp.experts.0.up_proj.weight     # (F, d)
model.layers.{L}.mlp.experts.0.down_proj.weight   # (d, F)
... × num_experts
```

FlexTrain consumes these as stacked tensors:

```
w_up   (num_experts, d_model, 2 * expert_dim)   # packed [up.T, gate.T]
w_down (num_experts, expert_dim, d_model)       # down.T
```

Note the packing order: **value (up) first, then gate**. The orig
SwiGLU MoE kernel reads `X[:, :F]` as the value (`x3`) and `X[:, F:]`
as the gate (`x1`).

The hook iterates layers × experts, reads the three per-expert HF
tensors, transposes + concatenates, writes into the pre-allocated
FlexTrain destination tensor. Reference:
[`flextrain/io/arch/olmoe.py`](../flextrain/io/arch/olmoe.py).

## Q/K halved → pair-interleave permutation

Most HF Llama-style models store Q/K projections so that RoPE pairs
`x[:half]` with `x[half:]`. FlexTrain's RoPE Triton kernel uses
pair-interleave (pairs `x[2i]` with `x[2i+1]`). After `load_hf` you
must permute the **last dim** of `w_q` and `w_k`:

```python
from tests.test_llama32_1b_parity import _permute_qk_for_pair_interleave

for i in range(n_layers):
    for name in ("w_q", "w_k"):
        w = am.buffers.host_params[i][name]
        am.buffers.host_params[i][name].copy_(
            _permute_qk_for_pair_interleave(w, head_dim)
        )
```

**Critical**: if your architecture has QK-norm (e.g. OLMoE, Qwen3),
you must also permute the **1-D** `w_q_norm` / `w_k_norm` weights —
they multiply post-projection Q/K per-dim, and changing the dim
ordering of Q/K must be reflected in the norm weight ordering.
The OLMoE arch's `post_load_permute` in `flextrain/io/arch/olmoe.py`
shows this end-to-end (Q/K halved-pair plus the matching norm
permutation).

## Skipping weights at load time

`strict=False` lets HF tensors that don't map cleanly be skipped
silently. This is appropriate for:
* Tied embeddings (`lm_head.weight` missing in HF, generated by
  copying `embed_tokens.weight`).
* MoE post_load_hook-populated weights (`w_up`, `w_down` —
  `WeightMapEntry`s aren't declared for them).
* Architecture-specific extras the engine doesn't yet read (rotary
  embedding cached freqs).

## Custom checkpoint format

If you want to save / load FlexTrain-native (no HF compatibility), just
serialize the whole `am.buffers` directly:

```python
torch.save({
    "embed": dict(am.buffers.host_embed_params),
    "layers": [dict(p) for p in am.buffers.host_params],
    "head": dict(am.buffers.host_head_params),
    # opt state, if you want to resume training:
    "opt_state": [dict(b.host) for b in am.buffers.host_opt],
}, "checkpoint.pt")
```

Loading inverts. There is no first-class checkpoint API yet — pull
this into your training script as a helper.
