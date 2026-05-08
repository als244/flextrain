# LoRA fine-tuning

FlexTrain supports LoRA (Low-Rank Adaptation) fine-tuning on every
architecture without modifying any base block code. LoRA is provided
as a thin **wrapper layer** that holds a base layer (LlamaBlock /
OLMoEBlock / Qwen3MoEBlock / etc.) and adds trainable A/B adapter
matrices. The base layer's weights are marked frozen — the engine
elides grad and optimizer-state allocations for them — so LoRA
training memory cost is dominated by the (tiny) A/B params and their
optimizer state.

## Concepts

LoRA replaces a frozen linear `y = x @ W` with

```
y = x @ W + (x @ A) @ B * (alpha / r)
  = x @ (W + A @ B * scale)        (algebraically equivalent)
```

where `A: (d_in, r)` and `B: (r, d_out)` are the trainable adapter
matrices. `r` is the LoRA rank (typically 8-32) and `alpha` is the
scaling factor.

### Per-expert LoRA on MoE

For 3-D MoE expert stacks `W: (E, d_in, d_out)`, FlexTrain creates
**per-expert adapters** `A: (E, d_in, r)`, `B: (E, r, d_out)`. Each
expert gets its own independent low-rank delta. The forward and
backward use batched matmul (`torch.bmm`) over the expert dim.

This is the **default** for MoE blocks. A future option will let users
choose a single shared adapter applied to all experts (matches HF PEFT
behavior on OLMoE).

## Quickstart

The simplest way to LoRA fine-tune any registered HF arch is
:func:`flextrain.from_pretrained`. It builds the engine, applies
arch-specific weight permutations, and **auto-initializes** LoRA A
to ``N(0, 0.02)`` and B to zero — the model behaves identically to the
base at step 0.

```python
import torch
from flextrain import from_pretrained
from flextrain.optim.adamw import AdamW, AdamWHyperparams

opt = AdamW(
    AdamWHyperparams(lr=1e-4, beta1=0.9, beta2=0.95, weight_decay=0.0),
    state_dtype=torch.float32,
)
am = from_pretrained(
    "models/Llama-3.2-1B",
    optimizer=opt,
    max_seq_len=1024, max_global_batch_tokens=1024,
    max_gpu_mem_bytes=int(8 * (1 << 30)),
    max_host_mem_bytes=int(110 * (1 << 30)),
    device="cuda:0",
    lora_targets="all", lora_rank=16, lora_alpha=16.0,
)
# Train as normal — am is fully configured.
```

For MoE archs (OLMoE, Qwen3-MoE) the same call works; the wrapper
auto-discovers per-expert 3-D weights and creates per-expert
adapters.

### Building the wrapper manually

If you need finer control (custom block configs, mixed arch
backbones, non-standard dims), wrap a base layer directly:

```python
from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer

cfg = LlamaBlockConfig(...)
dims = {"d_model": 2048, "n_heads": 32, ...}

base = LlamaBlock(layer_id=i, cfg=cfg)
layer = LoRAWrapperLayer(
    base,
    lora_targets="all",   # or ("w_q", "w_v"), or just "w_q"
    rank=16, alpha=16.0,
    dims=dims,
)
# Then pass `layer` to ActiveModel and initialize A/B yourself
# (see flextrain.api._init_lora_params for the standard recipe).
```

### `lora_targets` semantics

| Value | Meaning |
|---|---|
| `None` or `()` | No LoRA; behaves as a pass-through wrapper. |
| `"all"` | Every 2-D linear projection AND every 3-D MoE expert stack. Routers and 1-D weights (norms, biases) are excluded. |
| `("w_q", "w_v")` | Explicit list; conventional PEFT default. |
| `("w_up", "w_down")` | MoE expert stacks only. |

## Configuration

```python
LoRAWrapperLayer(
    base_layer,
    lora_targets,         # see table above
    rank: int = 16,
    alpha: float = 16.0,  # LoRA scaling: scale = alpha / rank
    dims: dict[str, int],  # standard FT dims map; used to resolve param shapes
)
```

`dims` is the same map you pass to `ActiveModel(dims=...)`. The wrapper
reads the base layer's `param_spec` shapes from this map at construction
time so it can size A and B correctly.

## Memory characteristics

For each LoRA-targeted base parameter `W: (d_in, d_out)`:

| What | Size |
|---|---|
| Base `W` (frozen) | `d_in * d_out` (compute slot + master, but no grad/opt-state) |
| LoRA `A` | `d_in * r` |
| LoRA `B` | `r * d_out` |
| LoRA `A` grad + `B` grad | `(d_in + d_out) * r * grad_dtype` |
| LoRA AdamW state | `2 * (d_in + d_out) * r * opt_dtype` |

For a 1B-param Llama at rank 16, total LoRA params ≈ 0.5% of base.
With bf16 grads + bf16 AdamW state, total LoRA-side memory is ~5 MB
per layer, vs 40 MB for full Q/K/V/O grad+state. **8x reduction.**

For OLMoE-1B-7B with per-expert LoRA (E=64) at rank 16, the per-expert
adapters ARE substantial (`(64, 2048, 16)` and `(64, 16, 2048)` per
target × 2 targets × 16 layers ≈ 200M LoRA params). Still ≪ the base
7B params, and nicely sharded into per-expert chunks.

### Forward overhead

The wrapper materializes `W' = W + A @ B * scale` per layer per fwd
call (and rebuilds in bwd). For dense layers this is ~one extra matmul
per LoRA target. For 3-D MoE stacks this is one `bmm` per call —
significant transient memory (`(E, d, d_out) bf16`), but it falls out
of scope between layers so peak memory only holds one layer's worth.

## End-to-end example: Llama-3.2-1B LoRA fine-tune on MathInstruct

```python
import os
import torch
from flextrain.bench.parity import _Seq, _flextrain_step
from flextrain.core.save_level import HardwareCost
from flextrain.core.working_set import determine_working_set_config
from flextrain.engine.active_model import ActiveModel
from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
from flextrain.nn.head import LMHead, LMHeadConfig
from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
from flextrain.optim.adamw import AdamW, AdamWHyperparams
from tests.test_llama32_1b_parity import (
    _halved_to_pair_perm, _permute_qk_for_pair_interleave, _pull_step_batches,
)

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
HF_PATH = "models/Llama-3.2-1B"

# 1. Read HF config.
import json
with open(os.path.join(HF_PATH, "config.json")) as f:
    hf_cfg = json.load(f)
d_model = hf_cfg["hidden_size"]
n_heads = hf_cfg["num_attention_heads"]
n_kv = hf_cfg["num_key_value_heads"]
head_dim = hf_cfg.get("head_dim") or (d_model // n_heads)
expert_dim = hf_cfg["intermediate_size"]
n_layers = hf_cfg["num_hidden_layers"]
vocab = hf_cfg["vocab_size"]

# 2. Build LlamaBlock + LoRA wrapper for each backbone layer.
cfg = LlamaBlockConfig(
    d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
    expert_dim=expert_dim,
    rms_norm_eps=hf_cfg.get("rms_norm_eps", 1e-5),
    rope_base=hf_cfg.get("rope_theta", 500_000.0),
    is_causal=True,
    compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    norm_grad_dtype=torch.float32,
)
dims = dict(
    d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv, head_dim=head_dim,
    attn_dim=n_heads*head_dim, kv_dim=n_kv*head_dim,
    expert_dim=expert_dim, vocab_size=vocab,
)
backbone = []
for i in range(n_layers):
    base = LlamaBlock(layer_id=i, cfg=cfg)
    layer = LoRAWrapperLayer(
        base, lora_targets="all", rank=16, alpha=16.0, dims=dims,
    )
    backbone.append(layer)

embed = TokenEmbedLayer(TokenEmbedConfig(
    vocab_size=vocab, d_model=d_model,
    compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
))
head = LMHead(LMHeadConfig(
    d_model=d_model, vocab_size=vocab,
    rms_norm_eps=hf_cfg.get("rms_norm_eps", 1e-5),
    head_chunk_size=512,
    compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    norm_grad_dtype=torch.float32,
))

# 3. Working set + engine.
ws = determine_working_set_config(
    model_dims=dict(
        d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv,
        head_dim=head_dim, expert_dim=expert_dim, vocab_size=vocab,
        n_layers=n_layers, num_shared_experts=1, num_routed_experts=0,
        top_k=0, is_causal=True,
        datatypes={"embed": "bfloat16", "head_proj": "bfloat16",
                   "attn_proj": "bfloat16", "expert_proj": "bfloat16",
                   "router": "bfloat16", "norm": "bfloat16",
                   "residual": "bfloat16"},
    ),
    max_seq_len=1024, max_global_batch_tokens=1024,
    training_config={"master_weight_dtype": "bfloat16",
                     "grad_dtype": "bfloat16",
                     "opt_choice": "AdamW", "opt_dtype": "bfloat16"},
    has_embed=True, has_head=True, num_local_layers=n_layers,
    max_gpu_mem_bytes=int(8 * (1 << 30)),
    max_host_mem_bytes=int(110 * (1 << 30)),
    leeway_gpu_mem_bytes=int(0.5 * (1 << 30)),
    leeway_host_mem_bytes=int(2 * (1 << 30)),
)
opt = AdamW(AdamWHyperparams(lr=1e-4, beta1=0.9, beta2=0.95, eps=1e-8),
            state_dtype=torch.bfloat16)
am = ActiveModel(
    embed=embed, backbone=backbone, head=head, optimizer=opt,
    working_set=ws, hw_cost=HardwareCost(peak_tflops=60.0, pcie_bw_gbps=20.0),
    dims=dims, device=DEVICE,
)

# 4. Load HF weights + Q/K halved->pair RoPE permutation on the BASE
#    weights and on the corresponding LoRA B's column dim.
am.load_hf(HF_PATH, strict=False)

attn_dim = n_heads * head_dim
kv_dim = n_kv * head_dim
q_perm = torch.tensor(_halved_to_pair_perm(attn_dim, head_dim), dtype=torch.int64)
k_perm = torch.tensor(_halved_to_pair_perm(kv_dim, head_dim), dtype=torch.int64)
for i in range(n_layers):
    for name in ("w_q", "w_k"):
        w = am.buffers.host_params[i][name]
        am.buffers.host_params[i][name].copy_(
            _permute_qk_for_pair_interleave(w, head_dim)
        )
    # Tied LoRA: B's column dim must match the permuted base's column dim.
    for nm, perm in (("w_q_lora_b", q_perm), ("w_k_lora_b", k_perm)):
        t = am.buffers.host_params[i].get(nm)
        if t is not None and t.dim() == 2:
            am.buffers.host_params[i][nm].copy_(t[:, perm])

# 5. Init LoRA A (random) and B (zero — so LoRA delta starts at zero
#    and the first-step output equals base behavior).
torch.manual_seed(20260424)
for L in range(n_layers):
    for name, t in am.buffers.host_params[L].items():
        if name.endswith("_lora_a"):
            t.normal_(mean=0.0, std=0.02)
        elif name.endswith("_lora_b"):
            t.zero_()

# Tied embeddings — for Llama-3.2 small variants, head is missing.
head_w = am.buffers.host_head_params.get("w_head_proj")
embed_w = am.buffers.host_embed_params.get("w_tok_embeddings")
if head_w is not None and head_w.abs().sum().item() == 0.0:
    head_w.copy_(embed_w.t())

am._refresh_gpu_residents()
for name, dev_t in am.buffers.gpu_head_params.items():
    dev_t.copy_(am.buffers.host_head_params[name])
torch.cuda.synchronize()

# 6. Train.
step_batches = _pull_step_batches(HF_PATH, n_steps=100, target_tokens_per_step=1024)
for step, batch in enumerate(step_batches):
    seqs = [_Seq(s.tokens.clone()) for s in batch]
    for d, s in zip(seqs, batch):
        d.targets = s.targets.clone()
    loss = _flextrain_step(am, seqs)
    if step < 5 or step % 10 == 0:
        print(f"step {step}: loss={loss:.4f}")

# 7. (Optional) Save just the LoRA params.
import safetensors.torch as st
lora_state = {}
for L in range(n_layers):
    for nm, t in am.buffers.host_params[L].items():
        if "_lora_" in nm:
            lora_state[f"layer_{L}.{nm}"] = t.cpu()
st.save_file(lora_state, "lora_adapter.safetensors")
```

## Llama-3.1-8B on a 24 GiB GPU

The same recipe scales to Llama-3.1-8B on a single 24 GiB card with
the working-set solver picking offloading automatically. The only
practical change vs the 1B example is letting the solver size things:

```python
ws = determine_working_set_config(
    model_dims=dict(...),    # built from HF config the same way
    max_seq_len=1024, max_global_batch_tokens=1024,
    training_config={"master_weight_dtype": "bfloat16",
                     "grad_dtype": "bfloat16",
                     "opt_choice": "AdamW", "opt_dtype": "bfloat16"},
    has_embed=True, has_head=True, num_local_layers=n_layers,
    max_gpu_mem_bytes=int(24 * (1 << 30)),
    max_host_mem_bytes=int(110 * (1 << 30)),
    leeway_gpu_mem_bytes=int(2 * (1 << 30)),
    leeway_host_mem_bytes=int(4 * (1 << 30)),
    verbose=True,
)
```

For Llama-3.1-8B this lands on `n_gpu_layers=13/32`, single-chunk
1024-token rounds, and ~1.94 GiB activation buffer; peak GPU
allocation through training is ~13.4 GiB. See
[`tests/test_lora_e2e_llama_8b.py`](../tests/test_lora_e2e_llama_8b.py)
for the runnable script (subprocess-isolated HF PEFT vs FT, 50 steps).

## Cross-stack parity

`tests/test_lora_e2e_llama_dense.py` runs Llama-3.2-1B with identical
LoRA inits across HuggingFace PEFT and FlexTrain (under both full-save
and offloaded working-set configs). 100 steps on MathInstruct yields:

| pair | max \|Δ\| over 100 steps |
|---|---|
| HF PEFT vs FT-full | ≈ 0.07 |
| HF PEFT vs FT-offload | ≈ 0.07 |
| FT-full vs FT-offload | **0.00** (bit-identical) |

`tests/test_lora_e2e_llama_8b.py` runs the same pattern on
Llama-3.1-8B for 50 steps with the auto solver and HF-matched
LoRA-side dtypes (base bf16 frozen; LoRA `A`/`B` fp32 master + fp32
grad + fp32 AdamW state — same as HF PEFT's defaults). Both stacks
use the model's correct **YARN RoPE scaling**
(`rope_type: llama3`, `factor: 8.0`); without it the FT/HF gap was
~2× larger and biased toward early layers.

| pair | max \|Δ\| over 50 steps | step-0 \|Δ\| | mean Δ | per-step Pearson |
|---|---|---|---|---|
| HF PEFT vs FT (auto-offloaded) | 0.112 | 0.0014 | +0.034 | 0.98 |

### Diagnostic test: `tests/test_lora_8b_diagnostics.py`

A focused step-0-only diagnostic (`tests/test_lora_8b_diagnostics.py`)
breaks the gap apart cleanly:

* **FT-vs-FT bit-identity across two working-set configs** (8 vs 3
  GPU layers): per-token CE max\|Δ\|=**0.0**, all LoRA-B grads
  max\|Δ\|=**0.0**. The engine is fully deterministic across
  offloading levels even on 8B with LoRA.
* **Mean loss FT vs HF**: 1.8398 vs 1.8402, Δ = -4e-4. After the
  YARN fix the means agree to 4 decimal places.
* **Per-token CE FT vs HF**: max\|Δ\|=0.116 on individual positions.
  Both stacks compute valid CE values; deltas average out across
  positions.

### Concretely: the residual gap is bf16, not a bug

The "bf16 noise" claim is verified by reference, not assumed. Running
HF in bf16 vs HF in fp32 on the **same** Llama-3.2-1B model (no FT
involved) yields:

| Comparison | model | logit max\|Δ\| | logit mean\|Δ\| |
|---|---|---|---|
| HF-bf16 vs HF-fp32 (within HF) | Llama-3.2-1B | 0.486 | 0.025 |
| FT-bf16 vs HF-bf16 (cross-stack) | Llama-3.2-1B | 0.438 | 0.036 |
| FT-bf16 vs HF-bf16 (cross-stack) | Llama-3.1-8B | 2.08  | 0.031 |

FT-vs-HF on 1B is **smaller than HF's own bf16 noise floor**. argmax
matches at every top-disagreement position. There is no algorithmic
disagreement — FT produces bf16-correct outputs. 8B is ~4× the 1B
floor, plausibly from 32 vs 16 layers compounding plus FT's Triton
flash attention vs HF's PyTorch SDPA.

* **LoRA-B gradient per-layer**: rel error degrades smoothly with
  depth (L31: 4%, L0: 12%) — backward bf16 numerics accumulate
  through 32 layers, not an isolated kernel bug.

### Notable RoPE bug fixed during this work

Llama 3.1 / 3.2 / 3.3 use a **frequency-band-scaled RoPE**
(`rope_type: llama3`) for long-context support. FlexTrain's RoPE
kernel originally hardcoded vanilla `inv_freq[i] = θ^(-2i/D)` and
silently ignored `config.rope_scaling`. The kernel now takes a
precomputed `inv_freq` array (length D/2) and the block-level
`build_rope_inv_freq` builds the YARN-scaled curve when the HF
config calls for it. To use this from your own code, pass
`rope_scaling=hf_cfg["rope_scaling"]` into `LlamaBlockConfig` —
or just use `flextrain.from_pretrained`, which does it for you.

`tests/test_lora_e2e_olmoe_moe.py` runs the same pattern for OLMoE-1B-7B
with **per-expert** LoRA (3-D adapters). Engine-determinism check
yields max |Δ| ≤ 0.005 between full-save and offloaded configs (small
non-determinism is from MoE routing decisions varying with chunk
packing under different working sets).

## What's verified

| Property | Test | Status |
|---|---|---|
| Math: dA, dB, dx vs autograd reference | `test_lora_wrapper_math.py` | ✓ bf16 noise |
| Frozen invariant: base weights don't change | `test_lora_engine_smoke.py` | ✓ bit-identical to init |
| LoRA A/B grads allocated; base grads NOT | `test_lora_engine_smoke.py` | ✓ |
| Loss decreases on real data | E2E tests | ✓ |
| HF PEFT parity (dense Llama) | `test_lora_e2e_llama_dense.py` | ✓ within bf16 noise |
| Engine determinism: full-save = offloaded | `test_lora_e2e_llama_dense.py` | ✓ bit-identical |
| Per-expert MoE adapters (3-D) | `test_lora_moe_math.py` | ✓ |
| OLMoE LoRA E2E under offload | `test_lora_e2e_olmoe_moe.py` | ✓ |

## Common pitfalls

* **Q/K halved→pair RoPE permutation**: HF stores Q/K weights in
  halved-split layout. FT's RoPE kernel uses pair-interleave. After
  `am.load_hf(...)` you must permute the OUT dim of `w_q`, `w_k`
  AND the matching dim of `w_q_lora_b`, `w_k_lora_b` so the LoRA
  delta is in the same coordinate system as the base. See the
  example above (and `tests/test_lora_e2e_llama_dense.py`).
* **B init**: must be **zero** so the first-step LoRA delta is zero.
  Without this, the model starts in a different state from the base
  HF model and you can't compare cross-stack.
* **OLMoE QK-norm permutation**: OLMoE has full-dim QK-norm. The
  norm weights `w_q_norm` / `w_k_norm` ARE 1-D and ALSO need
  permuting along their single dim when you permute `w_q` / `w_k`.

## What's not yet implemented

* **HF PEFT-compatible LoRA checkpoint export**. FT uses `w_q_lora_a` /
  `w_q_lora_b` naming and the (d_in, r), (r, d_out) shape convention.
  PEFT uses (r, d_in), (d_out, r). To use an FT-trained LoRA in
  inference via PEFT you need to transpose A and B and rename. A
  conversion utility will land later.
* **Shared-adapter MoE LoRA** (option (a) — one A,B applied to every
  expert). PEFT's OLMoE LoRA does this. Needed for direct cross-stack
  parity on MoE; FT defaults to per-expert (option (b), more standard).
* **Dropout**. PEFT supports `lora_dropout`; FT doesn't yet. Easy to
  add as a config field on `LoRAWrapperLayer` if needed.

See [docs/internal/SESSION_NOTES.md](internal/SESSION_NOTES.md) for the running design
log and any open questions.
