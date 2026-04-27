# Optimizers

FlexTrain ships three optimizers. All three honor per-`TensorSpec`
optimizer state dtypes (`opt_state_dtype`).

## AdamW

```python
from flextrain.optim.adamw import AdamW, AdamWHyperparams

opt = AdamW(
    AdamWHyperparams(lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8,
                     weight_decay=0.0),
    state_dtype=torch.float32,   # default; override to torch.bfloat16 for ~50% memory
)
```

State per param: `o_adam_m` (first moment), `o_adam_v` (second moment),
both at `state_dtype` (or per-tensor `opt_state_dtype` if set on
`TensorSpec`).

## Muon

```python
from flextrain.optim.muon import Muon, MuonHyperparams

opt = Muon(MuonHyperparams(lr=1e-3, beta=0.95, ns_iters=5))
```

State per param: `o_muon` (Newton-Schulz momentum), bf16 by default.

Muon's Newton-Schulz iteration is mathematically defined only on
**rank-2** matrices. Pure `Muon` is fine if every param in your model
is 2-D, but most transformers have RMSNorm γ (1-D) and embeddings
(2-D but typically AdamW-trained). Use `HybridMuonAdamW` instead.

## HybridMuonAdamW (recommended for production)

Applies Muon to 2-D dense projection weights and AdamW to everything
else. For 3-D MoE expert stacks `(num_experts, d, ...)`, the optimizer
iterates the expert dim and applies Muon per slice.

```python
from flextrain.optim.hybrid import (
    HybridMuonAdamW, HybridMuonAdamWHyperparams,
)
from flextrain.optim.adamw import AdamWHyperparams
from flextrain.optim.muon import MuonHyperparams

opt = HybridMuonAdamW(HybridMuonAdamWHyperparams(
    lr=3e-4,                         # informational only
    adamw=AdamWHyperparams(lr=3e-4, beta1=0.9, beta2=0.95, eps=1e-8),
    muon=MuonHyperparams(lr=3e-4, beta=0.95, ns_iters=5),
))
```

### Classification rules

`flextrain.optim.hybrid.infer_optimizer_for_param(spec, dims)` returns
`"muon"` or `"adamw"` based on:

1. Explicit `TensorSpec.optimizer` (always wins).
2. Otherwise:
   * 2-D tensors not matching an AdamW name fragment → Muon.
   * 3-D tensors (MoE expert stacks) not matching an AdamW name
     fragment → Muon (per-expert iteration).
   * Tensors whose names contain `_norm`, `embed`, `tok_embed`,
     `head`, `router`, or `lm_head` → AdamW.
   * 1-D tensors → AdamW.
   * 4-D+ tensors → AdamW.

You can override with `optimizer="adamw"` or `optimizer="muon"` on any
`TensorSpec`.

### State allocation

The opt-state spec declares all three slots (`o_adam_m`, `o_adam_v`,
`o_muon`) as the union, but the buffer allocator only materializes the
applicable subset per param. So a 2-D `w_q` (Muon) takes one
`o_muon_q` slot, while a 1-D `w_attn_norm` (AdamW) takes
`o_adam_m_attn_norm` + `o_adam_v_attn_norm`. No wasted bytes.

## Custom optimizer

To add a new optimizer, conform to `flextrain.optim.base.Optimizer`:

```python
class MyOpt(Optimizer):
    state_spec = OptimizerStateSpec(tensors=(
        OptStateTensor(name="o_my_state", dtype=torch.float32),
    ))

    def step(self, param_spec, master, grads, state, *, step_num):
        for p in param_spec.tensors:
            grad_key = "g_" + p.name[2:] if p.name.startswith("w_") else "g_" + p.name
            state_key = f"o_my_state_{p.name[2:]}"
            ret = my_kernel(master[p.name], grads[grad_key], state[state_key], ...)
            if ret != 0: return ret
        return 0
```

If your optimizer needs per-param classification (like the hybrid one),
subclass `OptimizerStateSpec` to override `per_param_state_tensors(p, dims)`.

## Loss scaling

The engine implicitly applies `loss / total_active_tokens` so the
optimizer sees a "per-token" gradient. There is no separate loss-scale
parameter. If you want a global loss scale, multiply `lr` by the inverse.
