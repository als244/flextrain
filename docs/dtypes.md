# Datatypes

FlexTrain has four datatype roles per parameter and one per activation
field. They're independent — you can have a bf16 compute weight with
an fp32 master, an fp32 grad accumulator, and an fp32 optimizer state,
all on the same tensor. Each role is honored by the engine where it
matters.

## Parameter dtype roles

| Role | Where it lives | Default | Notes |
|------|----------------|---------|-------|
| `compute_dtype` | GPU compute slot | bf16 | What the matmul / kernel sees |
| `master_dtype` | Host-side authoritative copy | = `compute_dtype` | Read at optimizer step time |
| `grad_dtype` | GPU grad slot + host grad accumulator | = `compute_dtype` | What the bwd kernel writes |
| `opt_state_dtype` | Optimizer state buffers | bf16 | AdamW m/v, Muon momentum |

The engine casts between roles at the buffer boundaries:
* On parameter prefetch (host → device): `master_dtype` → `compute_dtype`
  if they differ.
* On grad offload (device → host): `grad_dtype` → host accumulator
  (also `grad_dtype`).
* On optimizer step: read master, accumulate per-tensor optimizer state
  in `opt_state_dtype`, write back to master in `master_dtype`. The
  optimizer kernel handles the casts internally.

### Recommended combinations

**bf16 everywhere (minimal memory, fastest)**
```python
TensorSpec(name="w_q", shape_fn=..., compute_dtype=torch.bfloat16,
           master_dtype=torch.bfloat16, grad_dtype=torch.bfloat16,
           opt_state_dtype=torch.bfloat16)
```
Used for fine-tuning, where bf16 is empirically fine on most ops.
Saves ~50% opt-state memory vs the default.

**bf16 compute + fp32 master (training-from-scratch)**
```python
TensorSpec(name="w_q", shape_fn=..., compute_dtype=torch.bfloat16,
           master_dtype=torch.float32, grad_dtype=torch.bfloat16,
           opt_state_dtype=torch.float32)
```
Used for cold-start pretraining at scale (Llama / Qwen / OLMoE recipes).
fp32 master + fp32 opt state preserve update precision; bf16 grads still
halve grad memory vs fp32.

**fp32 norms + bf16 projection (default for RMSNormBlock)**
```python
RMSNormBlock(
    prefix="attn_norm", eps=1e-5,
    param_compute_dtype=torch.bfloat16,
    param_master_dtype=torch.float32,    # tiny, no harm to keep fp32
    param_grad_dtype=torch.float32,      # grads are tiny too
)
```
RMSNorm weights are 1-D (size = `d_model`), so fp32 master/grad costs
almost nothing and avoids round-to-zero on small-LR updates.

## Per-tensor optimizer dtype

The hybrid optimizer (`HybridMuonAdamW`) reads each `TensorSpec.opt_state_dtype`
to decide what dtype to use for that tensor's state buffer. Muon
momentum on a 30B-MoE expert stack might be bf16 (saves 50% on a big
tensor); AdamW state on a 1-D bias might be fp32.

You can also explicitly tell the hybrid optimizer which rule to use
per tensor via `TensorSpec.optimizer ∈ {"muon", "adamw", None}`.
`None` triggers auto-classification.

## Activation field dtype

Each `ActivationField` declares one dtype:

```python
ActivationField(
    name="xq",
    shape_fn=...,
    dtype=torch.bfloat16,    # what's stored in the activation slot
    tier=2,
)
```

The engine slot allocations match this dtype. Most activations are bf16;
RMSNorm rstd is fp32 (it's a per-row reciprocal-stdev, dominates
precision of the recompute path).

## Loss / softmax precision

Cross-entropy and softmax internally upcast to fp32 for numerical
stability, then the result is cast back to the input dtype. This is
hard-coded in `flextrain.ops` kernels — you can't override per-call.

## Recommended defaults summary

| Scenario | compute | master | grad | opt_state |
|----------|---------|--------|------|-----------|
| Fine-tuning a pretrained model | bf16 | bf16 | bf16 | bf16 |
| Cold-start pretraining | bf16 | bf16 | bf16 | fp32 |
| Production-grade pretraining | bf16 | fp32 | bf16 | fp32 |
| Small models (<1B), debug | fp32 | fp32 | fp32 | fp32 |

For RMSNorm and other 1-D weights: always keep grad / master in fp32 if
you can afford it — the bytes cost is negligible.

## Memory accounting

For a parameter of `N` elements:

| Storage | Bytes |
|---------|-------|
| compute (GPU resident slot) | `N * compute_dtype.itemsize` |
| master (host pinned) | `N * master_dtype.itemsize` |
| grad (GPU + host) | `2 * N * grad_dtype.itemsize` |
| opt state (host) | `N * opt_state_dtype.itemsize * num_state_tensors` |

The working-set solver accounts for all four roles when budgeting GPU
memory. See [docs/working_set.md](working_set.md) for how to tune the
plan.
