# LoRA cross-stack parity — findings log

Snapshot of the FT-vs-HF LoRA parity work. Numbers below were measured
during the LoRA-rollout debugging pass; they document a real bf16
noise floor and a since-fixed YARN RoPE bug. Treat these as
provenance, not as live regression bounds — current tolerances live in
`tests/test_arch_lora_e2e.py`.

## Loss-curve agreement (representative numbers)

Llama-3.2-1B, 100 steps on MathInstruct, identical LoRA inits across
HF PEFT and FlexTrain:

| pair | max \|Δ\| over 100 steps |
|---|---|
| HF PEFT vs FT-full | ≈ 0.07 |
| HF PEFT vs FT-offload | ≈ 0.07 |
| FT-full vs FT-offload | **0.00** (bit-identical) |

Llama-3.1-8B, 50 steps, auto solver, HF-matched LoRA-side dtypes
(base bf16 frozen; `A` / `B` fp32 master + fp32 grad + fp32 AdamW
state). YARN RoPE scaling enforced (`rope_type: llama3`,
`factor: 8.0`); without it the FT/HF gap was ~2× larger and biased
toward early layers.

| pair | max \|Δ\| over 50 steps | step-0 \|Δ\| | mean Δ | per-step Pearson |
|---|---|---|---|---|
| HF PEFT vs FT (auto-offloaded) | 0.112 | 0.0014 | +0.034 | 0.98 |

## Step-0 diagnostic decomposition (Llama-3.1-8B)

* **FT-vs-FT bit-identity across two working-set configs** (8 vs 3
  GPU layers): per-token CE max\|Δ\|=**0.0**, all LoRA-B grads
  max\|Δ\|=**0.0**. The engine is fully deterministic across
  offloading levels even on 8B with LoRA.
* **Mean loss FT vs HF**: 1.8398 vs 1.8402, Δ = -4e-4. After the
  YARN fix the means agree to 4 decimal places.
* **Per-token CE FT vs HF**: max\|Δ\|=0.116 on individual positions.
  Both stacks compute valid CE values; deltas average out across
  positions.

## bf16 noise floor (defines the residual gap)

HF in bf16 vs HF in fp32 on the **same** model — no FT involved —
defines the within-stack bf16 noise floor:

| Comparison | model | logit max\|Δ\| | logit mean\|Δ\| |
|---|---|---|---|
| HF-bf16 vs HF-fp32 (within HF) | Llama-3.2-1B | 0.486 | 0.025 |
| FT-bf16 vs HF-bf16 (cross-stack) | Llama-3.2-1B | 0.438 | 0.036 |
| FT-bf16 vs HF-bf16 (cross-stack) | Llama-3.1-8B | 2.08  | 0.031 |

FT-vs-HF on 1B is **smaller than HF's own bf16 noise floor**. argmax
matches at every top-disagreement position. No algorithmic
disagreement — FT produces bf16-correct outputs. 8B is ~4× the 1B
floor, plausibly from 32 vs 16 layers compounding plus FT's Triton
flash attention vs HF's PyTorch SDPA.

* **LoRA-B gradient per-layer**: rel error degrades smoothly with
  depth (L31: 4%, L0: 12%) — backward bf16 numerics accumulate
  through 32 layers, not an isolated kernel bug.

## RoPE bug fixed during this work

Llama 3.1 / 3.2 / 3.3 use a **frequency-band-scaled RoPE**
(`rope_type: llama3`) for long-context support. FlexTrain's RoPE
kernel originally hardcoded vanilla `inv_freq[i] = θ^(-2i/D)` and
silently ignored `config.rope_scaling`. The kernel now takes a
precomputed `inv_freq` array (length D/2) and the block-level
`build_rope_inv_freq` builds the YARN-scaled curve when the HF
config calls for it. To use this from your own code, pass
`rope_scaling=hf_cfg["rope_scaling"]` into `LlamaBlockConfig` —
or just use `flextrain.from_pretrained`, which does it for you.

## OLMoE — per-expert LoRA determinism

OLMoE-1B-7B with **per-expert** LoRA (3-D adapters). Engine-
determinism check yields max |Δ| ≤ 0.005 between full-save and
offloaded configs. The small non-determinism comes from MoE routing
decisions varying with chunk packing under different working sets,
not from the LoRA path itself.
