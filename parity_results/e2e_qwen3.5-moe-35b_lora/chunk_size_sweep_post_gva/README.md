# Qwen3.5-MoE-35B-A3B LoRA — chunk-size sweep + cu_seqlens fix

Date: 2026-04-29
HEAD at start of investigation: `52172e4` (linear_attn: drop GVA repeat_interleave + fp32 q/k recompute).

## Hardware
- RTX 3090 24 GiB GPU + 128 GiB host
- conda env: `flextrain`

## Common settings
```
--model models/Qwen3.5-35B-A3B
--mode lora --lora-rank 16 --lora-alpha 16
--seq-len 2048 --global-batch-tokens 65536 --steps 5
--max-gpu-mem-gib 22.5 --max-host-mem-gib 110 --leeway-gpu-mem-gib 5
--data-source json_sft --dataset datasets/mathinstruct.jsonl
```
LR schedule: cosine 1e-4 → 1e-5 over 5 steps (default).

## What we found

The README's prior baseline (`2.54 → 1.10`) was **not the correct loss
curve** for Qwen3.5-MoE-35B — it reflected a real bug:
``flextrain/nn/blocks/linear_attn.py`` was passing ``cu_seqlens=None``
to FLA's ``chunk_gated_delta_rule`` fwd, bwd, and recompute. With
``cu_seqlens=None``, FLA treats the entire flextrain chunk as one
continuous sequence, so the recurrent state of the linear-attention
layer leaks across packed-sequence boundaries inside each chunk.

Diagnostic: the loss varied monotonically with chunk size (more
sequences per chunk → more leak → higher loss). This was the smoking
gun.

Fix: pass ``chunk.q_seq_offsets.to(torch.int64)`` as ``cu_seqlens``
in all FLA call sites. Data was already on ``ChunkMeta``; just needed
to thread the chunk through ``GatedDeltaNetBlock.{fwd, bwd,
fwd_recompute_fla}`` and the layers that call them
(``Qwen3_5LinearLayer``, ``Qwen3NextLinearLayer``).

After the fix, the chunk-size dependence collapses to bf16 noise.

## Step-by-step losses

### Before fix (original picker / chunk choice)

```
                   step 1   step 2   step 3   step 4   step 5
chunk 8192  def    2.2816   2.0360   1.5302   1.2321   1.0436
chunk 16384 def    2.5420   2.3071   1.7207   1.3236   1.1009   ← README baseline
chunk 32768 fsl=0  2.8111   2.5300   1.9102   1.4371   1.1553
```

Step 1 spread: 2.28 → 2.81, **+23% over 4× chunk-size variation** —
diagnostic for cross-seq state leak.

### After fix (cu_seqlens = chunk.q_seq_offsets)

```
                  step 1   step 2   step 3   step 4   step 5
chunk 8192        1.3344   1.2452   1.0046   0.9345   0.8623
chunk 16384       1.3364   1.2512   1.0060   0.9366   0.8622
chunk 32768       1.3394   1.2542   1.0099   0.9389   0.8639
```

Step 1 spread: 1.3344 → 1.3394, **+0.37% over 4× chunk-size variation**.
Step 5 spread: 0.8622 → 0.8639, **+0.20%**. bf16 noise floor.

## Conclusion

- The cu_seqlens fix is a real correctness fix, not a noise change.
  The model's loss curve is uniformly ~30% better at step 1 and
  ~20% better at step 5 across all chunk sizes after the fix.
- Picker-driven chunk-size variation is now numerically benign
  (sub-percent), unblocking the lin_q/k schema shrink in `52172e4`.
- The original README "2.54 → 1.10" baseline should be considered
  superseded; new baseline is "1.34 → 0.86" at any chunk size.

## Artifacts in this directory

| # | File | Settings | step 1 | step 5 |
|---|------|----------|-------:|-------:|
| 1 | `01_fsl3_chunk32k.log`            | force-save-level 3, chunk=32k | (host OOM) | – |
| 2 | `02_fsl0_chunk32k.log`            | force-save-level 0, chunk=32k | 2.8111 | 1.1553 |
| 3 | `03_default_fsl_chunk16k.log`     | --max-chunk-size 16384 (no fix) | 2.5420 | 1.1009 |
| 4 | `04_fsl1_chunk32k.log`            | force-save-level 1, chunk=32k | (host OOM) | – |
| 5 | `05_default_fsl_chunk8k.log`      | --max-chunk-size 8192 (no fix) | 2.2816 | 1.0436 |
| 6 | `06_chunk16k_cu_seqlens_fix.log`  | --max-chunk-size 16384 (FIXED) | 1.3364 | 0.8622 |
| 7 | `07_chunk8k_cu_seqlens_fix.log`   | --max-chunk-size 8192 (FIXED) | 1.3344 | 0.8623 |
| 8 | `08_chunk32k_cu_seqlens_fix.log`  | default chunk=32768 (FIXED) | 1.3394 | 0.8639 |

## Pending follow-ups

- Linear-attn currently restricts each sequence to fit inside one
  flextrain chunk (max_seq_len ≤ chunk_size). Long-context training
  (sequence > chunk) would need per-sequence linear-attn state
  buffers carried across chunks via FLA's ``initial_state`` /
  ``output_final_state``, plus state-grad propagation in the bwd.
  Scope is analogous to how full-attn updates the KV cache across
  chunks today.
