# Verified end-to-end runs

Smoke-test record from training a handful of supported models end-to-end
on the reference workstation:

- **Hardware**: RTX 3090 24 GiB GPU + 117 GiB host RAM.
- **Workload**: 5 steps on `datasets/mathinstruct.jsonl` with the default
  `Instruction:/Response:` prompt template, `--max-seq-len 2048`,
  mean-over-active-tokens loss
  (`CrossEntropyLoss(ignore_index=-100)` convention; matches HF / PEFT).
- **Default mode**: LoRA-all at rank 16 unless the row says otherwise.
- **Generation**: greedy generation also verified — coherent output, hits
  EOS naturally.

| Model | Params | Arch | Mode | Batch tokens | Loss curve (5 steps) |
|---|---|---|---|---|---|
| Llama-3.2-1B | 1B | dense | LoRA | — | _not re-verified_ |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | LoRA | — | _not re-verified_ |
| OLMoE-1B-7B | 7B / 1B-active | MoE (64 experts) | full | — | _not re-verified_ |
| Qwen3-8B | 8B | dense, QK-norm | LoRA | — | _not re-verified_ |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | LoRA | 65k | 0.797 → 0.620 |
| Qwen3.5-9B | 9B | hybrid linear+full attn, dense MLP | full | 65k | 0.744 → 0.455 |
| Qwen3.5-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | — | _not re-verified_ |
| Qwen3.6-27B | 27B | hybrid linear+full attn, dense MLP | LoRA | — | _not re-verified_ |
| Qwen3-30B-A3B | 30B / 3B-active | MoE (128 experts) | LoRA | — | _not re-verified_ |
| Qwen3.5-MoE-35B-A3B | 35B / 3B-active | hybrid + MoE (256+1 experts) | LoRA | 65k | 0.743 → 0.541 |

Loss values reflect mean cross-entropy over response tokens (positions
where `targets != -100`); prior versions of this table reported a
different convention (mean over all tokens, including prompt-position
zeros) so older numbers are not directly comparable.

Additional models supported by the existing arch loaders (require a
larger machine to actually train): Qwen3.6-35B-A3B, Qwen3.5-122B-A10B,
Qwen3.5-397B-A17B, Qwen3-Coder-30B-A3B-Instruct (no new wiring needed;
they reuse `Qwen3_5*` / `Qwen3Moe*` arch ids).
