# Supported architectures

LoRA fine-tuning works on every entry below via
:class:`LoRAWrapperLayer` — see [docs/lora.md](lora.md). For MoE
architectures, the wrapper creates **per-expert adapters** by default.



| Family | Layer class | HF arch ID | `from_pretrained`? | Tested at scale | Notes |
|--------|-------------|------------|--------------------|-----------------|-------|
| Llama 2 / 3 / 3.1+ | `LlamaBlock` | `LlamaForCausalLM` | ✓ | 8B end-to-end PASSED | RoPE pair-interleave perm + Llama-3.1 YARN scaling |
| Qwen2 (with biases) | `Qwen2Block` | `Qwen2ForCausalLM` | ✓ | small parity PASSED | QKV biases supported |
| Qwen3 dense | `Qwen3DenseBlock` / `Qwen3DenseSWABlock` | `Qwen3ForCausalLM` | ✓ | 1.7B parity PASSED, 1.7B `from_pretrained` LoRA smoke PASSED | per-head QK-norm, optional alternating SWA |
| Qwen3-MoE | `Qwen3MoEBlock` | `Qwen3MoeForCausalLM` | ✓ | small-init parity PASSED | per-head QK-norm + MoE FFN |
| Qwen3.5 dense | `Qwen3_5FullLayer` + `Qwen3_5LinearLayer` | `Qwen3_5ForCausalLM` / `Qwen3_5ForConditionalGeneration` | ✓ | 9B end-to-end smoke PASSED (LoRA + full) | hybrid linear+full attention, dense MLP |
| Qwen3.5-MoE | `Qwen3_5FullLayer` + `Qwen3_5LinearLayer` (MoE FFN variant) | `Qwen3_5MoeForCausalLM` / `Qwen3_5MoeForConditionalGeneration` | ✓ | 35B-A3B end-to-end smoke PASSED (LoRA) | hybrid linear+full attention + MoE FFN (256+1 experts) |
| OLMoE | `OLMoEBlock` | `OlmoeForCausalLM` | ✓ | 1B-7B end-to-end PASSED, 1B-7B `from_pretrained` LoRA smoke PASSED | full-row QK-norm, softmax-then-topk routing |
| Qwen3-Next | `Qwen3NextLinearLayer` + `Qwen3NextFullLayer` | `Qwen3NextForCausalLM` | — (manual builder) | block-level fwd+bwd parity PASSED | alternating linear (Gated DeltaNet via FLA) and full attention |
| Mistral | `MistralBlock` | `MistralForCausalLM` | ✓ | small parity PASSED | sliding-window attention |

In-progress (not yet end-to-end): Gemma 2 / Gemma 3 (forward parity passes;
backward is stubbed — tracked in `docs/internal/`). GPT-OSS is deferred —
needs the attention-sink kernel.

For each architecture, the `flextrain/io/arch/<name>.py` file exposes:

* an `ArchSpec` registered at import time (used by `am.load_hf`);
* `hf_config_to_flextrain(hf_config)` — converts an HF config dict /
  `PretrainedConfig` into the FlexTrain `dims` map;
* `hf_config_to_hyperparams(hf_config)` — extracts per-layer
  hyperparameters (`rms_norm_eps`, `rope_theta`, `window_size_left`,
  `load_balance_coef`, `routing_mode`, ...).

To add a new architecture, see [implementing.md](implementing.md) and
[weights.md](weights.md).
