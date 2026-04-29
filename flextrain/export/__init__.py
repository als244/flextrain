"""Export FlexTrain checkpoints into formats serving engines load.

After a full or LoRA fine-tune, call one of the three top-level helpers
below to produce a directory that vLLM, sGLang, and ``transformers``
load directly:

* :func:`save_hf_full` — full base weights as a sharded HF checkpoint
  (``config.json`` + tokenizer + ``model.safetensors``). Use this after
  a full fine-tune, or after :func:`merge_lora_into_base` for a
  merged-LoRA export.

* :func:`save_lora_adapter` — PEFT-format adapter directory
  (``adapter_config.json`` + ``adapter_model.safetensors``). Use this
  to ship a small (rank-r) artifact that vLLM / sGLang / PEFT load on
  top of a stock base model.

  Currently supported for: ``LlamaForCausalLM``, ``Qwen3ForCausalLM``.
  For other architectures (notably the gated-q_proj Qwen3.5 family and
  per-expert MoE adapters), use :func:`save_hf_merged` instead.

* :func:`save_hf_merged` — fold the LoRA delta into the base weights
  in place, then call :func:`save_hf_full`. Universal: works for every
  architecture FlexTrain can train. Output is full base size; no
  adapter hot-swap support.

Quick reference
---------------
::

    from flextrain.export import save_hf_full, save_lora_adapter, save_hf_merged

    # Full fine-tune → standard HF checkpoint
    save_hf_full(am, "out/llama3-ft")

    # LoRA fine-tune (Llama / Qwen3-dense): PEFT adapter
    save_lora_adapter(am, "out/llama3-lora-adapter")

    # LoRA fine-tune (any arch): merged base
    save_hf_merged(am, "out/qwen35-lora-merged")
"""

from ._hf_full import save_hf_full
from ._lora_adapter import save_lora_adapter
from ._merged import merge_lora_into_base, save_hf_merged

__all__ = [
    "save_hf_full",
    "save_lora_adapter",
    "save_hf_merged",
    "merge_lora_into_base",
]
