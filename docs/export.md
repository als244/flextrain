# Export

After fine-tuning a model with FlexTrain, use `flextrain.export` to ship
it to a serving engine. There are three modes:

| function | when to use | output | serving |
|---|---|---|---|
| `save_hf_full` | after a full fine-tune (no LoRA) | sharded HF dir, full base size | vLLM, sGLang, transformers |
| `save_lora_adapter` | after a LoRA fine-tune on Llama / Qwen3-dense | small adapter dir (rank-r) | vLLM `--enable-lora`, sGLang `--lora-paths`, PEFT |
| `save_hf_merged` | after a LoRA fine-tune on any other arch | sharded HF dir, full base size | vLLM, sGLang, transformers (no LoRA support needed) |

All three write a self-contained directory: weights, tokenizer,
config.json, generation_config.json, and (where present) chat template.
You can hand the path directly to a serving engine.

## Quickstart

```python
from flextrain import from_pretrained
from flextrain.export import save_hf_full, save_hf_merged, save_lora_adapter
from flextrain.optim.adamw import AdamW, AdamWHyperparams

am = from_pretrained(
    "models/Llama-3.1-8B",
    optimizer=AdamW(AdamWHyperparams(lr=1e-4)),
    max_seq_len=2048, max_global_batch_tokens=2048,
    max_gpu_mem_bytes=int(24 * (1 << 30)),
    max_host_mem_bytes=int(110 * (1 << 30)),
    lora_targets="all", lora_rank=16, lora_alpha=16.0,
)

# ... your training loop ...

# Three export options:
save_hf_full(am, "out/llama-base")              # full base (LoRA dropped)
save_hf_merged(am, "out/llama-merged")          # base + LoRA delta merged in
save_lora_adapter(am, "out/llama-adapter")      # PEFT-format adapter
```

For a runnable example, see [`examples/finetune_and_export.py`](../examples/finetune_and_export.py).

## Choosing the right mode

### Full fine-tune → `save_hf_full`

After `--mode full` (no LoRA), the host master params hold the
fine-tuned weights. `save_hf_full` writes them directly. The output is
a normal HF checkpoint that any serving engine loads as a base model.

### LoRA fine-tune → `save_hf_merged` (universal)

`save_hf_merged` folds the LoRA delta into the base weights in place
(`W += scale * A @ B`, including per-expert merge for MoE 3-D adapter
stacks), then writes a normal HF dir. Universal — works for every
architecture FlexTrain trains.

Trade-off: output is the size of the full base model (e.g. ~17 GiB for
an 8B bf16 model), and you can't hot-swap multiple adapters at serve
time. But the serving engine doesn't need any LoRA support and treats
the model as ordinary base weights.

Pass `keep_lora_after_merge=True` if you need to keep training the
adapter after exporting (otherwise the in-memory state holds the
merged weights and zero'd A/B factors).

### LoRA fine-tune → `save_lora_adapter` (PEFT format)

`save_lora_adapter` writes a small (rank-r) adapter directory in HF
PEFT format:

```
out/llama-adapter/
  adapter_config.json          # PEFT schema: r, lora_alpha, target_modules, base_model_name_or_path, ...
  adapter_model.safetensors    # base_model.model.model.layers.{i}.<module>.lora_{A,B}.weight
  tokenizer.json, ...          # copied for convenience
```

Serving engines load this on top of an unmodified base model:

```bash
# vLLM
vllm serve models/Llama-3.1-8B \
  --enable-lora --max-loras 1 --max-lora-rank 16 \
  --lora-modules my=out/llama-adapter

# sGLang
python -m sglang.launch_server \
  --model-path models/Llama-3.1-8B \
  --lora-paths my=out/llama-adapter

# Python with HF + PEFT
from transformers import AutoModelForCausalLM
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("models/Llama-3.1-8B")
model = PeftModel.from_pretrained(base, "out/llama-adapter")
```

**Architecture support** — currently:
| arch family | adapter export | merged export |
|---|---|---|
| Llama | ✅ | ✅ |
| Qwen3 (dense) | ✅ | ✅ |
| Qwen3.5 / Qwen3.5-MoE / Qwen3.6 | ❌ (gated q_proj layout doesn't fit PEFT) | ✅ |
| OLMoE / Qwen3-MoE / Qwen3-Next | ❌ (per-expert MoE adapter not in PEFT spec) | ✅ |

For non-supported archs, `save_lora_adapter` raises with a clear
message pointing you at `save_hf_merged`. The Qwen3.5 limitation is
because HF's `q_proj` for Qwen3.5 is a doubled-output gated projection
(per-head `[q | gate]`); PEFT's adapter format expects a plain linear,
so a rank-r delta on the FlexTrain-side `w_q` doesn't roundtrip.

**Per-layer LoRA target filtering** — for hybrid backbones (Qwen3.5
linear-attn layers can't take attn-only LoRA targets), the loader
auto-filters per-layer. The export skips any layer that didn't get
LoRA wrapping; `save_hf_merged` is safe regardless.

## Correctness validation

`tests/test_export_correctness.py` is the end-to-end check: it loads
a model, runs a few LoRA steps, exports all three ways, then reloads
each via subprocess (clean CUDA context) using HF transformers (and
PEFT for the adapter case) and verifies the top-1 next-token id
matches FlexTrain in-memory. Currently passing for:

* Llama-3.2-1B (dense, halved→pair RoPE permutation)
* Qwen3-1.7B (dense, q_norm/k_norm permutation on top of RoPE)

Run it for your own model:

```bash
python tests/test_export_correctness.py --model models/Llama-3.1-8B \
    --n-steps 50 --rank 16
```

## Implementation notes

* **RoPE permutation** — FlexTrain stores Q/K in pair-interleaved
  layout (even/odd channel = cos/sin) for its RoPE kernel; HF stores
  them in halved layout (first half / second half = cos/sin). The
  exporter inverts the permutation on Q / K weight axis 0 (and on
  LoRA-B's matching axis). Same permutation is undone on q_norm /
  k_norm for Qwen3-dense.

* **Tied embeddings** — Llama-3.2 / Qwen3 with
  `tie_word_embeddings: true` get the head's `lm_head.weight` mirrored
  from the embedding table at load time. The exporter writes both
  back through the ArchSpec so reload is consistent regardless of the
  original tie state.

* **Sharded output** — the exporter splits at 5 GiB per shard
  (matching HF's default), producing `model-00001-of-NNNNN.safetensors`
  and a `model.safetensors.index.json` when needed. For models ≤ 5 GiB
  it writes a single `model.safetensors`.

* **MoE expert merge** — for per-expert LoRA (e.g. OLMoE,
  Qwen3.5-MoE), `save_hf_merged` does per-expert `bmm`:
  `W'[e] = W[e] + scale * A[e] @ B[e]` for each routed/shared expert,
  in fp32 on CPU to avoid loading the large 3-D expert stack onto GPU.
  The merged HF dir stores experts as the standard
  `experts.{e}.gate_proj/up_proj/down_proj` layout, so vLLM /
  transformers load them with no MoE-LoRA support required.

## Known limitations

* `save_lora_adapter` doesn't yet support Qwen3.5 / Qwen3.5-MoE /
  Qwen3.6 / Qwen3-Next. Use `save_hf_merged`.
* `save_hf_full` after a full-FT on hybrid linear+full architectures
  (Qwen3.5 etc.) is best-effort; the gated-q_proj inversion isn't
  wired. For now, full-FT on these archs and reload via HF works only
  if you also patch the q_proj layout — `save_hf_merged` (which goes
  through a LoRA-zero base + merge) avoids this entirely.
