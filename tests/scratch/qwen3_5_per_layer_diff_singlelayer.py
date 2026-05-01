"""Per-layer FT-vs-HF diff that loads ONE HF layer at a time.

Goal: localize which layer of Qwen3.5-27B (or any Qwen3.5 dense)
diverges between FT and HF. The full HF model doesn't fit on a 24GB
GPU at 27B scale, so we load each `Qwen3_5DecoderLayer` standalone,
populate just its weights from the safetensors shards, run forward
on a captured input, and compare with FT's same-layer output.

Strategy:
    1. Bring up FT model via from_pretrained, run a small fwd, capture
       embed output and per-layer outputs (existing pattern from
       qwen3_5_per_layer_diff.py).
    2. Free FT's GPU compute usage but keep the captures + the HF
       config. Build HF's `Qwen3_5TextRotaryEmbedding` (cheap) and
       precompute (cos, sin) for our positions.
    3. For each layer L = 0..n_layers-1:
         a. Build a fresh `Qwen3_5DecoderLayer(config, L)` on GPU.
         b. Load its weights from the right safetensors shard via the
            file index.
         c. Run its forward on FT's captured (input to layer L) using
            the precomputed (cos, sin). Build a causal attention_mask
            (bias-style, additive 4D mask).
         d. Compare HF output to FT's captured layer-L output.
         e. Drop the HF layer; free memory.

Usage:
    PYTHONPATH=. python tests/qwen3_5_per_layer_diff_singlelayer.py \\
        --model models/Qwen3.5-27B \\
        --prompt "Four score and"

Notes:
    * mRoPE: HF uses 3-axis position ids; for text-only fwd we
      replicate the same text positions across all 3 axes so mRoPE
      collapses to standard RoPE. Same approach used in the training
      harness.
    * QK-norm + halved->pair: FT applies a head-internal halved->pair
      permutation to W_q / W_k / w_q_norm / w_k_norm at load time.
      We do NOT permute the HF-side layer (HF applies halved-RoPE
      natively); FT's permutation produces equivalent results.
    * w_q split: FT permutes per-head [q|gate] -> flat [Q|gate]; HF
      keeps per-head [q|gate]. FT's permutation is matched to its
      forward; we only need each side's outputs to match.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _build_ft_and_capture(model_path: str, prompt_ids: list[int]):
    """Build FT model, run forward, return (embed_input, per_layer_outputs,
    hf_config_dict)."""
    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.engine.schedule import prepare_training_chunks

    am = from_pretrained(
        model_path,
        optimizer=AdamW(
            AdamWHyperparams(
                lr=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
            ),
            state_dtype=torch.float32,
        ),
        max_seq_len=max(len(prompt_ids) + 8, 1024),
        max_global_batch_tokens=max(len(prompt_ids) + 8, 1024),
        max_gpu_mem_bytes=int(22.5 * (1 << 30)),
        max_host_mem_bytes=int(110.0 * (1 << 30)),
        device="cuda:0",
        leeway_gpu_mem_bytes=int(2 * (1 << 30)),
        leeway_host_mem_bytes=int(10 * (1 << 30)),
        # LoRA mode: frozen base, tiny adapters → smaller baseline
        # so 27B fits + we have budget for HF layer instantiation.
        # Step-1 fwd ≡ pretrained-base fwd because adapters init to 0.
        lora_targets="all",
        lora_rank=8,
        lora_alpha=8.0,
        strict=False, verbose=False,
    )

    # Capture per-layer outputs + the embed (= input to layer 0)
    captured_outputs: list[torch.Tensor | None] = [None] * len(am.backbone)
    captured_inputs: list[torch.Tensor | None] = [None] * len(am.backbone)
    original_forwards = []
    for i, layer in enumerate(am.backbone):
        orig = layer.forward
        original_forwards.append((layer, orig))

        def make_wrapped(idx, orig_fn):
            def wrapped(x, chunk_meta, weights, slot, ctx):
                captured_inputs[idx] = x.detach().clone()
                y = orig_fn(x, chunk_meta, weights, slot, ctx)
                captured_outputs[idx] = y.detach().clone()
                return y
            return wrapped

        layer.forward = make_wrapped(i, orig)

    class _Seq:
        def __init__(self, t):
            self.tokens = t
            T = len(t)
            self.targets = torch.zeros(T, dtype=torch.int64)
            self.per_token_loss = torch.zeros(T, dtype=torch.float32)
            self.seq_id = 0
        def __len__(self):
            return len(self.tokens)

    seq = _Seq(torch.tensor(prompt_ids, dtype=torch.int64))
    prepared = prepare_training_chunks(
        [seq], max_chunk_size=am.working_set.max_chunk_size,
        device=am.device, policy=am.chunk_policy,
    )
    am._allocate_moe_chunk_scratch(prepared)
    am.events.clear_per_round()
    plan = am._plan_save_levels(prepared)
    am.streams.compute.synchronize()
    am._setup_round(prepared, plan)
    am._forward_pass(prepared, plan)
    am.streams.compute.synchronize()

    for layer, orig in original_forwards:
        layer.forward = orig

    # Read raw HF config so we can construct Qwen3_5TextRotaryEmbedding etc.
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        full_hf_config = json.load(f)
    text_config = full_hf_config.get("text_config", full_hf_config)

    # Read shard index for layer-by-layer weight loading.
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    return am, captured_inputs, captured_outputs, text_config, weight_map


def _build_hf_layer(text_config: dict, layer_idx: int, device: str):
    """Build a single Qwen3_5DecoderLayer with empty weights."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
    )
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )
    cfg = Qwen3_5TextConfig(**text_config)
    layer = Qwen3_5DecoderLayer(cfg, layer_idx).to(
        device=device, dtype=torch.bfloat16,
    )
    layer.eval()
    return layer, cfg


def _load_layer_weights(
    layer, layer_idx: int, model_path: str, weight_map: dict,
):
    """Load weights from safetensors shards into the standalone HF
    layer. Walk the layer's state_dict and resolve each name to its
    safetensors shard."""
    from safetensors import safe_open
    state = layer.state_dict()
    # HF layer names are relative; we need to prepend
    # ``model.language_model.layers.{layer_idx}.``
    prefix = f"model.language_model.layers.{layer_idx}"

    # Group keys by shard for efficient open-once-per-shard.
    by_shard: dict[str, list[tuple[str, str]]] = {}
    for local_name in state.keys():
        full_name = f"{prefix}.{local_name}"
        shard = weight_map.get(full_name)
        if shard is None:
            print(f"  WARN: missing in weight_map: {full_name}")
            continue
        by_shard.setdefault(shard, []).append((local_name, full_name))

    new_state = {}
    for shard, names in by_shard.items():
        path = os.path.join(model_path, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            for local, fn in names:
                t = f.get_tensor(fn)
                new_state[local] = t.to(
                    dtype=state[local].dtype, device=state[local].device,
                )

    missing = set(state.keys()) - set(new_state.keys())
    if missing:
        print(f"  WARN: missing keys for layer {layer_idx}: {missing}")
    layer.load_state_dict(new_state, strict=False)


def _build_hf_position_embeddings(
    text_config: dict, prompt_ids: list[int], device: str,
):
    """Build (cos, sin) the way Qwen3_5TextModel does — using the
    text-only collapse of mRoPE (all 3 axes use the same text
    positions)."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5TextRotaryEmbedding,
    )
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5TextConfig,
    )
    cfg = Qwen3_5TextConfig(**text_config)
    rotary = Qwen3_5TextRotaryEmbedding(cfg).to(device=device)
    rotary.eval()

    T = len(prompt_ids)
    # 2D position_ids -> Qwen3_5TextRotaryEmbedding expands to (3, B, T)
    # with all three axes equal (text-only collapse).
    position_ids = torch.arange(T, dtype=torch.int64, device=device).unsqueeze(0)  # (1, T)
    # Need a dummy x with .device + .dtype attrs.
    dummy = torch.zeros(1, T, 1, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        cos, sin = rotary(dummy, position_ids)
    return cos, sin, position_ids


def _stats(name: str, ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float, float]:
    """Print + return (max_abs, rel, fraction_above_eps)."""
    if ref.shape != got.shape:
        print(f"  {name:14s} SHAPE MISMATCH ref={tuple(ref.shape)} got={tuple(got.shape)}")
        return float("inf"), float("inf"), float("inf")
    diff = (ref.float() - got.float()).abs()
    max_abs = float(diff.max().item())
    ref_norm = float(ref.float().norm().item())
    rel = max_abs / max(ref_norm, 1e-12)
    bf16_eps = 5e-3
    flag = "OK" if rel < bf16_eps else ("DIVERGE" if rel < 0.1 else "BAD")
    print(
        f"  {name:14s} max|Δ|={max_abs:9.4f}  rel={rel:.3e}  [{flag}]",
        flush=True,
    )
    return max_abs, rel, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen3.5-27B")
    ap.add_argument("--prompt", default="Four score and seven years ago our fathers")
    args = ap.parse_args()

    device = "cuda:0"

    # Tokenize. Use plain tokenization (no chat template) so we get a
    # multi-token sequence regardless of how chat templates are applied.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok.encode(args.prompt, add_special_tokens=True)
    prompt_ids = list(enc)
    T = len(prompt_ids)
    print(f"Prompt ({T} tokens): head={prompt_ids[:8]}...")
    print()

    print("=== Capturing FT activations ===")
    am, ft_inputs, ft_outputs, text_config, weight_map = _build_ft_and_capture(
        args.model, prompt_ids,
    )
    n_layers = len(am.backbone)
    layer_types = text_config.get("layer_types") or []
    print(f"  FT: {n_layers} layers, types head: {layer_types[:6]}...")

    # mRoPE position embeddings (precomputed once).
    cos, sin, position_ids = _build_hf_position_embeddings(
        text_config, prompt_ids, device,
    )

    # Causal attention mask (4D additive).
    # HF Qwen3_5Attention reads attention_mask from kwargs / position_ids
    # paths; we pass a None mask + position_ids = arange(T), and the
    # default flash_attn-compatible causal logic kicks in.
    # For sdpa fallback we may need an explicit mask; build it as
    # (1, 1, T, T) lower-triangular zero, upper -inf.
    attn_mask = torch.full(
        (1, 1, T, T), -1e9, dtype=torch.bfloat16, device=device,
    )
    attn_mask = torch.triu(attn_mask, diagonal=1)
    # zero out lower triangle (default value 0 means no penalty).
    attn_mask = attn_mask  # already (-inf above diag, 0 on/below).

    print()
    print("=== Per-layer FT vs HF ===")
    earliest_diverge = None
    for L in range(n_layers):
        if ft_inputs[L] is None or ft_outputs[L] is None:
            print(f"  layer {L:3d} ({layer_types[L]:18s}) : FT capture missing")
            continue
        # Build the HF layer fresh and load weights.
        try:
            hf_layer, cfg = _build_hf_layer(text_config, L, device)
            _load_layer_weights(hf_layer, L, args.model, weight_map)
        except Exception as e:
            print(f"  layer {L:3d} BUILD FAIL: {e}")
            continue

        # FT-side input is (T, d_model) (no batch). HF expects (B, T, d_model).
        ft_in = ft_inputs[L].to(torch.bfloat16)
        if ft_in.dim() == 2:
            hf_in = ft_in.unsqueeze(0).contiguous()
        else:
            hf_in = ft_in.contiguous()

        with torch.no_grad():
            try:
                hf_out = hf_layer(
                    hidden_states=hf_in,
                    position_embeddings=(cos, sin),
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                )
            except Exception as e:
                print(f"  layer {L:3d} ({layer_types[L]:18s}) HF FWD FAIL: {e}")
                del hf_layer
                torch.cuda.empty_cache()
                continue

        # Strip batch dim from HF output, compare with FT.
        hf_out = hf_out.squeeze(0).contiguous()
        ft_out = ft_outputs[L]
        max_abs, rel, _ = _stats(
            f"L{L:3d} ({layer_types[L][:8]})", hf_out, ft_out,
        )
        if earliest_diverge is None and rel > 5e-3:
            earliest_diverge = L

        del hf_layer
        torch.cuda.empty_cache()

    print()
    if earliest_diverge is None:
        print("  All layers within tolerance -- bug is in head/embed.")
    else:
        print(f"  EARLIEST DIVERGENCE: layer {earliest_diverge} "
              f"(type: {layer_types[earliest_diverge] if earliest_diverge < len(layer_types) else '?'})")

    print()
    print("=== Embed + final-norm + head check ===")

    # FT embed: read it via the engine.
    ft_embed_out = ft_inputs[0]  # (T, d_model)

    # HF embed: load on CPU (vocab is large; embed is ~2.4GiB at fp16),
    # gather only the rows we need, then move to GPU. Keeps GPU memory
    # available for the layer-wise HF instantiations.
    from safetensors import safe_open
    embed_name = "model.language_model.embed_tokens.weight"
    embed_shard = weight_map[embed_name]
    ids_t_cpu = torch.tensor(prompt_ids, dtype=torch.int64)
    with safe_open(os.path.join(args.model, embed_shard),
                   framework="pt", device="cpu") as f:
        embed_w_cpu = f.get_tensor(embed_name).to(dtype=torch.bfloat16)
    hf_embed_rows = embed_w_cpu[ids_t_cpu].to(device=device).contiguous()
    _stats("embed", hf_embed_rows, ft_embed_out)
    del embed_w_cpu, hf_embed_rows
    torch.cuda.empty_cache()

    # FT final-norm + head_proj output (logits).
    from flextrain.ops import flextrain_rmsnorm_fwd
    head_w = am.buffers.gpu_head_params
    rms_eps = float(am.head.cfg.rms_norm_eps)
    ft_post_layer_out = ft_outputs[n_layers - 1]  # (T, d_model)
    ft_norm_out, _rstd = flextrain_rmsnorm_fwd(
        ft_post_layer_out, W=head_w["w_final_norm"], rms_norm_eps=rms_eps,
    )
    ft_logits = torch.mm(ft_norm_out, head_w["w_head_proj"])

    # HF final-norm + lm_head:
    # final norm γ shift: HF's Qwen3_5RMSNorm does ``output * (1 + weight)``
    # so weights stored as γ-1. FT's loader shifts +1 at load time. The
    # FT norm weight is therefore the canonical γ; we should compare HF's
    # output of (1 + γ_stored) * RMSNorm(x) which equals FT's
    # γ_canonical * RMSNorm(x). They should match.
    final_norm_name = "model.language_model.norm.weight"
    final_norm_shard = weight_map[final_norm_name]
    with safe_open(os.path.join(args.model, final_norm_shard),
                   framework="pt", device="cpu") as f:
        final_norm_w_hf = f.get_tensor(final_norm_name).to(
            dtype=torch.bfloat16, device=device,
        )
    # HF's "stored" weight = γ_canonical - 1. FT's loaded = γ_canonical.
    final_norm_w_canonical = final_norm_w_hf + 1.0

    head_proj_name = "lm_head.weight"
    head_proj_shard = weight_map.get(head_proj_name)
    if head_proj_shard is None:
        print("  lm_head.weight missing from index — tied?")
        return 0

    # HF lm_head: shape (vocab, hidden). Load to CPU; do head projection
    # on CPU then move logits-only to GPU (T*vocab much smaller than
    # vocab*hidden at T=31).
    with safe_open(os.path.join(args.model, head_proj_shard),
                   framework="pt", device="cpu") as f:
        head_proj_w_cpu = f.get_tensor(head_proj_name).to(dtype=torch.bfloat16)

    # Compute HF reference final-norm + logits.
    # rmsnorm: (x * rsqrt(mean(x^2) + eps)) * γ_canonical, in float
    final_norm_w_canonical_cpu = final_norm_w_canonical.to("cpu")
    ft_post_layer_out_cpu = ft_post_layer_out.to("cpu")
    x_f = ft_post_layer_out_cpu.float()
    rms = (x_f.pow(2).mean(dim=-1, keepdim=True) + rms_eps).rsqrt()
    normed_cpu = (x_f * rms).to(torch.bfloat16) * final_norm_w_canonical_cpu
    hf_logits_cpu = torch.mm(normed_cpu, head_proj_w_cpu.t())

    _stats("final_norm_x", normed_cpu.to(device), ft_norm_out)
    _stats("logits", hf_logits_cpu.to(device), ft_logits)

    # Argmax of last-token logits
    hf_arg = int(hf_logits_cpu[-1].argmax().item())
    ft_arg = int(ft_logits[-1].argmax().item())
    from transformers import AutoTokenizer
    tok2 = AutoTokenizer.from_pretrained(args.model)
    print(f"  HF predicts: {hf_arg} ({tok2.decode([hf_arg])!r})")
    print(f"  FT predicts: {ft_arg} ({tok2.decode([ft_arg])!r})")

    # Compute next-token CE loss from both logit sources for direct
    # comparison against training-loop FT step-1 loss.
    print()
    print("=== Cross-entropy loss vs prompt tokens ===")
    # Predict ids[1:] from logits[:-1].
    targets = torch.tensor(prompt_ids[1:], dtype=torch.int64, device=device)
    pred_ft = ft_logits[:-1, :].float()
    ce_ft = torch.nn.functional.cross_entropy(
        pred_ft, targets, reduction="mean",
    )
    pred_hf = hf_logits_cpu[:-1, :].float().to(device)
    ce_hf = torch.nn.functional.cross_entropy(
        pred_hf, targets, reduction="mean",
    )
    print(f"  HF CE loss: {ce_hf.item():.4f}")
    print(f"  FT CE loss: {ce_ft.item():.4f}")

    del head_proj_w_cpu, normed_cpu, hf_logits_cpu, final_norm_w_hf
    torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
