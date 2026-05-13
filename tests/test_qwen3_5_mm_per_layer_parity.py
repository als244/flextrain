"""Per-layer LM forward parity HF vs flextrain (multimodal).

For each LM decoder layer i in 0..23, capture the post-residual hidden
state at position-level and compare HF vs flextrain with multiple
metrics:

* per-token cos sim (min / mean / max)
* abs error (max / mean)
* sign-match rate (fraction of elements with same sign)
* mean/std per stack
* relative error magnitude

This identifies WHERE the multimodal forward drift compounds. If
layer 0 already shows large drift -> input to LM differs (splice or
encoder issue, even if encoder is "byte-exact" standalone). If layer
0 matches but a specific layer N shows a jump -> a per-layer bug
specific to multimodal (MRoPE rotation, etc).

Per the precision-drift feedback rule: assume bug until proven
otherwise. cos > 0.999 per layer is the target (matches Gemma3/4
parity standards). Anything less needs investigation.

Run: ``./run_with_env.sh python tests/test_qwen3_5_mm_per_layer_parity.py``
"""

from __future__ import annotations

import os
import sys
import gc

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-2B")


def _layer_metrics(hf_h: torch.Tensor, ft_h: torch.Tensor) -> dict:
    """Per-layer comparison metrics. Both inputs shape (T, H) fp32 CPU."""
    diff = (hf_h - ft_h).abs()
    cos = torch.cosine_similarity(hf_h, ft_h, dim=-1)
    sign_match = ((hf_h.sign() == ft_h.sign()).float().mean().item())
    hf_norm = hf_h.norm(dim=-1)
    ft_norm = ft_h.norm(dim=-1)
    return {
        "cos_min": float(cos.min()),
        "cos_mean": float(cos.mean()),
        "abs_max": float(diff.max()),
        "abs_mean": float(diff.mean()),
        "sign_match": sign_match,
        "hf_mean": float(hf_h.mean()),
        "hf_std": float(hf_h.std()),
        "ft_mean": float(ft_h.mean()),
        "ft_std": float(ft_h.std()),
        "rel_err_mean": float((diff / (hf_h.abs() + 1e-6)).mean()),
    }


def _print_metrics(name: str, m: dict) -> None:
    print(
        f"  {name:>16s}: cos={m['cos_min']:.4f}/{m['cos_mean']:.4f}  "
        f"abs={m['abs_max']:.2e}/{m['abs_mean']:.2e}  "
        f"sign={m['sign_match']:.4f}  "
        f"std hf={m['hf_std']:.3f} ft={m['ft_std']:.3f}  "
        f"rel_err={m['rel_err_mean']:.3e}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available."); return
    if not os.path.isdir(MODEL_PATH):
        print(f"SKIP: {MODEL_PATH} not present."); return

    from flextrain.io.image_processing import (
        MultimodalProcessorBundle, build_multimodal_sequence,
    )
    import numpy as np
    from PIL import Image

    bundle = MultimodalProcessorBundle.from_pretrained(MODEL_PATH)
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "What is this?"}],
    }]
    chat = bundle.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    seq = build_multimodal_sequence(chat, [img], bundle)
    print(f"[setup] tokens={seq.tokens.numel()}, "
          f"image_placeholders={(seq.tokens == bundle.image_token_id).sum().item()}")

    # ----- HF: capture per-layer hidden states -----
    print("[hf] loading + hooking layers...")
    from transformers import AutoModelForImageTextToText
    hf_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()

    enc = bundle.processor(
        text=[chat], images=[img], return_tensors="pt", add_special_tokens=False,
    )
    enc = {k: v.to("cuda") for k, v in enc.items()}

    hf_per_layer: list[torch.Tensor] = []
    def hf_hook(_mod, _inp, out):
        # out is the layer's hidden_states (post residual). Could be a
        # tuple in older HF; in 5.7 it's just the tensor.
        h = out if isinstance(out, torch.Tensor) else out[0]
        hf_per_layer.append(h.detach().float().cpu().clone())

    n_layers = len(hf_model.model.language_model.layers)
    handles = [
        layer.register_forward_hook(hf_hook)
        for layer in hf_model.model.language_model.layers
    ]
    # Also capture the inputs_embeds (post image scatter) to verify
    # that's matching before the LM begins.
    inputs_embeds_capture: list[torch.Tensor] = []
    def embed_hook(_mod, inp, _out):
        # text_model.forward receives inputs_embeds as kwarg; capture from kwarg.
        # Simpler: capture inputs_embeds passed into the LM via a hook on
        # the LM's first layer's input.
        pass
    with torch.inference_mode():
        out = hf_model(**enc)
    for h in handles:
        h.remove()
    print(f"[hf] captured {len(hf_per_layer)} layer outputs; shape={tuple(hf_per_layer[0].shape)}")

    # Also capture inputs_embeds (post image scatter) manually for the
    # pre-LM-layer comparison.
    with torch.inference_mode():
        inputs_embeds = hf_model.model.get_input_embeddings()(enc["input_ids"])
        image_outputs = hf_model.model.get_image_features(
            enc["pixel_values"], enc["image_grid_thw"], return_dict=True,
        )
        image_embeds = image_outputs.pooler_output
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = torch.cat(list(image_embeds), dim=0)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = hf_model.model.get_placeholder_mask(
            enc["input_ids"], inputs_embeds=inputs_embeds, image_features=image_embeds,
        )
        hf_inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
    hf_pre_lm = hf_inputs_embeds[0].detach().float().cpu()
    print(f"[hf] pre-LM inputs_embeds shape: {tuple(hf_pre_lm.shape)}")

    del hf_model
    torch.cuda.empty_cache(); gc.collect()

    # ----- Flextrain: build + hook each layer to capture transition output -----
    print("[ft] loading + hooking layers...")
    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.io.sequence import Sequence

    am = from_pretrained(
        MODEL_PATH,
        optimizer=AdamW(AdamWHyperparams(lr=3e-5)),
        max_seq_len=2048, max_global_batch_tokens=4096,
        max_gpu_mem_bytes=30 * (1 << 30), max_host_mem_bytes=120 * (1 << 30),
        leeway_gpu_mem_bytes=3 * (1 << 30),
        enable_multimodal=True, freeze_modality_encoders=True,
        lora_targets="all", verbose=False,
    )

    ft_per_layer: list[torch.Tensor] = []
    # Wrap each backbone layer's forward to snapshot output AFTER the
    # layer call. Output IS the new transition residual stream
    # (same buffer as input, modified in place; clone needed).
    for layer in am.backbone:
        original_forward = layer.forward
        def make_wrapped(orig, idx):
            def wrapped(x, chunk, weights, slot, ctx):
                out = orig(x, chunk, weights, slot, ctx)
                # Snapshot a CPU copy. Note: ``out`` may be the same
                # buffer as x (in-place residual), so clone immediately.
                ft_per_layer.append(out.detach().clone().to("cpu", dtype=torch.float32))
                return out
            return wrapped
        layer.forward = make_wrapped(original_forward, layer.layer_id)

    # Also capture the post-splice embedding (pre-LM input).
    ft_pre_lm_capture: list[torch.Tensor] = []
    original_embed_fwd = am.embed.forward
    def wrapped_embed_fwd(token_ids, chunk, weights, ctx):
        out = original_embed_fwd(token_ids, chunk, weights, ctx)
        ft_pre_lm_capture.append(out.detach().clone().to("cpu", dtype=torch.float32))
        return out
    am.embed.forward = wrapped_embed_fwd

    ft_seq = Sequence(
        tokens=seq.tokens, targets=torch.roll(seq.tokens, -1),
        modality_inputs={"image": seq.modality_inputs["image"]},
    )
    print("[ft] running fwd_bwd...")
    _ = am.fwd_bwd([ft_seq])

    print(f"[ft] captured {len(ft_per_layer)} layer outputs; "
          f"shape={tuple(ft_per_layer[0].shape) if ft_per_layer else None}")
    ft_pre_lm = ft_pre_lm_capture[0]
    print(f"[ft] pre-LM embedding shape: {tuple(ft_pre_lm.shape)}")

    # ----- Compare -----
    print("\n=== Per-layer parity (HF vs flextrain, fp32 cosine) ===")
    print(f"{'name':>18s}: cos_min/cos_mean    abs_max/abs_mean    "
          f"sign_match   std_hf/std_ft   rel_err_mean")
    print("-" * 120)
    pre_metrics = _layer_metrics(hf_pre_lm, ft_pre_lm)
    _print_metrics("pre_LM_input", pre_metrics)
    for i in range(n_layers):
        m = _layer_metrics(hf_per_layer[i], ft_per_layer[i])
        _print_metrics(f"layer_{i:02d}", m)


if __name__ == "__main__":
    main()
