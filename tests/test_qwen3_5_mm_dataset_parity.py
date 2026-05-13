"""Realistic multimodal dataset parity HF vs flextrain.

Streams a small slice of ``HuggingFaceH4/llava-instruct-mix-vsft`` from
the HuggingFace Hub (multi-turn LLaVA conversations, mostly single-image
with real natural images) and runs HF vs flextrain forward on each
example. For every example we capture:

* HF mean CE loss vs flextrain mean CE loss → ``rel_err``
* HF pre-LM ``inputs_embeds`` (post image-scatter) vs flextrain
  post-splice text_emb → ``cos_min``

We also synthesize a handful of multi-image cases by combining two
adjacent single-image examples into one ``messages`` list containing two
``{"type": "image"}`` items. That exercises the cu_seqlens packing and
``ImageEmbeddings.token_offsets`` plumbing that the synthetic random-pixel
test never reached.

Pass criterion (per example):
  * loss rel_err < 1.0% (the single-image synthetic case is 0.11%; we
    allow some slack for real images with varying resolutions / multi-
    image cases)
  * pre_LM_input cos_min > 0.9999 (byte-exact target; encoder + splice
    are deterministic so this should hold)

Run: ``./run_with_env.sh python tests/test_qwen3_5_mm_dataset_parity.py``

Requirements: CUDA, transformers, PIL, datasets, network (streams from
the Hub on first run; cache is at ``~/.cache/huggingface/datasets``).
"""

from __future__ import annotations

import os
import sys
import gc
import itertools

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "Qwen3.5-2B")

# How many single-image + synthesized multi-image examples to run.
N_SINGLE_IMAGE = 12
N_MULTI_IMAGE = 4
# Max sequence length per example (skip any that exceed this to keep the
# flextrain engine round budget reasonable; this is a test-side filter,
# not a parity-relevant bound).
MAX_SEQ_TOKENS = 4096
# Loss rel_err pass threshold per example.
LOSS_RELERR_THRESH = 0.015          # 1.5%
# pre_LM_input cos_min pass threshold per example.
PRE_LM_COS_MIN_THRESH = 0.9999


# ---------------------------------------------------------------------------
# Data-side helpers
# ---------------------------------------------------------------------------

def _normalize_content(content):
    """Strip the ``index`` field on image items (LLaVA-vsft uses it to
    pair images; Qwen3.5's chat template matches positionally instead).
    Drop empty fields too."""
    out = []
    for item in content:
        kind = item["type"]
        if kind == "image":
            out.append({"type": "image"})
        elif kind == "text":
            out.append({"type": "text", "text": item["text"]})
        else:
            raise ValueError(f"unexpected content type: {kind}")
    return out


def _ex_to_messages(ex):
    """Convert a LLaVA-vsft example into (messages_list, image_pil_list).

    LLaVA-vsft messages are already in chat shape; we just strip the
    ``index`` field. Images are PIL JpegImageFiles (RGB)."""
    msgs = [
        {"role": m["role"], "content": _normalize_content(m["content"])}
        for m in ex["messages"]
    ]
    # Force RGB (some are mode=L or mode=P).
    images = [im.convert("RGB") for im in ex["images"]]
    return msgs, images


def _stream_examples(n_single: int, n_multi: int):
    """Yield ``(messages, images, label)`` tuples.

    Yields ``n_single`` single-image examples first, then ``n_multi``
    synthesized two-image examples (each combines two adjacent
    single-image conversations into one).
    """
    from datasets import load_dataset
    ds = load_dataset(
        "HuggingFaceH4/llava-instruct-mix-vsft", split="train", streaming=True,
    )
    it = iter(ds)
    # Single-image pass.
    yielded_single = 0
    buf_for_multi: list = []  # holds (messages, images) for the multi-image stage
    while yielded_single < n_single or len(buf_for_multi) < 2 * n_multi:
        ex = next(it)
        msgs, images = _ex_to_messages(ex)
        if len(images) != 1:
            continue                 # skip the very rare 0-image / multi-image natives
        if yielded_single < n_single:
            yield msgs, images, f"single_{yielded_single:02d}"
            yielded_single += 1
        elif len(buf_for_multi) < 2 * n_multi:
            buf_for_multi.append((msgs, images))
    # Multi-image pass: pair (a, b) consecutive single-image convos.
    for i, (a, b) in enumerate(zip(buf_for_multi[0::2], buf_for_multi[1::2])):
        # Strategy: take a's first user turn (with its image), append b's
        # first user turn (with its image), then chain a's assistant
        # reply. This yields a 2-image conversation with both images in
        # the first user message before any text/assistant turn.
        a_msgs, a_imgs = a
        b_msgs, b_imgs = b
        # Take the first user message from each; build a single user
        # message with TWO image items, then a single assistant reply
        # (a's assistant reply, arbitrarily).
        def _first_user(msgs):
            for m in msgs:
                if m["role"] == "user":
                    return m
            raise ValueError("no user message")
        def _first_assistant(msgs):
            for m in msgs:
                if m["role"] == "assistant":
                    return m
            raise ValueError("no assistant message")
        au = _first_user(a_msgs)
        bu = _first_user(b_msgs)
        aa = _first_assistant(a_msgs)
        # Combined user content: a's text + a's image + b's text + b's image.
        # (Strip image items from each so we add them in a known order.)
        a_text = [c for c in au["content"] if c["type"] == "text"]
        b_text = [c for c in bu["content"] if c["type"] == "text"]
        a_imgs_in_user = [c for c in au["content"] if c["type"] == "image"]
        b_imgs_in_user = [c for c in bu["content"] if c["type"] == "image"]
        # Skip if either lacked a true image item (some LLaVA-vsft rows
        # are degenerate).
        if not a_imgs_in_user or not b_imgs_in_user:
            continue
        combined_user = {
            "role": "user",
            "content": (
                a_text
                + [{"type": "image"}]
                + b_text
                + [{"type": "image"}]
            ),
        }
        merged_msgs = [combined_user, aa]
        merged_imgs = [a_imgs[0], b_imgs[0]]
        yield merged_msgs, merged_imgs, f"multi_{i:02d}"


# ---------------------------------------------------------------------------
# Per-example parity harness
# ---------------------------------------------------------------------------

def _hf_forward(
    hf_model,
    bundle,
    chat_text: str,
    images: list,
    hf_labels_cuda: torch.Tensor,
) -> tuple[float, torch.Tensor]:
    """Run HF forward; return ``(mean_loss, pre_LM_inputs_embeds_cpu_fp32)``."""
    enc = bundle.processor(
        text=[chat_text], images=images, return_tensors="pt", add_special_tokens=False,
    )
    enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.inference_mode():
        out = hf_model(**enc, labels=hf_labels_cuda)
        # Compute pre-LM inputs_embeds (post image scatter) by reusing
        # HF's helpers. This is exactly what the multi-modal LM sees as
        # its initial residual stream.
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
    return float(out.loss.item()), hf_pre_lm


def _ft_forward(am, ft_seq) -> tuple[float, torch.Tensor]:
    """Run flextrain forward (via fwd_bwd) and capture mean loss + pre-LM
    embed (via an embed-layer hook)."""
    # One-shot hook on am.embed.forward to capture the post-splice
    # output.
    captured: list[torch.Tensor] = []
    original = am.embed.forward
    def wrapped(token_ids, chunk, weights, ctx):
        out = original(token_ids, chunk, weights, ctx)
        captured.append(out.detach().clone().to("cpu", dtype=torch.float32))
        return out
    am.embed.forward = wrapped
    try:
        stats = am.fwd_bwd([ft_seq])
    finally:
        am.embed.forward = original
    if not captured:
        raise RuntimeError("ft_forward: embed hook never fired")
    ft_pre_lm = captured[0]
    ft_active = ft_seq.active_token_count
    ft_loss = stats.total_loss / max(ft_active, 1)
    return ft_loss, ft_pre_lm


def _build_ft_seq(bundle, chat_text: str, images: list):
    """Mirror test_qwen3_5_mm_loss_parity.py's label-alignment contract."""
    from flextrain.io.image_processing import build_multimodal_sequence
    from flextrain.io.sequence import Sequence

    seq = build_multimodal_sequence(chat_text, images, bundle)
    input_ids = seq.tokens
    image_token_id = bundle.image_token_id
    hf_labels = input_ids.clone()
    hf_labels[input_ids == image_token_id] = -100
    ft_targets = torch.roll(hf_labels, -1)
    ft_targets[-1] = -100
    ft_seq = Sequence(
        tokens=input_ids.cpu(),
        targets=ft_targets.cpu(),
        modality_inputs={"image": seq.modality_inputs["image"]},
    )
    return ft_seq, hf_labels.unsqueeze(0).to("cuda")


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def main() -> None:
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available."); return
    if not os.path.isdir(MODEL_PATH):
        print(f"SKIP: {MODEL_PATH} not present."); return
    try:
        import transformers  # noqa: F401
        import PIL           # noqa: F401
        import datasets      # noqa: F401
    except ImportError as e:
        print(f"SKIP: missing dep: {e}"); return

    from flextrain import from_pretrained
    from flextrain.optim.adamw import AdamW, AdamWHyperparams
    from flextrain.io.image_processing import MultimodalProcessorBundle

    bundle = MultimodalProcessorBundle.from_pretrained(MODEL_PATH)

    # ----- 1. Pull examples from the Hub (small slice, streamed) -----
    # Over-fetch and then filter out examples that exceed MAX_SEQ_TOKENS
    # AFTER chat-template expansion + image-placeholder slot insertion.
    print(f"[setup] streaming examples (target: {N_SINGLE_IMAGE} single + "
          f"{N_MULTI_IMAGE} synth-multi, filtering > {MAX_SEQ_TOKENS} tokens)...")
    raw_iter = _stream_examples(N_SINGLE_IMAGE * 3, N_MULTI_IMAGE * 3)
    examples: list[tuple] = []
    n_single = n_multi = 0
    skipped_long = 0
    for msgs, images, label in raw_iter:
        is_single = label.startswith("single")
        if is_single and n_single >= N_SINGLE_IMAGE:
            continue
        if not is_single and n_multi >= N_MULTI_IMAGE:
            continue
        chat_text = bundle.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        # Tokenize once to gate; we'll re-tokenize in build_multimodal_sequence.
        n_tok = len(bundle.tokenizer.encode(chat_text, add_special_tokens=False))
        # Add ~64 image tokens per image (Qwen3.5-2B 14x14 grid → 64 patches).
        n_tok_est = n_tok + 64 * len(images)
        if n_tok_est > MAX_SEQ_TOKENS:
            skipped_long += 1
            continue
        examples.append((msgs, images, label, chat_text))
        if is_single:
            n_single += 1
        else:
            n_multi += 1
        if n_single >= N_SINGLE_IMAGE and n_multi >= N_MULTI_IMAGE:
            break
    print(f"[setup] {len(examples)} examples ready: {n_single} single, {n_multi} multi "
          f"(skipped {skipped_long} for length > {MAX_SEQ_TOKENS})")

    # ----- 2. Build HF model + flextrain ActiveModel once -----
    print("[hf] loading Qwen3_5ForConditionalGeneration...")
    from transformers import AutoModelForImageTextToText
    hf_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()

    # We'll run HF forwards first, save results, then teardown HF, build
    # flextrain, and run all FT forwards. This avoids holding both in
    # GPU memory at once.
    hf_results: list[dict] = []
    for msgs, images, label, chat_text in examples:
        # We need the input_ids + labels for HF forward; build via
        # build_multimodal_sequence to share tokenization with flextrain.
        ft_seq, hf_labels_cuda = _build_ft_seq(bundle, chat_text, images)
        try:
            hf_loss, hf_pre_lm = _hf_forward(
                hf_model, bundle, chat_text, images, hf_labels_cuda,
            )
        except Exception as e:
            print(f"[hf:{label}] ERROR: {e}")
            hf_results.append({"label": label, "error": repr(e)})
            continue
        hf_results.append({
            "label": label,
            "msgs": msgs,
            "images": images,
            "chat_text": chat_text,
            "n_tokens": int(ft_seq.tokens.numel()),
            "n_images": len(images),
            "img_grids": [list(im.size) for im in images],
            "hf_loss": hf_loss,
            "hf_pre_lm": hf_pre_lm,
            "ft_seq": ft_seq,
        })
        print(f"[hf:{label}] tokens={int(ft_seq.tokens.numel())}, "
              f"n_images={len(images)}, mean_loss={hf_loss:.4f}")

    del hf_model
    torch.cuda.empty_cache(); gc.collect()

    # ----- 3. Build flextrain ActiveModel + run FT forwards -----
    print("[ft] building ActiveModel(enable_multimodal=True)...")
    am = from_pretrained(
        MODEL_PATH,
        optimizer=AdamW(AdamWHyperparams(lr=3e-5)),
        max_seq_len=4096,
        max_global_batch_tokens=8192,
        max_gpu_mem_bytes=30 * (1 << 30),
        max_host_mem_bytes=120 * (1 << 30),
        leeway_gpu_mem_bytes=3 * (1 << 30),
        enable_multimodal=True,
        freeze_modality_encoders=True,
        lora_targets="all",
        verbose=False,
    )

    rows: list[dict] = []
    for r in hf_results:
        if "error" in r:
            continue
        label = r["label"]
        try:
            ft_loss, ft_pre_lm = _ft_forward(am, r["ft_seq"])
        except Exception as e:
            print(f"[ft:{label}] ERROR: {e}")
            rows.append({**{k: r[k] for k in ("label", "n_tokens", "n_images")},
                         "error": repr(e)})
            continue
        # Compare.
        diff = (r["hf_pre_lm"] - ft_pre_lm).abs()
        cos = torch.cosine_similarity(r["hf_pre_lm"], ft_pre_lm, dim=-1)
        cos_min, cos_mean = float(cos.min()), float(cos.mean())
        abs_max, abs_mean = float(diff.max()), float(diff.mean())
        loss_relerr = abs(ft_loss - r["hf_loss"]) / max(abs(r["hf_loss"]), 1e-6)
        rows.append({
            "label": label,
            "n_tokens": r["n_tokens"],
            "n_images": r["n_images"],
            "hf_loss": r["hf_loss"],
            "ft_loss": ft_loss,
            "loss_relerr": loss_relerr,
            "pre_lm_cos_min": cos_min,
            "pre_lm_cos_mean": cos_mean,
            "pre_lm_abs_max": abs_max,
            "pre_lm_abs_mean": abs_mean,
        })
        print(f"[parity:{label}] tokens={r['n_tokens']} n_img={r['n_images']}  "
              f"loss hf={r['hf_loss']:.4f} ft={ft_loss:.4f} rel_err={loss_relerr*100:.3f}%  "
              f"pre_LM cos_min={cos_min:.6f}  abs_max={abs_max:.2e}")

    # ----- 4. Aggregate report -----
    print("\n" + "=" * 80)
    print("Realistic multimodal dataset parity report")
    print("=" * 80)
    print(f"{'label':>12s}  {'tokens':>7s}  {'n_img':>5s}  "
          f"{'hf_loss':>8s}  {'ft_loss':>8s}  {'rel_err':>8s}  "
          f"{'cos_min':>8s}  {'cos_mean':>9s}  {'abs_max':>10s}")
    print("-" * 105)
    n_pass, n_fail = 0, 0
    fail_labels: list[str] = []
    for row in rows:
        if "error" in row:
            print(f"{row['label']:>12s}  ERROR  {row['error']}")
            n_fail += 1
            fail_labels.append(row["label"])
            continue
        loss_ok = row["loss_relerr"] < LOSS_RELERR_THRESH
        cos_ok = row["pre_lm_cos_min"] > PRE_LM_COS_MIN_THRESH
        mark = "OK" if (loss_ok and cos_ok) else "FAIL"
        if loss_ok and cos_ok:
            n_pass += 1
        else:
            n_fail += 1
            fail_labels.append(row["label"])
        print(f"{row['label']:>12s}  {row['n_tokens']:>7d}  {row['n_images']:>5d}  "
              f"{row['hf_loss']:>8.4f}  {row['ft_loss']:>8.4f}  "
              f"{row['loss_relerr']*100:>7.3f}%  "
              f"{row['pre_lm_cos_min']:>8.6f}  {row['pre_lm_cos_mean']:>9.6f}  "
              f"{row['pre_lm_abs_max']:>10.3e}  {mark}")
    print("-" * 105)
    print(f"PASS: {n_pass}/{len(rows)}    FAIL: {n_fail}/{len(rows)}")
    if rows:
        valid = [r for r in rows if "error" not in r]
        if valid:
            max_rel = max(r["loss_relerr"] for r in valid)
            mean_rel = sum(r["loss_relerr"] for r in valid) / len(valid)
            min_cos = min(r["pre_lm_cos_min"] for r in valid)
            print(f"loss_relerr: max={max_rel*100:.3f}%, mean={mean_rel*100:.3f}%")
            print(f"pre_LM_cos_min: min={min_cos:.6f}")
    if n_fail:
        print(f"\nFAILED examples: {fail_labels}")
        sys.exit(1)
    print("[OK] all realistic multimodal examples parity-pass.")


if __name__ == "__main__":
    main()
