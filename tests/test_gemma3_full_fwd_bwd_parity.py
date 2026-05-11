"""End-to-end forward+backward parity for Gemma 3 dense vs HF transformers.

Runs a single SFT-style microbatch (next-token prediction loss on a
real text prompt) through both stacks. Captures per-layer forward
hidden states, per-layer backward upstream-grad transitions, all
weight gradients, and the loss value. Compares everything with cosine
similarity / sign match / L2 relative error.

This is the parity gate that closes the loop:
  * forward already validated → ``test_gemma3_full_forward_parity.py``
  * block-level backward already validated → ``test_gemma3_block_parity.py``
  * here we glue them: real loss → real backward → real grads.

Memory: HF model is loaded, used, and freed BEFORE flextrain runs, so
peak GPU memory is roughly ``max(hf_model, ft_weights+grads)``. 1B and
4B comfortably fit on a 24 GB+ card.

The flextrain side bypasses the engine and runs blocks manually via the
same pattern as ``test_gemma3_full_forward_parity.py``: build the
``Gemma3Block`` stack, load HF weights with +1 shift and halved→pair
permutes (Q, K, w_q_norm, w_k_norm), drive forward and backward
layer-by-layer. Engine wiring (block builder, ``post_load_permute``,
``from_pretrained``) is deliberately out of scope for this test — it's
plumbing for the loader, not for the math.
"""
from __future__ import annotations

import gc
import os
import sys
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.test_gemma3_block_parity import (
    _compare, _diffstats, _MiniKV, _allocate_slot, _make_chunk,
)
from tests.test_gemma3_full_forward_parity import (
    _MODEL_SPECS, _build_flextrain_blocks, _halved_to_pair_perm,
    _load_global_weights, _load_layer_weights, _open_safetensors,
    _model_available, _rmsnorm_fp32,
)
from flextrain.core.layer import LayerContext
from flextrain.engine.buffers import ScratchPool
from flextrain.nn.blocks.norm import RMSNormBlock


DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MODELS_ROOT = os.path.join(ROOT, "models")

# bf16 + 26-to-48-layer chain accumulates more drift than block-level
# parity. Empirically observed for 1B at T=48: cos > 0.999 on weight
# grads, rel_l2 ~5–10%. The 0.20 rel_l2 ceiling accommodates the
# noise floor on tiny γ vectors (256-d w_q_norm with mean abs ~1e-3
# where one bf16 quantum per element compounds to ~15% relative error).
# The 0.98 cosine threshold is the actual correctness gate: directions
# stay aligned across bf16 noise; sign + L2 are loose corroborators.
PARITY_COS_TOL = 0.98
PARITY_SIGN_TOL = 0.92
PARITY_REL_L2_TOL = 2e-1
LOSS_REL_TOL = 5e-3

# When the reference gradient is below this absolute magnitude, the
# signal is buried in bf16 quantization noise. We still check that the
# direction isn't catastrophically wrong (cos > 0.5 = <60° misalign),
# but the standard rel_l2 / cos thresholds don't apply. Without this
# escape, well-trained γ vectors with near-zero gradients (γ ≈ 1
# already, so the loss-pressure on γ is small) would trip a false
# positive on a handful of layers.
TINY_GRAD_REF_SCALE = 1e-4
TINY_GRAD_COS_TOL = 0.5


# ---------------------------------------------------------------------------
# HF capture
# ---------------------------------------------------------------------------


def _get_hf_decoder_layers(hf_model, arch_id: str):
    """Return the list of ``Gemma3DecoderLayer`` modules and the parent
    text model (for embeddings / final norm access)."""
    if arch_id == "Gemma3ForConditionalGeneration":
        text_model = hf_model.model.language_model
    else:
        text_model = hf_model.model
    return text_model, text_model.layers


def _hf_fwd_bwd_capture(
    model_dir: str, arch_id: str, input_ids: torch.Tensor,
) -> Dict:
    """Run HF forward + backward and capture everything we'll compare.

    Returns a dict on CPU with keys:
      * ``loss``         : scalar Python float
      * ``fwd_states``   : list[n_layers] of bf16 tensors (output of
                           each decoder layer, post-residual)
      * ``bwd_dx_in``    : list[n_layers] of bf16 tensors (grad coming
                           INTO each layer, i.e. ``grad_output[0]``)
      * ``grads``        : dict[hf_param_name -> Tensor]
      * ``embed_grad``   : tensor (vocab, d_model)
    """
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    Loader = (
        AutoModelForImageTextToText
        if arch_id == "Gemma3ForConditionalGeneration"
        else AutoModelForCausalLM
    )
    model = Loader.from_pretrained(
        model_dir, torch_dtype=DTYPE, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    text_model, layers = _get_hf_decoder_layers(model, arch_id)

    # Hooks: capture per-layer fwd output and per-layer backward
    # grad_output[0] (the upstream grad arriving at the layer).
    fwd_outputs: List[torch.Tensor] = [None] * len(layers)
    bwd_dx_in: List[torch.Tensor] = [None] * len(layers)

    def make_fwd_hook(idx: int):
        def hook(_module, _args, output):
            # Gemma3DecoderLayer.forward returns either a Tensor or a
            # tuple depending on transformers version. We want the
            # hidden state, which is element 0 of the tuple if present.
            t = output[0] if isinstance(output, tuple) else output
            fwd_outputs[idx] = t.detach()
        return hook

    def make_bwd_hook(idx: int):
        def hook(_module, _grad_input, grad_output):
            # grad_output[0] = dloss/d(layer_output) — the grad coming
            # INTO this layer from the layer above. This is exactly
            # what flextrain's block.backward(dx, ...) receives as dx.
            if grad_output[0] is not None:
                bwd_dx_in[idx] = grad_output[0].detach()
        return hook

    fwd_handles = [
        layers[i].register_forward_hook(make_fwd_hook(i))
        for i in range(len(layers))
    ]
    bwd_handles = [
        layers[i].register_full_backward_hook(make_bwd_hook(i))
        for i in range(len(layers))
    ]

    # Forward + loss + backward. labels=input_ids → HF shifts internally
    # so position i predicts token at i+1.
    labels = input_ids.clone().to(DEVICE).unsqueeze(0)
    ids = input_ids.to(DEVICE).unsqueeze(0)
    out = model(input_ids=ids, labels=labels, use_cache=False)
    loss = out.loss
    loss.backward()

    loss_value = float(loss.detach().item())

    # Snapshot grads (some param names live under ``model.``,
    # ``language_model.model.``, etc.; keep full HF names verbatim).
    grads: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().to("cpu", copy=True)

    fwd_states_cpu = [t.to("cpu", copy=True) for t in fwd_outputs]
    bwd_dx_in_cpu = [
        (t.to("cpu", copy=True) if t is not None else None)
        for t in bwd_dx_in
    ]

    for h in fwd_handles + bwd_handles:
        h.remove()
    del model, text_model, layers, out, loss
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "loss": loss_value,
        "fwd_states": fwd_states_cpu,
        "bwd_dx_in": bwd_dx_in_cpu,
        "grads": grads,
    }


# ---------------------------------------------------------------------------
# Flextrain end-to-end driver
# ---------------------------------------------------------------------------


def _ft_fwd_bwd(
    *,
    model_dir: str,
    spec: Dict,
    input_ids: torch.Tensor,
    text_cfg,
) -> Dict:
    """Run a full forward + backward through the manually-assembled
    flextrain Gemma3Block stack. Returns per-layer fwd outputs, per-
    layer dx (incoming to each block during bwd), per-block weight
    grads, and the loss value."""
    n_heads = int(text_cfg.num_attention_heads)
    n_kv_heads = int(text_cfg.num_key_value_heads)
    head_dim = int(text_cfg.head_dim)
    d_model = int(text_cfg.hidden_size)
    expert_dim = int(text_cfg.intermediate_size)
    rms_eps = float(text_cfg.rms_norm_eps)
    sliding_window = int(text_cfg.sliding_window)
    layer_types = list(text_cfg.layer_types)
    rope_parameters = getattr(text_cfg, "rope_parameters", None) or {}
    n_layers = int(text_cfg.num_hidden_layers)

    dims = {
        "d_model": d_model, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
        "head_dim": head_dim, "expert_dim": expert_dim,
        "attn_dim": n_heads * head_dim, "kv_dim": n_kv_heads * head_dim,
    }
    hp = {
        "rms_norm_eps": rms_eps,
        "sliding_window": sliding_window,
        "attn_logit_softcap": None,
        "final_logit_softcap": None,
    }

    blocks = _build_flextrain_blocks(
        dims=dims, hp=hp, layer_types=layer_types,
        rope_parameters=rope_parameters,
    )

    get, _ = _open_safetensors(model_dir)
    weights_per_layer = [
        _load_layer_weights(
            get, spec["hf_prefix"], i, head_dim, n_heads, n_kv_heads,
        )
        for i in range(n_layers)
    ]
    embed_w, final_norm_w = _load_global_weights(get, spec["hf_prefix"])

    input_ids_dev = input_ids.to(DEVICE)
    t = int(input_ids_dev.shape[0])

    # ============ Forward ============
    # Embedding + Gemma scale; capture per-layer outputs.
    x = embed_w[input_ids_dev].clone().float().mul_(d_model ** 0.5).to(DTYPE)
    chunk = _make_chunk(t)

    fwd_outputs: List[torch.Tensor] = []
    slots = []
    ctxs = []
    for block, w in zip(blocks, weights_per_layer):
        ctx = LayerContext(
            scratch=ScratchPool(device=DEVICE),
            kv_cache=_MiniKV(
                max_t=t, n_kv_heads=n_kv_heads, head_dim=head_dim,
            ),
            stream=torch.cuda.current_stream(),
            secondary_stream=None,
            total_tokens_per_step=t,
        )
        slot = _allocate_slot(block, t, dims, level=block.schema.max_tier)
        x = block.forward(x, chunk, w, slot, ctx)
        fwd_outputs.append(x.detach().clone())
        slots.append(slot)
        ctxs.append(ctx)

    # Final norm in plain torch fp32 (matches HF's GemmaRMSNorm).
    h_pre_norm = x.detach()
    final_norm_rstd = torch.rsqrt(
        h_pre_norm.float().pow(2).mean(-1, keepdim=True) + rms_eps
    )
    h_normed = (h_pre_norm.float() * final_norm_rstd).to(DTYPE) * final_norm_w

    # Tied LM head + cross-entropy loss (HF shifts internally;
    # mirror that here so the loss values line up).
    logits = h_normed @ embed_w.t()
    shift_logits = logits[:-1].contiguous()
    shift_labels = input_ids_dev[1:].contiguous()
    loss_ft = F.cross_entropy(
        shift_logits.float().view(-1, embed_w.shape[0]),
        shift_labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    )
    loss_value = float(loss_ft.detach().item())

    # ============ Backward ============
    # Use torch autograd for the loss → head → final_norm chain
    # (these aren't yet wrapped as flextrain layers in this test). The
    # block backward is what we're actually testing.
    embed_g = embed_w.detach().clone().requires_grad_(True)
    final_norm_g = final_norm_w.detach().clone().requires_grad_(True)
    h_g = h_pre_norm.detach().clone().requires_grad_(True)

    # Recompute final_norm + head + loss with grad tracking, in matching
    # cast convention.
    rstd = torch.rsqrt(h_g.float().pow(2).mean(-1, keepdim=True) + rms_eps)
    h_normed_g = (h_g.float() * rstd).to(DTYPE) * final_norm_g
    logits_g = h_normed_g @ embed_g.t()
    shift_logits_g = logits_g[:-1].contiguous()
    loss_g = F.cross_entropy(
        shift_logits_g.float().view(-1, embed_w.shape[0]),
        shift_labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    )
    # Grads we capture from autograd:
    #   dh_g  : grad w.r.t. last block's output (feeds into block bwd)
    #   d_final_norm_g : grad for the final RMSNorm γ
    #   d_embed_from_head : the embed grad contribution from the LM head
    dh_g, d_final_norm_g, d_embed_from_head = torch.autograd.grad(
        loss_g, [h_g, final_norm_g, embed_g],
        retain_graph=False,
    )

    # Now drive flextrain block backward in reverse.
    dx = dh_g.to(DTYPE)
    bwd_dx_in_ft: List[torch.Tensor] = [None] * len(blocks)
    grads_per_layer: List[Dict[str, torch.Tensor]] = []
    for i in reversed(range(len(blocks))):
        bwd_dx_in_ft[i] = dx.detach().clone()
        block = blocks[i]
        w = weights_per_layer[i]
        slot = slots[i]
        ctx = ctxs[i]
        grads: Dict[str, torch.Tensor] = {}
        for ts in block.param_spec.tensors:
            grads["g_" + ts.name[2:]] = torch.zeros(
                ts.shape(dims), dtype=ts.grad_dtype, device=DEVICE,
            )
        dx = block.backward(dx, chunk, w, grads, slot, ctx)
        grads_per_layer.append(grads)
    grads_per_layer.reverse()

    # dembed = scatter_add(dx * scale, input_ids) + d_embed_from_head
    # where scale = sqrt(d_model). The grad coming OUT of block 0 IS
    # the grad w.r.t. (embed[ids] * sqrt(d_model)); chain rule pulls
    # the sqrt(d_model) factor onto the embed grad.
    d_embed_from_input = torch.zeros_like(embed_w, dtype=torch.float32)
    d_embed_from_input.index_add_(
        0, input_ids_dev, dx.float() * (d_model ** 0.5),
    )
    d_embed_total = d_embed_from_input.to(DTYPE) + d_embed_from_head.to(DTYPE)

    return {
        "loss": loss_value,
        "fwd_states": [t.to("cpu", copy=True) for t in fwd_outputs],
        "bwd_dx_in": [t.to("cpu", copy=True) for t in bwd_dx_in_ft],
        "grads_per_layer": [
            {k: v.to("cpu", copy=True) for k, v in g.items()}
            for g in grads_per_layer
        ],
        "embed_grad": d_embed_total.to("cpu", copy=True),
        "final_norm_grad": d_final_norm_g.detach().to("cpu", copy=True),
    }


# ---------------------------------------------------------------------------
# HF → flextrain weight-grad layout adapter
# ---------------------------------------------------------------------------


def _map_hf_grad_to_ft_layout(
    ft_name: str, hf_grad: torch.Tensor,
    head_dim: int, n_heads: int, n_kv_heads: int,
) -> torch.Tensor:
    """Convert an HF parameter gradient into flextrain layout so we
    can compare against ``grads[g_*]`` directly.

    The mappings mirror the load path exactly, just inverted for the
    grad (which moves contravariantly): transpose for linear weights,
    halved→pair permute on the post-transpose axis for Q/K, head-axis
    permute for the QK-norm γ vectors.
    """
    if ft_name in (
        "w_pre_attn_norm", "w_post_attn_norm",
        "w_pre_ffn_norm", "w_post_ffn_norm",
    ):
        return hf_grad

    if ft_name in ("w_q",):
        attn_dim = n_heads * head_dim
        perm = _halved_to_pair_perm(attn_dim, head_dim)
        return hf_grad.t().contiguous()[:, perm].contiguous()

    if ft_name in ("w_k",):
        kv_dim = n_kv_heads * head_dim
        perm = _halved_to_pair_perm(kv_dim, head_dim)
        return hf_grad.t().contiguous()[:, perm].contiguous()

    if ft_name in ("w_v", "w_o", "w_1", "w_2", "w_3"):
        return hf_grad.t().contiguous()

    if ft_name in ("w_q_norm", "w_k_norm"):
        head_perm = _halved_to_pair_perm(head_dim, head_dim)
        return hf_grad[head_perm].contiguous()

    raise KeyError(f"unknown flextrain param name {ft_name!r}")


_HF_PARAM_SUFFIX = {
    "w_pre_attn_norm":  "input_layernorm.weight",
    "w_post_attn_norm": "post_attention_layernorm.weight",
    "w_pre_ffn_norm":   "pre_feedforward_layernorm.weight",
    "w_post_ffn_norm":  "post_feedforward_layernorm.weight",
    "w_q":              "self_attn.q_proj.weight",
    "w_k":              "self_attn.k_proj.weight",
    "w_v":              "self_attn.v_proj.weight",
    "w_o":              "self_attn.o_proj.weight",
    "w_q_norm":         "self_attn.q_norm.weight",
    "w_k_norm":         "self_attn.k_norm.weight",
    "w_1":              "mlp.gate_proj.weight",
    "w_2":              "mlp.down_proj.weight",
    "w_3":              "mlp.up_proj.weight",
}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fwd+bwd parity requires CUDA",
)


def _maybe_skip_for_memory(size_name: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    free_b, total_b = torch.cuda.mem_get_info()
    # Rough heuristic. HF + grads is ~2× param bytes (params+grads in bf16).
    needs_gb = {"1B": 8, "4B": 18, "12B": 36}[size_name]
    if free_b < needs_gb * 2**30:
        pytest.skip(
            f"{size_name} parity needs ~{needs_gb} GB free; "
            f"have {free_b / 2**30:.1f} of {total_b / 2**30:.1f} GB"
        )


@pytest.mark.parametrize("size_name", ["1B", "4B"])
def test_gemma3_full_fwd_bwd_parity(size_name: str) -> None:
    spec = _MODEL_SPECS[size_name]
    model_dir = os.path.join(MODELS_ROOT, spec["dir"])
    if not _model_available(model_dir):
        pytest.skip(f"{spec['dir']} not present under models/")
    _maybe_skip_for_memory(size_name)

    from transformers import AutoConfig, AutoTokenizer

    hf_cfg = AutoConfig.from_pretrained(model_dir)
    text_cfg = hf_cfg.get_text_config()

    # One real SFT-style microbatch (next-token CE on a real prompt).
    tok = AutoTokenizer.from_pretrained(model_dir)
    prompt = (
        "Programming is the art of constructing precise instructions for a "
        "computer. The smallest detail in a program can completely change "
        "the result. Therefore, programmers must be careful and rigorous "
        "thinkers."
    )
    input_ids = tok(prompt, return_tensors="pt").input_ids.squeeze(0)
    # Cap so 4B fits comfortably; pre-shifting handled by HF/flextrain.
    input_ids = input_ids[:48]

    head_dim = int(text_cfg.head_dim)
    n_heads = int(text_cfg.num_attention_heads)
    n_kv_heads = int(text_cfg.num_key_value_heads)
    n_layers = int(text_cfg.num_hidden_layers)

    # === HF capture ===
    print(f"\n[{size_name}] running HF fwd+bwd...", flush=True)
    hf = _hf_fwd_bwd_capture(model_dir, spec["arch_id"], input_ids)
    print(f"[{size_name}] HF loss={hf['loss']:.6f}", flush=True)

    # === Flextrain fwd+bwd ===
    print(f"[{size_name}] running flextrain fwd+bwd...", flush=True)
    ft = _ft_fwd_bwd(
        model_dir=model_dir, spec=spec,
        input_ids=input_ids, text_cfg=text_cfg,
    )
    print(f"[{size_name}] flextrain loss={ft['loss']:.6f}", flush=True)

    # === Loss parity ===
    rel_loss_err = abs(ft["loss"] - hf["loss"]) / max(abs(hf["loss"]), 1e-9)
    print(
        f"[{size_name}] loss: hf={hf['loss']:.6f} ft={ft['loss']:.6f} "
        f"rel_err={rel_loss_err:.3e}",
        flush=True,
    )
    assert rel_loss_err < LOSS_REL_TOL, (
        f"loss parity broken: hf={hf['loss']} ft={ft['loss']} "
        f"rel_err={rel_loss_err}"
    )

    # === Per-layer forward transition parity ===
    print(f"[{size_name}] per-layer fwd transitions:", flush=True)
    fwd_fail: List[str] = []
    for i in range(n_layers):
        hf_t = hf["fwd_states"][i].squeeze(0).to(DEVICE)
        ft_t = ft["fwd_states"][i].to(DEVICE)
        s = _diffstats(ft_t, hf_t)
        print(
            f"  L{i:02d} cos={s['cos']:.6f} sign={s['sign_match']:.4f} "
            f"rel_l2={s['rel_l2']:.3e}",
            flush=True,
        )
        try:
            _compare(
                f"fwd[L{i:02d}]", ft_t, hf_t,
                cos_tol=PARITY_COS_TOL,
                sign_tol=PARITY_SIGN_TOL,
                rel_l2_tol=PARITY_REL_L2_TOL,
            )
        except AssertionError as e:
            fwd_fail.append(str(e))

    # === Per-layer backward dx-in transition parity ===
    print(f"[{size_name}] per-layer bwd dx-in transitions:", flush=True)
    bwd_fail: List[str] = []
    for i in range(n_layers):
        if hf["bwd_dx_in"][i] is None:
            continue
        hf_t = hf["bwd_dx_in"][i].squeeze(0).to(DEVICE)
        ft_t = ft["bwd_dx_in"][i].to(DEVICE)
        s = _diffstats(ft_t, hf_t)
        print(
            f"  L{i:02d} cos={s['cos']:.6f} sign={s['sign_match']:.4f} "
            f"rel_l2={s['rel_l2']:.3e}",
            flush=True,
        )
        try:
            _compare(
                f"bwd_dx_in[L{i:02d}]", ft_t, hf_t,
                cos_tol=PARITY_COS_TOL,
                sign_tol=PARITY_SIGN_TOL,
                rel_l2_tol=PARITY_REL_L2_TOL,
            )
        except AssertionError as e:
            bwd_fail.append(str(e))

    # === Per-layer weight-grad parity ===
    print(f"[{size_name}] per-layer weight grads:", flush=True)
    grad_fail: List[str] = []
    # HF safetensor prefix differs from HF in-memory ``named_parameters``
    # prefix for the multimodal wrapper:
    #   * 1B  (Gemma3ForCausalLM):              both = "model.layers"
    #   * 4B/12B (Gemma3ForConditionalGeneration):
    #         safetensor: "language_model.model.layers"
    #         in-memory:  "model.language_model.layers"
    # The grads dict is keyed by in-memory names from ``named_parameters``.
    if spec["arch_id"] == "Gemma3ForConditionalGeneration":
        hf_grad_prefix = "model.language_model.layers"
    else:
        hf_grad_prefix = "model.layers"
    for i in range(n_layers):
        ft_grads = ft["grads_per_layer"][i]
        for ft_name, hf_suffix in _HF_PARAM_SUFFIX.items():
            grad_key_ft = "g_" + ft_name[2:]
            if grad_key_ft not in ft_grads:
                continue
            hf_key = f"{hf_grad_prefix}.{i}.{hf_suffix}"
            hf_grad = hf["grads"].get(hf_key)
            if hf_grad is None:
                grad_fail.append(
                    f"L{i}.{ft_name}: no HF grad at {hf_key!r}"
                )
                continue
            hf_grad_in_ft = _map_hf_grad_to_ft_layout(
                ft_name, hf_grad.to(DEVICE),
                head_dim, n_heads, n_kv_heads,
            )
            ft_grad = ft_grads[grad_key_ft].to(DEVICE)
            s = _diffstats(ft_grad, hf_grad_in_ft)
            # Tiny-grad noise-floor escape: when the reference signal is
            # below the bf16 noise floor, only require that the
            # direction isn't catastrophically wrong (cos > 0.5).
            if s["ref_scale"] < TINY_GRAD_REF_SCALE:
                if s["cos"] < TINY_GRAD_COS_TOL:
                    grad_fail.append(
                        f"g[L{i:02d}].{ft_name}: tiny-grad cos="
                        f"{s['cos']:.4f} < {TINY_GRAD_COS_TOL} "
                        f"(ref_scale={s['ref_scale']:.2e})"
                    )
            else:
                try:
                    _compare(
                        f"g[L{i:02d}].{ft_name}", ft_grad, hf_grad_in_ft,
                        cos_tol=PARITY_COS_TOL,
                        sign_tol=PARITY_SIGN_TOL,
                        rel_l2_tol=PARITY_REL_L2_TOL,
                    )
                except AssertionError as e:
                    grad_fail.append(str(e))
            if i in (0, n_layers // 2, n_layers - 1):
                print(
                    f"  L{i:02d}.{ft_name:18s} "
                    f"cos={s['cos']:.6f} sign={s['sign_match']:.4f} "
                    f"rel_l2={s['rel_l2']:.3e}",
                    flush=True,
                )

    failures = fwd_fail + bwd_fail + grad_fail
    if failures:
        msg = "\n".join(failures[:30])  # cap to first 30 to keep output sane
        more = (
            f"\n... and {len(failures) - 30} more" if len(failures) > 30 else ""
        )
        raise AssertionError(
            f"{size_name} parity failures ({len(failures)} total):\n"
            f"{msg}{more}"
        )
