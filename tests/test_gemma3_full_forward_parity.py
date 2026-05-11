"""Full-model forward parity for Gemma 3 dense vs HF transformers.

Loads each local Gemma 3 checkpoint (1B / 4B / 12B), runs one forward
through both the HF reference and a hand-assembled flextrain
``Gemma3Block`` stack with HF weights remapped, then compares the
final-layer logits via cosine similarity / sign-match / L2-relative
error.

This is the gate the user requires before any backward work lands: if
the layer math is wrong end-to-end, no amount of grad-routing will save
us. Backward implementation (Phase B from
``docs/internal/gemma3_status.md``) is blocked behind this passing for
every model scale.

The test deliberately bypasses ``from_pretrained`` / ``ARCH_MODULES``
to keep the verification narrow: it only validates the per-layer math
+ HF-weight mapping (transpose, +1 norm shift, halved→pair Q/K perm,
per-layer rope-base alternation, optional linear rope_scaling on full
layers). Block-builder/registry/engine wiring is the responsibility of
a later step (4d–4f in the status doc) and gets its own test.

Memory note for 12B: ~24 GB just for params at bf16. We load HF,
forward, copy logits to CPU, free, then load flextrain. The 3090 fits
all three model sizes this way; if a tighter GPU is in play, the test
falls back gracefully via ``pytest.skip``.
"""
from __future__ import annotations

import dataclasses
import gc
import json
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
from flextrain.core.layer import LayerContext
from flextrain.engine.buffers import ScratchPool
from flextrain.nn.layers.gemma3 import Gemma3Block, Gemma3BlockConfig


DEVICE = "cuda:0"
DTYPE = torch.bfloat16
MODELS_ROOT = os.path.join(ROOT, "models")

# Empirical thresholds for full-model logits parity (bf16 noise over
# the entire stack — 26 / 34 / 48 layers of accumulated quantization).
# Reported below the test; tighten if a real bug shows up.
LOGITS_COS_TOL = 0.995
LOGITS_SIGN_TOL = 0.95
LOGITS_REL_L2_TOL = 1e-1


_MODEL_SPECS = {
    "1B": {
        "dir": "Gemma-3-1B-Instruct",
        "arch_id": "Gemma3ForCausalLM",
        "hf_prefix": "model",            # safetensor: ``model.layers.{i}.*``
    },
    "4B": {
        "dir": "Gemma-3-4B-Instruct",
        "arch_id": "Gemma3ForConditionalGeneration",
        "hf_prefix": "language_model.model",
    },
    "12B": {
        "dir": "Gemma-3-12B-Instruct",
        "arch_id": "Gemma3ForConditionalGeneration",
        "hf_prefix": "language_model.model",
    },
}


# ---------------------------------------------------------------------------
# HF weight loading + remapping
# ---------------------------------------------------------------------------


def _open_safetensors(model_dir: str):
    """Open a single-file or multi-shard safetensors checkpoint and
    return a callable ``get(name) -> Tensor`` plus the set of available
    keys."""
    from safetensors.torch import safe_open

    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        idx = json.load(open(idx_path))
        wm = idx["weight_map"]
        shard_files = sorted(set(wm.values()))
        readers = {
            s: safe_open(os.path.join(model_dir, s), framework="pt")
            for s in shard_files
        }
        keys = set(wm.keys())

        def get(name: str) -> torch.Tensor:
            return readers[wm[name]].get_tensor(name)

        return get, keys

    # Single file.
    r = safe_open(
        os.path.join(model_dir, "model.safetensors"), framework="pt"
    )
    keys = set(r.keys())

    def get(name: str) -> torch.Tensor:
        return r.get_tensor(name)

    return get, keys


_HF_TO_FT_LAYER = {
    "input_layernorm.weight":              ("w_pre_attn_norm",  False),
    "post_attention_layernorm.weight":     ("w_post_attn_norm", False),
    "pre_feedforward_layernorm.weight":    ("w_pre_ffn_norm",   False),
    "post_feedforward_layernorm.weight":   ("w_post_ffn_norm",  False),
    "self_attn.q_norm.weight":             ("w_q_norm",         False),
    "self_attn.k_norm.weight":             ("w_k_norm",         False),
    "self_attn.q_proj.weight":             ("w_q",              True),
    "self_attn.k_proj.weight":             ("w_k",              True),
    "self_attn.v_proj.weight":             ("w_v",              True),
    "self_attn.o_proj.weight":             ("w_o",              True),
    "mlp.gate_proj.weight":                ("w_1",              True),
    "mlp.up_proj.weight":                  ("w_3",              True),
    "mlp.down_proj.weight":                ("w_2",              True),
}


def _halved_to_pair_perm(total_dim: int, head_dim: int) -> torch.Tensor:
    """Permute axis-1 of a 2-D Q or K weight so per-head channels go
    from HF's halved-split RoPE layout (pairs ``(d[i], d[i+H/2])``) to
    flextrain's pair-interleave RoPE layout (pairs ``(d[2i], d[2i+1])``).
    Same convention as the Llama parity test.
    """
    n_heads = total_dim // head_dim
    half = head_dim // 2
    perm: List[int] = []
    for h in range(n_heads):
        base = h * head_dim
        for i in range(half):
            perm.append(base + i)
            perm.append(base + half + i)
    return torch.tensor(perm, dtype=torch.int64)


def _load_layer_weights(
    get, prefix: str, layer_idx: int, head_dim: int,
    n_heads: int, n_kv_heads: int,
) -> Dict[str, torch.Tensor]:
    """Load all HF weights for one layer, apply transpose / +1 / Q,K perm.
    Returns a dict keyed by flextrain weight names (``w_q``, ``w_pre_attn_norm``,
    ``w_q_norm``, …) on CUDA in bf16.
    """
    out: Dict[str, torch.Tensor] = {}
    layer_pfx = f"{prefix}.layers.{layer_idx}."
    for hf_suffix, (ft_name, do_transpose) in _HF_TO_FT_LAYER.items():
        t = get(layer_pfx + hf_suffix).to(DEVICE).to(DTYPE)
        if do_transpose:
            t = t.t().contiguous()
        out[ft_name] = t

    # +1 shift on all RMSNorm γ (residual-stream and QK-norm — same
    # convention; see _gemma3_post_load_hook).
    for n in (
        "w_pre_attn_norm", "w_post_attn_norm",
        "w_pre_ffn_norm", "w_post_ffn_norm",
        "w_q_norm", "w_k_norm",
    ):
        out[n] = out[n] + 1.0

    # Halved → pair-interleave permute on Q and K (axis-1 of the
    # transposed (d_model, attn_dim/kv_dim) weights). Plus the SAME
    # per-head channel permute on the QK-norm γ vectors (length
    # head_dim) — they index the post-permute channel axis.
    attn_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q_perm = _halved_to_pair_perm(attn_dim, head_dim).to(DEVICE)
    k_perm = _halved_to_pair_perm(kv_dim, head_dim).to(DEVICE)
    head_perm = _halved_to_pair_perm(head_dim, head_dim).to(DEVICE)
    out["w_q"] = out["w_q"][:, q_perm].contiguous()
    out["w_k"] = out["w_k"][:, k_perm].contiguous()
    out["w_q_norm"] = out["w_q_norm"][head_perm].contiguous()
    out["w_k_norm"] = out["w_k_norm"][head_perm].contiguous()
    return out


def _load_global_weights(
    get, prefix: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (embed_weight, final_norm_weight) on CUDA bf16. Final norm
    gets the +1 shift; embed does not."""
    embed = get(f"{prefix}.embed_tokens.weight").to(DEVICE).to(DTYPE)
    final_norm = get(f"{prefix}.norm.weight").to(DEVICE).to(DTYPE) + 1.0
    return embed, final_norm


# ---------------------------------------------------------------------------
# Flextrain block construction
# ---------------------------------------------------------------------------


def _build_flextrain_blocks(
    *, dims: Dict[str, int], hp: Dict, layer_types: List[str],
    rope_parameters: Dict,
) -> List[Gemma3Block]:
    """Build one Gemma3Block per layer with per-layer rope-base and
    rope-scaling (full-attention layers get the linear scaling on 4B/12B;
    sliding layers stay vanilla).
    """
    base = Gemma3BlockConfig(
        d_model=int(dims["d_model"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=int(dims["n_kv_heads"]),
        head_dim=int(dims["head_dim"]),
        expert_dim=int(dims["expert_dim"]),
        rms_norm_eps=float(hp["rms_norm_eps"]),
        is_causal=True,
        attn_logit_softcap=hp.get("attn_logit_softcap"),
        final_logit_softcap=hp.get("final_logit_softcap"),
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    )

    sliding_params = rope_parameters.get("sliding_attention", {})
    full_params = rope_parameters.get("full_attention", {})

    blocks: List[Gemma3Block] = []
    for i, lt in enumerate(layer_types):
        is_sliding = lt == "sliding_attention"
        rp = sliding_params if is_sliding else full_params
        rope_base = float(rp.get("rope_theta", 10_000.0))
        rope_scaling: Optional[dict] = None
        rtype = rp.get("rope_type", "default")
        if rtype == "linear":
            rope_scaling = {
                "rope_type": "linear",
                "factor": float(rp.get("factor", 1.0)),
            }
        elif rtype not in ("default", None):
            raise NotImplementedError(
                f"rope_type {rtype!r} not implemented (layer {i})"
            )
        cfg = dataclasses.replace(
            base,
            rope_base=rope_base,
            rope_scaling=rope_scaling,
            window_size_left=int(hp["sliding_window"]) if is_sliding else -1,
        )
        blocks.append(Gemma3Block(layer_id=i, cfg=cfg))
    return blocks


# ---------------------------------------------------------------------------
# Forward driver
# ---------------------------------------------------------------------------


def _rmsnorm_fp32(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp = x.float()
    rstd = torch.rsqrt(x_fp.pow(2).mean(-1, keepdim=True) + eps)
    return (x_fp * rstd).to(x.dtype) * w


def _drive_flextrain_forward(
    *,
    blocks: List[Gemma3Block],
    weights_per_layer: List[Dict[str, torch.Tensor]],
    embed_weight: torch.Tensor,
    final_norm_weight: torch.Tensor,
    input_ids: torch.Tensor,           # (T,) int
    dims: Dict[str, int],
    rms_norm_eps: float,
) -> torch.Tensor:
    """Drive a full forward through the flextrain block stack. Returns
    bf16 logits of shape ``(T, vocab)``."""
    d_model = int(dims["d_model"])
    head_dim = int(dims["head_dim"])
    n_kv_heads = int(dims["n_kv_heads"])
    t = int(input_ids.shape[0])

    # Embedding + Gemma's sqrt(d_model) scaling (HF applies this between
    # embed_tokens and the first decoder layer).
    x = embed_weight[input_ids].clone()
    x = x.float().mul_(d_model ** 0.5).to(DTYPE)

    chunk = _make_chunk(t)
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
        # Free per-layer scratch + slot tensors immediately so we don't
        # accumulate 26/34/48 copies of every activation tensor.
        del ctx, slot

    # Final norm + tied LM head.
    x = _rmsnorm_fp32(x, final_norm_weight, rms_norm_eps)
    logits = x @ embed_weight.t()
    return logits


# ---------------------------------------------------------------------------
# HF reference forward
# ---------------------------------------------------------------------------


def _hf_reference_logits(
    model_dir: str, arch_id: str, input_ids: torch.Tensor,
) -> torch.Tensor:
    """Load the HF model, run one forward, copy logits to CPU, free the
    model. Returns CPU bf16 logits of shape ``(T, vocab)``."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    if arch_id == "Gemma3ForCausalLM":
        Loader = AutoModelForCausalLM
    else:
        Loader = AutoModelForImageTextToText
    model = Loader.from_pretrained(
        model_dir, torch_dtype=DTYPE, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()

    # Layout differs between the two Gemma 3 wrappers:
    #   - Gemma3ForCausalLM (1B):        text model at ``model.model``;
    #                                     LM head at ``model.lm_head``.
    #   - Gemma3ForConditionalGeneration (4B/12B):
    #                                     ``model.model`` is the multimodal
    #                                     wrapper; text branch at
    #                                     ``model.model.language_model``;
    #                                     LM head at ``model.lm_head``.
    if arch_id == "Gemma3ForConditionalGeneration":
        text_model = model.model.language_model
    else:
        text_model = model.model
    lm_head = getattr(model, "lm_head", None)

    with torch.no_grad():
        outputs = text_model(
            input_ids=input_ids.to(DEVICE).unsqueeze(0),
            use_cache=False,
            output_hidden_states=False,
        )
        last_hidden = outputs.last_hidden_state  # (1, T, d_model)
        if lm_head is None:
            # Tied embeddings — apply embed.T manually.
            embed = text_model.embed_tokens.weight
            logits = last_hidden @ embed.t()
        else:
            logits = lm_head(last_hidden)

    logits_cpu = logits.squeeze(0).detach().to("cpu")
    del model, text_model, outputs, last_hidden, logits, lm_head
    gc.collect()
    torch.cuda.empty_cache()
    return logits_cpu


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _model_available(model_dir: str) -> bool:
    return os.path.isdir(model_dir) and any(
        f.startswith("model") and f.endswith(".safetensors")
        for f in os.listdir(model_dir)
    )


def _maybe_skip_for_memory(size_name: str) -> None:
    """Skip 12B if the GPU clearly can't fit it. Sufficient indicator:
    24 GB total memory and another resident process. We approximate by
    requiring at least 18 GB of free memory."""
    if size_name != "12B":
        return
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    free_b, total_b = torch.cuda.mem_get_info()
    if free_b < 18 * 2**30:
        pytest.skip(
            f"12B parity needs ~24 GB free; "
            f"have {free_b / 2**30:.1f} of {total_b / 2**30:.1f} GB"
        )


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="flextrain forward parity requires CUDA",
)


@pytest.mark.parametrize("size_name", ["1B", "4B", "12B"])
def test_gemma3_full_forward_parity(size_name: str) -> None:
    spec = _MODEL_SPECS[size_name]
    model_dir = os.path.join(MODELS_ROOT, spec["dir"])
    if not _model_available(model_dir):
        pytest.skip(f"{spec['dir']} not present under models/")
    _maybe_skip_for_memory(size_name)

    from transformers import AutoConfig, AutoTokenizer

    hf_cfg = AutoConfig.from_pretrained(model_dir)
    text_cfg = hf_cfg.get_text_config()

    tok = AutoTokenizer.from_pretrained(model_dir)
    # Short fixed prompt — keeps memory bounded and reproducible.
    prompt = (
        "The quick brown fox jumps over the lazy dog. "
        "In a single pass through this sentence we want bf16 forward parity."
    )
    input_ids = tok(prompt, return_tensors="pt").input_ids.squeeze(0)
    # Cap length so we don't blow past a sliding window or eat memory.
    input_ids = input_ids[: 64]
    t = int(input_ids.shape[0])

    # --- HF reference forward (sequential to keep memory bounded) ---
    hf_logits_cpu = _hf_reference_logits(model_dir, spec["arch_id"], input_ids)

    # --- Flextrain forward ---
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

    get, _keys = _open_safetensors(model_dir)
    weights_per_layer = [
        _load_layer_weights(
            get, spec["hf_prefix"], i, head_dim, n_heads, n_kv_heads,
        )
        for i in range(n_layers)
    ]
    embed_w, final_norm_w = _load_global_weights(get, spec["hf_prefix"])

    ft_logits = _drive_flextrain_forward(
        blocks=blocks,
        weights_per_layer=weights_per_layer,
        embed_weight=embed_w,
        final_norm_weight=final_norm_w,
        input_ids=input_ids.to(DEVICE),
        dims=dims, rms_norm_eps=rms_eps,
    )

    stats = _diffstats(ft_logits, hf_logits_cpu.to(DEVICE))
    print(
        f"\n[{size_name}] T={t} n_layers={n_layers} "
        f"cos={stats['cos']:.6f} sign={stats['sign_match']:.4f} "
        f"rel_l2={stats['rel_l2']:.3e} max_abs={stats['max_abs']:.3e} "
        f"ref_scale={stats['ref_scale']:.3e}"
    )
    _compare(
        f"logits[{size_name}]", ft_logits, hf_logits_cpu.to(DEVICE),
        cos_tol=LOGITS_COS_TOL,
        sign_tol=LOGITS_SIGN_TOL,
        rel_l2_tol=LOGITS_REL_L2_TOL,
    )
