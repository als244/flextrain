"""Smoke tests for ``flextrain/io/arch/gemma4.py``.

Covers the pieces that don't need a real 31B checkpoint or GPU:

* Per-layer block-config translation from the HF Gemma-4-31B-Instruct
  config (sliding head shape vs global head shape, k_eq_v on globals,
  proportional partial rope on globals, rope bases per layer-type).
* Math of the partial halved → pair-interleave permute used by
  ``post_load_permute``.

Full-model load (which needs ~62 GB host RAM) and full-model parity
(needs GPU + the real safetensors) are covered separately in the
Stage 2 / Stage 3 tests (not yet written; see ``docs/internal/gemma4_status.md``).
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, "models", "Gemma-4-31B-Instruct")


# ---------------------------------------------------------------------------
# Config translation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(MODEL_DIR, "config.json")),
    reason="Gemma-4-31B-Instruct/config.json not present",
)
def test_gemma4_config_translates_to_dims_and_hyperparams() -> None:
    """Sanity-check the HF config → flextrain dims/hyperparams pipeline."""
    from flextrain.io.arch.gemma4 import (
        hf_config_to_flextrain, hf_config_to_hyperparams,
    )

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        hf_config = json.load(f)

    dims = hf_config_to_flextrain(hf_config)
    hp = hf_config_to_hyperparams(hf_config)

    # Dims: sliding-layer shape (most common) carried at the top level.
    assert dims["n_layers"] == 60
    assert dims["d_model"] == 5376
    assert dims["n_heads"] == 32
    assert dims["n_kv_heads"] == 16
    assert dims["head_dim"] == 256
    assert dims["expert_dim"] == 21504
    assert dims["vocab_size"] == 262144

    # Per-layer-type knobs.
    assert hp["global_head_dim"] == 512
    assert hp["num_global_key_value_heads"] == 4
    assert hp["attention_k_eq_v"] is True
    assert hp["global_partial_rotary_factor"] == 0.25
    assert hp["sliding_window"] == 1024
    assert hp["rope_theta"] == 1_000_000.0
    assert hp["rope_local_base"] == 10_000.0
    assert hp["final_logit_softcap"] == 30.0
    assert hp["tie_word_embeddings"] is True
    assert hp["rms_norm_eps"] == 1e-6

    # layer_types: 5:1 sliding:global pattern, full at index 5, 11, 17 ...
    assert len(hp["layer_types"]) == 60
    full_idx = [i for i, t in enumerate(hp["layer_types"]) if t == "full_attention"]
    assert full_idx == [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]


@pytest.mark.skipif(
    not os.path.isfile(os.path.join(MODEL_DIR, "config.json")),
    reason="Gemma-4-31B-Instruct/config.json not present",
)
def test_gemma4_block_builder_per_layer_config() -> None:
    """The block builder produces sliding configs on sliding indices
    and global+k_eq_v configs on full_attention indices."""
    import types
    from flextrain.io.arch.gemma4 import (
        _gemma4_block_builder, hf_config_to_flextrain, hf_config_to_hyperparams,
    )

    with open(os.path.join(MODEL_DIR, "config.json")) as f:
        hf_config = json.load(f)
    dims = hf_config_to_flextrain(hf_config)
    hp = hf_config_to_hyperparams(hf_config)

    ctx = types.SimpleNamespace(
        dims=dims, hyperparams=hp,
        compute_dtype=torch.bfloat16, master_dtype=None, grad_dtype=None,
        norm_grad_dtype=torch.float32, lora_targets=None,
    )

    # Sliding layer (L=0).
    b0 = _gemma4_block_builder(0, ctx)
    c0 = b0.cfg
    assert c0.head_dim == 256
    assert c0.n_kv_heads == 16
    assert c0.k_eq_v is False
    assert c0.v_norm is True
    assert c0.partial_rotary_factor == 1.0
    assert c0.rope_base == 10_000.0
    assert c0.window_size_left == 1024
    attn0_params = {t.name for t in b0.attn.param_spec().tensors}
    assert "w_v" in attn0_params
    assert "w_q_norm" in attn0_params and "w_k_norm" in attn0_params

    # Global layer (L=5).
    b5 = _gemma4_block_builder(5, ctx)
    c5 = b5.cfg
    assert c5.head_dim == 512
    assert c5.n_kv_heads == 4
    assert c5.k_eq_v is True
    assert c5.v_norm is True
    assert c5.partial_rotary_factor == 0.25
    assert c5.rope_base == 1_000_000.0
    assert c5.window_size_left == -1
    attn5_params = {t.name for t in b5.attn.param_spec().tensors}
    assert "w_v" not in attn5_params, (
        "global layer should not declare w_v (k_eq_v=True)"
    )
    assert "w_q_norm" in attn5_params and "w_k_norm" in attn5_params

    # Per-layer ParamSpec on the block carries through.
    block_specs = {t.name for t in b5.param_spec.tensors}
    assert "w_v" not in block_specs


# ---------------------------------------------------------------------------
# Partial halved→pair permute math
# ---------------------------------------------------------------------------


def test_partial_halved_to_pair_perm_full_rope_matches_gemma3() -> None:
    """When ``rope_angles = head_dim/2`` (full rope), the partial
    permute reduces to the Gemma-3 halved→pair permute exactly."""
    from flextrain.io.arch.gemma4 import _partial_halved_to_pair_perm

    head_dim = 16
    rope_angles = head_dim // 2

    ft_perm = _partial_halved_to_pair_perm(head_dim, rope_angles)

    # Reference: gemma3._halved_to_pair_perm pseudocode.
    half = head_dim // 2
    expected = torch.empty(head_dim, dtype=torch.int64)
    for i in range(half):
        expected[2 * i] = i
        expected[2 * i + 1] = half + i

    assert torch.equal(ft_perm, expected), (
        f"full-rope reduction mismatch\n  ft={ft_perm.tolist()}\n  ex={expected.tolist()}"
    )


def test_partial_halved_to_pair_perm_partial_31b_global() -> None:
    """For Gemma 4's global layers (head_dim=512, prf=0.25): rotated
    prefix has 128 channels, non-rotated suffix has 384 channels.
    """
    from flextrain.io.arch.gemma4 import _partial_halved_to_pair_perm

    head_dim = 512
    rope_angles = int(0.25 * head_dim // 2)  # 64
    rot_dim = 2 * rope_angles                # 128

    ft_perm = _partial_halved_to_pair_perm(head_dim, rope_angles)

    # Rotated prefix: FT[2i]=HF[i], FT[2i+1]=HF[256+i] for i in [0, 64).
    for i in range(rope_angles):
        assert ft_perm[2 * i].item() == i
        assert ft_perm[2 * i + 1].item() == head_dim // 2 + i

    # Non-rotated suffix: HF positions [64, 256) then [256+64, 512) in order.
    suffix = list(range(rope_angles, head_dim // 2)) + list(
        range(head_dim // 2 + rope_angles, head_dim)
    )
    assert ft_perm[rot_dim:].tolist() == suffix

    # Permutation must be a valid bijection.
    assert sorted(ft_perm.tolist()) == list(range(head_dim))


def test_partial_permute_preserves_un_rotated_channels() -> None:
    """For Gemma 4 globals, the non-rotated tail channels in HF layout
    pass through unchanged in numeric value (just reordered). Verify
    by checking the permuted tensor at the suffix positions matches
    HF positions [rope_angles, half) ∪ [half + rope_angles, head_dim)."""
    from flextrain.io.arch.gemma4 import _partial_halved_to_pair_perm

    head_dim = 64
    rope_angles = 8
    rot_dim = 16
    ft_perm = _partial_halved_to_pair_perm(head_dim, rope_angles)

    # Synthetic HF tensor: hf[i] = i + 100 for visibility.
    hf = torch.arange(head_dim, dtype=torch.float32) + 100.0
    ft = hf[ft_perm]

    # The suffix [rot_dim, head_dim) of ft must equal HF's non-rotated
    # positions in natural order: [rope_angles, half) ∪ [half+rope_angles, head_dim).
    half = head_dim // 2
    hf_suffix_positions = list(range(rope_angles, half)) + list(
        range(half + rope_angles, head_dim)
    )
    expected_suffix = hf[hf_suffix_positions]
    assert torch.equal(ft[rot_dim:], expected_suffix)


# ---------------------------------------------------------------------------
# ArchSpec registration
# ---------------------------------------------------------------------------


def test_gemma4_archspec_registered() -> None:
    """``Gemma4ForCausalLM`` and ``Gemma4ForConditionalGeneration`` both
    resolve to the gemma4 ArchSpec via ``select_arch``."""
    from flextrain.io.hf_weights import select_arch

    for arch_id in ("Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"):
        spec = select_arch({"architectures": [arch_id]})
        assert arch_id in spec.hf_arch_ids
        # Layer weight map: optional w_v exists (k_eq_v on globals).
        v_entries = [e for e in spec.layer if e.flextrain_name == "w_v"]
        assert len(v_entries) == 1
        assert v_entries[0].optional is True


def test_gemma4_in_arch_modules() -> None:
    """Short-name lookup works."""
    from flextrain.io.arch import ARCH_MODULES, get_arch_module
    assert "gemma4" in ARCH_MODULES
    mod = get_arch_module("gemma4")
    assert mod.ARCH_NAME == "gemma4"
    assert callable(mod.BLOCK_BUILDER)
