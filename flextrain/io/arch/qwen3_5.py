"""Qwen3.5 family HF <-> FlexTrain mapping.

Qwen3.5 is a hybrid linear-attention + full-attention dense model with
multimodal wrapper (text-only weights live under
``model.language_model.layers.{i}.*``). For the LM training path we
ignore the vision tower entirely.

Layer layout
------------
Each layer's ``layer_types[i]`` is either ``"linear_attention"`` or
``"full_attention"``. Both share:

* ``input_layernorm.weight`` / ``post_attention_layernorm.weight``
* dense MLP (``mlp.{gate_proj, up_proj, down_proj}``)

Linear-attn layers add (under ``linear_attn.*``):

* ``in_proj_qkv``: (key_dim*2 + value_dim, hidden) — flat ``[q | k | v]``
  along output. Bundled with ``in_proj_z`` at load time into FT's
  ``w_lin_qkvz``: block-major ``[Q | K | V | Z]`` along the column axis
  (matches the MoE arch's loader, consumed zero-copy by
  ``linear_attn._split_qkvz_ft``).
* ``in_proj_z``: (value_dim, hidden) — flat ``z`` along output.
* ``in_proj_b``: (num_v_heads, hidden) and ``in_proj_a`` similarly.
  Bundled into FT's ``w_lin_ba`` as block-major ``[B | A]``.
* ``conv1d``, ``dt_bias``, ``A_log``, ``norm.weight`` (1D, no bundling).
* ``out_proj``: (hidden, value_dim).

Full-attn layers add (under ``self_attn.*``):

* ``q_proj``: (num_heads * head_dim * 2, hidden) — per-head
  interleaved ``[q_h | gate_h]``. Permuted at load time to FT's flat
  ``[Q_all | gate_all]`` layout (``GQAAttentionGatedBlock`` convention).
* ``k_proj``, ``v_proj``: (num_kv * head_dim, hidden).
* ``o_proj``: (hidden, num_heads * head_dim).
* ``q_norm.weight``, ``k_norm.weight``: per-``head_dim`` 1D vector.

Parity caveats
--------------
* HF computes Q/K with the "halved" RoPE convention (first half +
  second half along the head_dim axis). FT applies pair-interleaved
  RoPE. We permute Q/K weight matrices at post-load with the same
  ``halved->pair`` permutation used by Llama / Qwen3 dense.
* HF uses **mRoPE** (multi-axis: text + height + width) for video.
  For text-only inputs, mRoPE collapses to standard RoPE if all 3
  position-id axes carry the same text positions — which is how HF
  serves text-only inference. FT uses standard partial-RoPE; loss-
  curve parity vs HF requires the HF side to be invoked with
  text-only ``position_ids``. The training harness handles this.
* QK-norm weights: FT applies them per-``head_dim`` after slicing the
  per-head Q/K vector. We permute these with the same head-internal
  halved->pair permutation as Q/K.
* Tied embeddings: ``lm_head.weight`` is missing from safetensors
  (``tie_word_embeddings: true``); we mirror the embedding into the
  head at post-load time (same as Llama-3.2 / Qwen3 small).
"""
from __future__ import annotations

import json
import os
from typing import Any, Mapping

import torch
from safetensors import safe_open

from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


# ---------------------------------------------------------------------------
# post_load_hook: bundle split linear-attn projections, mirror tied
# embed -> head, leave Q/K halved->pair permutation to post_load_permute.
# ---------------------------------------------------------------------------


def _resolve_layer_prefix(file_index, hf_path, layer_idx: int) -> str | None:
    """Qwen3.5 multimodal wrapper: text weights are at
    ``model.language_model.layers.{i}.*``. Older / non-multimodal saves
    might use ``model.layers.{i}.*`` directly. Probe both."""
    for prefix in (
        f"model.language_model.layers.{layer_idx}",
        f"model.layers.{layer_idx}",
    ):
        for probe in (
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
        ):
            if file_index is not None:
                if probe in file_index:
                    return prefix
            else:
                # Single-shard case; just return the language_model
                # prefix and let safe_open handle the lookup.
                return prefix
    return None


def _open_for(file_index, hf_path, name):
    if file_index is not None:
        shard = file_index.get(name)
        if shard is None:
            return None
        return os.path.join(hf_path, shard)
    # Single-shard: probe all *.safetensors files in dir.
    for fn in sorted(os.listdir(hf_path)):
        if fn.endswith(".safetensors"):
            full = os.path.join(hf_path, fn)
            try:
                with safe_open(full, framework="pt", device="cpu") as f:
                    if name in f.keys():
                        return full
            except Exception:
                continue
    return None


def _qwen3_5_post_load_hook(
    hf_path: str, dest: Mapping, num_layers: int,
) -> None:
    """Two operations:

    1. **RMSNorm γ shift** (matches Qwen3-Next / Gemma 2 convention):
       ``Qwen3_5RMSNorm.forward`` does ``output * (1 + weight)``,
       i.e. the weight is stored as ``γ_canonical - 1``. FT's
       ``RMSNormBlock`` applies ``output * weight`` directly, so we
       shift every loaded RMSNorm γ by +1 at load time.
    2. **Linear-attn projection bundling**: HF stores split
       ``in_proj_qkv`` + ``in_proj_z`` + ``in_proj_b`` + ``in_proj_a``;
       FT consumes them bundled as ``w_lin_qkvz`` and ``w_lin_ba`` in
       block-major column layout (matches the MoE arch loader; see
       :func:`flextrain.nn.blocks.linear_attn.build_qkvz_perm`).

    HF layout (text-only, ignoring batch):
      in_proj_qkv : (out=key_dim*2 + value_dim, in=hidden)  flat [q | k | v]
      in_proj_z   : (out=value_dim,             in=hidden)  flat z
      in_proj_b   : (out=num_v_heads,           in=hidden)
      in_proj_a   : (out=num_v_heads,           in=hidden)

    FT layout (after transpose to (in, out)):
      w_lin_qkvz : (hidden, 2*key_dim + 2*value_dim) — block-major
                   ``[Q | K | V | Z]`` along the column axis; each
                   block laid out as (head, dim) row-major.
      w_lin_ba   : (hidden, 2*num_v_heads) — block-major ``[B | A]``.
    """
    # ----- (1) RMSNorm γ shift: HF's ``Qwen3_5RMSNorm.forward`` does
    # ``output * (1 + weight)`` (so weights are stored as γ - 1).
    # ``Qwen3_5RMSNormGated.forward`` does plain ``weight * x`` (no +1).
    # FT applies plain ``weight * x`` everywhere, so we shift the
    # non-gated norms by +1. ``w_lin_norm`` is the gated one inside
    # the linear-attn block; do NOT shift it.
    _shift_norm_names = (
        "w_attn_norm",     # input_layernorm  (Qwen3_5RMSNorm)
        "w_ffn_norm",      # post_attention_layernorm (Qwen3_5RMSNorm)
        "w_q_norm",        # per-head q_norm   (Qwen3_5RMSNorm)
        "w_k_norm",        # per-head k_norm   (Qwen3_5RMSNorm)
    )
    for L in range(num_layers):
        for n in _shift_norm_names:
            t = dest.get((f"layer_{L}", n))
            if t is not None:
                t.add_(1.0)
    final_norm = dest.get(("head", "w_final_norm"))
    if final_norm is not None:
        final_norm.add_(1.0)

    # ----- (2) Linear-attn bundling. -----
    # Find shard index (or None if single-file).
    idx_path = os.path.join(hf_path, "model.safetensors.index.json")
    if os.path.isfile(idx_path):
        with open(idx_path) as f:
            file_index = json.load(f)["weight_map"]
    else:
        file_index = None

    # Read num_k_heads / num_v_heads from config so we don't have to
    # guess. The 2B has grp=1 (num_v=num_k=16); the 9B has grp=2
    # (num_v=32, num_k=16). Bundling depends on this.
    cfg_path = os.path.join(hf_path, "config.json")
    with open(cfg_path) as f:
        hf_cfg = json.load(f)
    text_cfg = hf_cfg.get("text_config", hf_cfg)
    cfg_num_k = int(text_cfg.get("linear_num_key_heads", 16))
    cfg_num_v = int(text_cfg.get("linear_num_value_heads", 16))

    for L in range(num_layers):
        # Linear-attn entries only present for layers where the HF
        # config marks layer_types[L]=="linear_attention".
        ft_qkvz = dest.get((f"layer_{L}", "w_lin_qkvz"))
        ft_ba = dest.get((f"layer_{L}", "w_lin_ba"))
        if ft_qkvz is None or ft_ba is None:
            continue  # not a linear-attn layer
        prefix = _resolve_layer_prefix(file_index, hf_path, L)
        if prefix is None:
            continue
        in_qkv_name = f"{prefix}.linear_attn.in_proj_qkv.weight"
        in_z_name = f"{prefix}.linear_attn.in_proj_z.weight"
        in_b_name = f"{prefix}.linear_attn.in_proj_b.weight"
        in_a_name = f"{prefix}.linear_attn.in_proj_a.weight"

        shard_qkv = _open_for(file_index, hf_path, in_qkv_name)
        if shard_qkv is None:
            raise FileNotFoundError(
                f"Qwen3.5 loader: missing {in_qkv_name}"
            )
        with safe_open(shard_qkv, framework="pt", device="cpu") as f:
            hf_qkv = f.get_tensor(in_qkv_name)            # (kd2+vd, hidden)
        with safe_open(_open_for(file_index, hf_path, in_z_name),
                         framework="pt", device="cpu") as f:
            hf_z = f.get_tensor(in_z_name)                # (vd, hidden)
        with safe_open(_open_for(file_index, hf_path, in_b_name),
                         framework="pt", device="cpu") as f:
            hf_b = f.get_tensor(in_b_name)                # (num_v, hidden)
        with safe_open(_open_for(file_index, hf_path, in_a_name),
                         framework="pt", device="cpu") as f:
            hf_a = f.get_tensor(in_a_name)                # (num_v, hidden)

        ft_dtype = ft_qkvz.dtype

        # ---- Decode shapes from FT-bundled tensor sizes.
        proj_qkvz_dim = ft_qkvz.shape[1]   # = 2*key_dim + 2*value_dim
        proj_ba_dim = ft_ba.shape[1]       # = 2*num_v_heads
        hidden = ft_qkvz.shape[0]
        num_v = proj_ba_dim // 2
        kd2_plus_vd = hf_qkv.shape[0]
        # We have: kd2+vd known, vd known (=hf_z.shape[0]), so key_dim:
        value_dim = hf_z.shape[0]
        key_dim_x2 = kd2_plus_vd - value_dim
        assert key_dim_x2 % 2 == 0
        key_dim = key_dim_x2 // 2

        assert cfg_num_v == num_v, (
            f"qwen3_5 loader: ba shape implies num_v={num_v} but "
            f"config says linear_num_value_heads={cfg_num_v}"
        )
        assert num_v % cfg_num_k == 0, (
            f"linear_num_value_heads ({num_v}) must be divisible by "
            f"linear_num_key_heads ({cfg_num_k})"
        )
        assert 2 * key_dim + 2 * value_dim == proj_qkvz_dim, (
            f"qkvz bundling mismatch: key_dim={key_dim}, "
            f"value_dim={value_dim}, expected proj_qkvz_dim="
            f"{2 * key_dim + 2 * value_dim}, got {proj_qkvz_dim}"
        )

        # FT block-major column layout: ``[Q | K | V | Z]`` over the
        # column axis; each block laid out as (head, dim) row-major.
        # Matches the layout produced by the MoE arch loader and
        # consumed by ``linear_attn._split_qkvz_ft``. See
        # :func:`flextrain.nn.blocks.linear_attn.build_qkvz_perm`.
        # HF input shapes are (out, in); we transpose to (in, out) at
        # the end. ``hf_qkv`` is the fused [Q-block | K-block | V-block]
        # along its first axis already.
        q_block = hf_qkv[:key_dim, :]                           # (key_dim, hidden)
        k_block = hf_qkv[key_dim:2*key_dim, :]                  # (key_dim, hidden)
        v_block = hf_qkv[2*key_dim:, :]                         # (value_dim, hidden)
        z_block = hf_z                                          # (value_dim, hidden)
        bundled = torch.cat([q_block, k_block, v_block, z_block], dim=0)
        bundled = bundled.T.contiguous()                        # (hidden, proj_qkvz_dim)
        ft_qkvz.copy_(bundled.to(ft_dtype))

        # FT ba layout: ``[B | A]`` along columns (no per-K-head
        # structure). HF stores in_proj_b / in_proj_a as separate
        # (num_v, hidden) tensors.
        bundled_ba = torch.cat([hf_b, hf_a], dim=0)             # (2*num_v, hidden)
        bundled_ba = bundled_ba.T.contiguous()                  # (hidden, proj_ba_dim)
        ft_ba.copy_(bundled_ba.to(ft_dtype))


# ---------------------------------------------------------------------------
# Weight map entries.
# ---------------------------------------------------------------------------


# Common per-layer entries.
_COMMON = (
    WeightMapEntry(
        flextrain_name="w_attn_norm",
        hf_name="model.language_model.layers.{i}.input_layernorm.weight",
        transform=Transform.NONE,
    ),
    WeightMapEntry(
        flextrain_name="w_ffn_norm",
        hf_name="model.language_model.layers.{i}.post_attention_layernorm.weight",
        transform=Transform.NONE,
    ),
    WeightMapEntry(
        flextrain_name="w_1",
        hf_name="model.language_model.layers.{i}.mlp.gate_proj.weight",
        transform=Transform.TRANSPOSE,
    ),
    WeightMapEntry(
        flextrain_name="w_2",
        hf_name="model.language_model.layers.{i}.mlp.down_proj.weight",
        transform=Transform.TRANSPOSE,
    ),
    WeightMapEntry(
        flextrain_name="w_3",
        hf_name="model.language_model.layers.{i}.mlp.up_proj.weight",
        transform=Transform.TRANSPOSE,
    ),
)

# Linear-attn entries (only present at linear_attention layer indices).
# w_lin_qkvz / w_lin_ba populated by post_load_hook via bundling.
_LINEAR_ATTN = (
    WeightMapEntry(
        flextrain_name="w_lin_out",
        hf_name="model.language_model.layers.{i}.linear_attn.out_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_conv",
        hf_name="model.language_model.layers.{i}.linear_attn.conv1d.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_dt_bias",
        hf_name="model.language_model.layers.{i}.linear_attn.dt_bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_A_log",
        hf_name="model.language_model.layers.{i}.linear_attn.A_log",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_lin_norm",
        hf_name="model.language_model.layers.{i}.linear_attn.norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
)

# Full-attn entries (only at full_attention layer indices).
_FULL_ATTN = (
    WeightMapEntry(
        flextrain_name="w_q_norm",
        hf_name="model.language_model.layers.{i}.self_attn.q_norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_k_norm",
        hf_name="model.language_model.layers.{i}.self_attn.k_norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_q",
        hf_name="model.language_model.layers.{i}.self_attn.q_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_k",
        hf_name="model.language_model.layers.{i}.self_attn.k_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_v",
        hf_name="model.language_model.layers.{i}.self_attn.v_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="w_o",
        hf_name="model.language_model.layers.{i}.self_attn.o_proj.weight",
        transform=Transform.TRANSPOSE,
        optional=True,
    ),
)


def _qwen3_5_pre_export_hook(am, dst, num_layers: int) -> None:
    """Inverse of ``_qwen3_5_post_load_hook`` (and the linear-attn
    bundling). Three operations:

    1. **(1+w) RMSNorm shift inverse**: subtract 1.0 from every
       non-gated norm γ. HF's ``Qwen3_5RMSNorm`` forwards
       ``output * (1 + weight)``, i.e. saves ``γ_canonical - 1``;
       FT stores ``γ_canonical`` directly so the export needs the
       inverse. Applies to: ``input_layernorm``,
       ``post_attention_layernorm``, ``q_norm``, ``k_norm``, and the
       backbone-level ``model.language_model.norm.weight``. The
       linear-attn norm (``linear_attn.norm.weight``) is the GATED
       variant and must NOT be shifted.
    2. **Linear-attn unbundle**: split FT ``w_lin_qkvz`` (block-major
       ``[Q | K | V | Z]``) and ``w_lin_ba`` (block-major ``[B | A]``)
       back into HF's split per-K-head ``in_proj_qkv`` /
       ``in_proj_z`` / ``in_proj_b`` / ``in_proj_a``.
    3. **Tied embedding**: drop ``lm_head.weight`` when the source
       config has ``tie_word_embeddings: True`` (Qwen3.5-2B,
       Qwen3.5-0.8B-Base). HF will re-mirror at load.

    Note: the per-head Q/K halved->pair RoPE permutation and the
    per-head ``[q_h | gate_h]`` -> flat ``[Q_all | gate_all]``
    layout split are handled by ``_post_export_permute_for_arch``
    in ``flextrain.export._hf_full`` (see ``_invert_qwen3_5_*``).
    """
    from flextrain.export._pre_export_helpers import (
        emit_linear_attn_unbundle,
        read_tie_word_embeddings,
        subtract_one_from_norms,
    )
    # ----- (1) Norm γ shift inverse. -----
    norm_names: list[str] = []
    for L in range(num_layers):
        norm_names.append(
            f"model.language_model.layers.{L}.input_layernorm.weight"
        )
        norm_names.append(
            f"model.language_model.layers.{L}.post_attention_layernorm.weight"
        )
        norm_names.append(
            f"model.language_model.layers.{L}.self_attn.q_norm.weight"
        )
        norm_names.append(
            f"model.language_model.layers.{L}.self_attn.k_norm.weight"
        )
    norm_names.append("model.language_model.norm.weight")
    subtract_one_from_norms(dst, norm_names)

    # ----- (2) Linear-attn unbundle. -----
    # Read linear-attn dim config from am.dims (populated by
    # hf_config_to_flextrain at load time).
    dims = am.dims
    num_v = int(dims.get("linear_num_v_heads", dims.get("num_v_heads", 16)))
    num_k = int(dims.get("linear_num_k_heads", dims.get("num_k_heads", 16)))
    head_k = int(dims.get("linear_head_k_dim", dims.get("head_k_dim", 128)))
    head_v = int(dims.get("linear_head_v_dim", dims.get("head_v_dim", 128)))
    for L in range(num_layers):
        host = am.buffers.host_params[L]
        w_qkvz = host.get("w_lin_qkvz")
        w_ba = host.get("w_lin_ba")
        if w_qkvz is None or w_ba is None:
            continue   # not a linear-attn layer (full-attn layer)
        emit_linear_attn_unbundle(
            dst,
            layer_prefix=f"model.language_model.layers.{L}",
            w_lin_qkvz=w_qkvz,
            w_lin_ba=w_ba,
            num_k_heads=num_k,
            num_v_heads=num_v,
            head_k_dim=head_k,
            head_v_dim=head_v,
        )

    # ----- (3) Tied embedding. -----
    if read_tie_word_embeddings(am):
        dst.pop("lm_head.weight", None)


# ---------------------------------------------------------------------------
# Vision-tower weight map (Qwen3.5/3.6 multimodal). HF stores them under
# ``model.visual.*``. Flextrain places them in scope ``"embed"`` with the
# encoder-id prefix ``"image0_"`` so they live in the merged
# ``MultimodalInputLayer.param_spec`` (which the engine maps to the
# always-on-GPU ``gpu_embed_params`` dict). All entries are
# ``Transform.NONE`` -- the Qwen-VL vision encoder consumes HF's
# native ``(out, in)`` Linear weight layout via ``F.linear``.
# ``optional=True`` keeps the loader from faulting when the encoder
# isn't constructed (text-only mode).
# ---------------------------------------------------------------------------


_VISION_EMBED = (
    WeightMapEntry(
        flextrain_name="image0_patch_embed_proj_w",
        hf_name="model.visual.patch_embed.proj.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_patch_embed_proj_b",
        hf_name="model.visual.patch_embed.proj.bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_pos_embed_w",
        hf_name="model.visual.pos_embed.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_merger_norm_w",
        hf_name="model.visual.merger.norm.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_merger_norm_b",
        hf_name="model.visual.merger.norm.bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_merger_fc1_w",
        hf_name="model.visual.merger.linear_fc1.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_merger_fc1_b",
        hf_name="model.visual.merger.linear_fc1.bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_merger_fc2_w",
        hf_name="model.visual.merger.linear_fc2.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_merger_fc2_b",
        hf_name="model.visual.merger.linear_fc2.bias",
        transform=Transform.NONE,
        optional=True,
    ),
)


_VISION_LAYER = (
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_norm1_w",
        hf_name="model.visual.blocks.{i}.norm1.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_norm1_b",
        hf_name="model.visual.blocks.{i}.norm1.bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_norm2_w",
        hf_name="model.visual.blocks.{i}.norm2.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_norm2_b",
        hf_name="model.visual.blocks.{i}.norm2.bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_qkv_w",
        hf_name="model.visual.blocks.{i}.attn.qkv.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_qkv_b",
        hf_name="model.visual.blocks.{i}.attn.qkv.bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_proj_w",
        hf_name="model.visual.blocks.{i}.attn.proj.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_proj_b",
        hf_name="model.visual.blocks.{i}.attn.proj.bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_mlp_fc1_w",
        hf_name="model.visual.blocks.{i}.mlp.linear_fc1.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_mlp_fc1_b",
        hf_name="model.visual.blocks.{i}.mlp.linear_fc1.bias",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_mlp_fc2_w",
        hf_name="model.visual.blocks.{i}.mlp.linear_fc2.weight",
        transform=Transform.NONE,
        optional=True,
    ),
    WeightMapEntry(
        flextrain_name="image0_layer_{i}_mlp_fc2_b",
        hf_name="model.visual.blocks.{i}.mlp.linear_fc2.bias",
        transform=Transform.NONE,
        optional=True,
    ),
)


QWEN3_5_ARCH = ArchSpec(
    hf_arch_ids=("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"),
    embed=(
        WeightMapEntry(
            flextrain_name="w_tok_embeddings",
            hf_name="model.language_model.embed_tokens.weight",
            transform=Transform.NONE,
        ),
    ),
    head=(
        WeightMapEntry(
            flextrain_name="w_final_norm",
            hf_name="model.language_model.norm.weight",
            transform=Transform.NONE,
        ),
        WeightMapEntry(
            flextrain_name="w_head_proj",
            # tied with embed; missing from shard files. Marked
            # optional so strict=True doesn't blow up. post_load_permute
            # mirrors embed.t() into the head if needed.
            hf_name="lm_head.weight",
            transform=Transform.TRANSPOSE,
            optional=True,
        ),
    ),
    layer=_COMMON + _LINEAR_ATTN + _FULL_ATTN,
    vision_embed=_VISION_EMBED,
    vision_layer=_VISION_LAYER,
    post_load_hook=_qwen3_5_post_load_hook,
    pre_export_hook=_qwen3_5_pre_export_hook,
)

register_arch(QWEN3_5_ARCH)


# ---------------------------------------------------------------------------
# Config translation.
# ---------------------------------------------------------------------------


def _text_cfg(hf_config):
    """Pull the text section out of the multimodal wrapper. Falls back
    to top-level for non-multimodal saves."""
    if isinstance(hf_config, dict):
        return hf_config.get("text_config", hf_config)
    return getattr(hf_config, "text_config", hf_config)


ARCH_NAME = "qwen3_5"


_REQUIRED_DIMS = (
    "vocab_size", "n_layers", "d_model", "n_heads", "head_dim", "expert_dim",
)
_DEFAULT_DATATYPES = {
    "embed": "bfloat16", "head_proj": "bfloat16", "attn_proj": "bfloat16",
    "expert_proj": "bfloat16", "router": "bfloat16", "norm": "bfloat16",
    "residual": "bfloat16",
}


def expand_dims(dims) -> dict:
    """Qwen3.5 dense dims schema = Llama + linear-attn primaries +
    optional ``layer_pattern`` shorthand.

    Required: ``vocab_size, n_layers, d_model, n_heads, head_dim,
    expert_dim``.

    Linear-attn primaries (default match the Qwen3.5 dense reference):
    ``linear_num_v_heads=16``, ``linear_num_k_heads=16``,
    ``linear_head_k_dim=128``, ``linear_head_v_dim=128``,
    ``linear_conv_kernel=4``.

    ``layer_pattern`` (optional): a shorthand string like ``"3L1F"``
    that ``flextrain.from_dims`` expands into ``hyperparams["layer_types"]``
    of length ``n_layers``. Without it the user must supply
    ``hyperparams["layer_types"]`` directly.
    """
    out = dict(dims)
    missing = [k for k in _REQUIRED_DIMS if k not in out]
    if missing:
        raise KeyError(
            f"qwen3_5 dims missing required keys: {missing}. "
            f"Got keys: {sorted(out)}"
        )
    out.setdefault("n_kv_heads", out["n_heads"])
    out.setdefault("num_shared_experts", 1)
    out.setdefault("num_routed_experts", 0)
    out.setdefault("top_k", 0)
    out.setdefault("is_causal", True)
    out.setdefault("datatypes", dict(_DEFAULT_DATATYPES))
    # Linear-attn primaries.
    out.setdefault("linear_num_v_heads", 16)
    out.setdefault("linear_num_k_heads", 16)
    out.setdefault("linear_head_k_dim", 128)
    out.setdefault("linear_head_v_dim", 128)
    out.setdefault("linear_conv_kernel", 4)
    # Standard derived attn dims.
    out["attn_dim"] = int(out["n_heads"]) * int(out["head_dim"])
    out["kv_dim"] = int(out["n_kv_heads"]) * int(out["head_dim"])
    # Linear-attn aliases + derived dims (the GatedDeltaNet param spec
    # reads several of these — match what hf_config_to_flextrain emits).
    nv = int(out["linear_num_v_heads"])
    nk = int(out["linear_num_k_heads"])
    hk = int(out["linear_head_k_dim"])
    hv = int(out["linear_head_v_dim"])
    ck = int(out["linear_conv_kernel"])
    key_dim = nk * hk
    value_dim = nv * hv
    out.setdefault("num_v_heads", nv)
    out.setdefault("num_k_heads", nk)
    out.setdefault("head_k_dim", hk)
    out.setdefault("head_v_dim", hv)
    out.setdefault("key_dim", key_dim)
    out.setdefault("value_dim", value_dim)
    out.setdefault("conv_dim", 2 * key_dim + value_dim)
    out.setdefault("proj_qkvz_dim", 2 * key_dim + 2 * value_dim)
    out.setdefault("proj_ba_dim", 2 * nv)
    out.setdefault("conv_kernel_size", ck)
    return out


def default_hyperparams() -> dict:
    """Qwen3.5 dense defaults: eps=1e-6, rope=1e7, partial_rotary=0.25.
    ``layer_types=None`` means the caller must either set it explicitly
    via ``hyperparams`` or pass ``dims["layer_pattern"]`` for shorthand
    expansion."""
    return {
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000_000.0,
        "rope_scaling": None,
        "partial_rotary_factor": 0.25,
        "layer_types": None,
        "full_attention_interval": None,
        "attn_output_gate": True,
    }


def flextrain_to_hf_config(dims, hyperparams=None) -> dict:
    """Inverse mapping for Qwen3.5 dense. Emits ``layer_types`` (if set)
    and the linear-attn primaries (under HF's
    ``linear_num_value_heads`` / ``linear_num_key_heads`` / etc names)."""
    hp = dict(default_hyperparams())
    if hyperparams:
        hp.update(hyperparams)
    d = expand_dims(dims)
    cfg = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "vocab_size": int(d["vocab_size"]),
        "num_hidden_layers": int(d["n_layers"]),
        "hidden_size": int(d["d_model"]),
        "num_attention_heads": int(d["n_heads"]),
        "num_key_value_heads": int(d["n_kv_heads"]),
        "head_dim": int(d["head_dim"]),
        "intermediate_size": int(d["expert_dim"]),
        "linear_num_value_heads": int(d["linear_num_v_heads"]),
        "linear_num_key_heads": int(d["linear_num_k_heads"]),
        "linear_key_head_dim": int(d["linear_head_k_dim"]),
        "linear_value_head_dim": int(d["linear_head_v_dim"]),
        "linear_conv_kernel_dim": int(d["linear_conv_kernel"]),
        "rms_norm_eps": float(hp["rms_norm_eps"]),
        "rope_theta": float(hp["rope_theta"]),
        "rope_scaling": hp.get("rope_scaling"),
        "partial_rotary_factor": float(hp["partial_rotary_factor"]),
        "max_position_embeddings": 32768,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "attn_output_gate": bool(hp.get("attn_output_gate", True)),
        "initializer_range": 0.02,
        "torch_dtype": "bfloat16",
        "use_cache": True,
    }
    if hp.get("layer_types"):
        cfg["layer_types"] = list(hp["layer_types"])
    if hp.get("full_attention_interval") is not None:
        cfg["full_attention_interval"] = int(hp["full_attention_interval"])
    return cfg


def hf_config_to_flextrain(hf_config: Any) -> dict:
    tc = _text_cfg(hf_config)
    g = (tc.get if isinstance(tc, dict) else lambda k, default=None: getattr(tc, k, default))
    n_heads = g("num_attention_heads")
    hidden = g("hidden_size")
    head_dim = g("head_dim") or (hidden // n_heads)
    # Linear-attn dim derivation (also consumed by linear_attn.param_spec
    # via the model-wide dims dict).
    num_v_heads = g("linear_num_value_heads", 16)
    num_k_heads = g("linear_num_key_heads", 16)
    head_k_dim = g("linear_key_head_dim", 128)
    head_v_dim = g("linear_value_head_dim", 128)
    conv_kernel = g("linear_conv_kernel_dim", 4)
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    return {
        "vocab_size": g("vocab_size"),
        "n_layers": g("num_hidden_layers"),
        "d_model": hidden,
        "n_heads": n_heads,
        "n_kv_heads": g("num_key_value_heads") or n_heads,
        "head_dim": head_dim,
        "expert_dim": g("intermediate_size"),
        # Dense MLP (no MoE).
        "num_shared_experts": 1,
        "num_routed_experts": 0,
        "top_k": 0,
        "is_causal": True,
        # Linear-attn config dims.
        "linear_num_v_heads": num_v_heads,
        "linear_num_k_heads": num_k_heads,
        "linear_head_k_dim": head_k_dim,
        "linear_head_v_dim": head_v_dim,
        "linear_conv_kernel": conv_kernel,
        # Linear-attn derived dims (used by GatedDeltaNetBlock.param_spec).
        "num_v_heads": num_v_heads,
        "num_k_heads": num_k_heads,
        "head_k_dim": head_k_dim,
        "head_v_dim": head_v_dim,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "conv_dim": 2 * key_dim + value_dim,  # qkv concat (no z) -> conv input
        "proj_qkvz_dim": 2 * key_dim + 2 * value_dim,
        "proj_ba_dim": 2 * num_v_heads,
        "conv_kernel_size": conv_kernel,
        "datatypes": {
            "embed": "bfloat16", "head_proj": "bfloat16",
            "attn_proj": "bfloat16", "expert_proj": "bfloat16",
            "router": "bfloat16", "norm": "bfloat16",
            "residual": "bfloat16",
        },
    }


def hf_config_to_hyperparams(hf_config: Any) -> dict:
    tc = _text_cfg(hf_config)
    g = (tc.get if isinstance(tc, dict) else lambda k, default=None: getattr(tc, k, default))
    rope_params = g("rope_parameters") or {}
    is_rope_dict = isinstance(rope_params, dict)
    rope_theta = (
        rope_params.get("rope_theta") if is_rope_dict else None
    ) or g("rope_theta", 10_000_000.0)
    partial_rotary = (
        rope_params.get("partial_rotary_factor")
        if is_rope_dict else None
    ) or g("partial_rotary_factor", 0.25)
    # MRoPE knobs -- Qwen3.5/3.6 multimodal. Present in text_config.
    # ``mrope_section`` length-3 tuple summing to ``rot_dim // 2``.
    # ``mrope_interleaved`` controls per-pair axis layout (HF interleaved
    # variant vs contiguous sections). Both default to None / False;
    # text-only training is unaffected (block dispatches on
    # ``seq_positions.shape[-1]``, not on these fields).
    mrope_section_raw = (
        rope_params.get("mrope_section") if is_rope_dict else None
    )
    mrope_section: tuple[int, int, int] | None = None
    if mrope_section_raw is not None:
        if len(mrope_section_raw) != 3:
            raise ValueError(
                f"qwen3.5 mrope_section must have length 3, got "
                f"{mrope_section_raw!r}"
            )
        mrope_section = tuple(int(x) for x in mrope_section_raw)
    mrope_interleaved = bool(
        rope_params.get("mrope_interleaved", False) if is_rope_dict else False
    )
    return {
        "rms_norm_eps": g("rms_norm_eps", 1e-6),
        "rope_theta": rope_theta,
        "partial_rotary_factor": partial_rotary,
        "mrope_section": mrope_section,
        "mrope_interleaved": mrope_interleaved,
        "layer_types": g("layer_types"),
        "full_attention_interval": g("full_attention_interval"),
        "attn_output_gate": g("attn_output_gate", True),
    }


# ---------------------------------------------------------------------------
# Block builder.
# ---------------------------------------------------------------------------


def _qwen3_5_block_builder(layer_idx: int, ctx) -> object:
    from flextrain.nn.layers.qwen3_5 import (
        Qwen3_5LayerConfig, Qwen3_5FullLayer, Qwen3_5LinearLayer,
    )

    dims = ctx.dims
    hp = ctx.hyperparams
    layer_types = hp.get("layer_types") or []
    if layer_idx >= len(layer_types):
        raise ValueError(
            f"Qwen3.5 layer {layer_idx}: layer_types list has only "
            f"{len(layer_types)} entries"
        )
    is_full = (layer_types[layer_idx] == "full_attention")

    block_cfg = Qwen3_5LayerConfig(
        d_model=int(dims["d_model"]),
        expert_dim=int(dims["expert_dim"]),
        n_heads=int(dims["n_heads"]),
        n_kv_heads=int(dims["n_kv_heads"]),
        head_dim=int(dims["head_dim"]),
        linear_num_v_heads=int(dims.get("linear_num_v_heads", 16)),
        linear_num_k_heads=int(dims.get("linear_num_k_heads", 16)),
        linear_head_k_dim=int(dims.get("linear_head_k_dim", 128)),
        linear_head_v_dim=int(dims.get("linear_head_v_dim", 128)),
        linear_conv_kernel=int(dims.get("linear_conv_kernel", 4)),
        rms_norm_eps=float(hp.get("rms_norm_eps", 1e-6)),
        rope_base=float(hp.get("rope_theta", 10_000_000.0)),
        is_causal=True,
        partial_rotary_factor=float(hp.get("partial_rotary_factor", 0.25)),
        mrope_section=hp.get("mrope_section"),
        mrope_interleaved=bool(hp.get("mrope_interleaved", False)),
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )

    if is_full:
        base = Qwen3_5FullLayer(layer_id=layer_idx, cfg=block_cfg)
    else:
        base = Qwen3_5LinearLayer(layer_id=layer_idx, cfg=block_cfg)

    if not ctx.lora_targets:
        return base
    from flextrain.nn.layers.lora_wrapper import (
        LoRAWrapperLayer, _discover_lora_eligible_names,
    )
    layer_dims = dict(
        dims,
        attn_dim=int(dims["n_heads"]) * int(dims["head_dim"]),
        kv_dim=int(dims["n_kv_heads"]) * int(dims["head_dim"]),
    )
    # Hybrid backbone: linear-attn layers have no w_q/w_k/w_v/w_o, so an
    # explicit target list like ("w_q","w_k","w_v","w_o") would fail
    # the wrapper's strict validation. Filter the user's targets to the
    # subset eligible on this layer (mirrors HF PEFT's per-layer
    # auto-skip semantics). When the filtered set is empty (linear-attn
    # under attn-only LoRA), still freeze the base params so the
    # working-set + buffer manager skip grad / opt-state allocation.
    if ctx.lora_targets != "all" and ctx.lora_targets is not None:
        eligible = set(_discover_lora_eligible_names(base.param_spec, layer_dims))
        kept = tuple(t for t in ctx.lora_targets if t in eligible)
        if not kept:
            from dataclasses import replace as _replace
            from flextrain.core.layer import ParamSpec
            base.param_spec = ParamSpec(tensors=tuple(
                _replace(t, frozen=True) for t in base.param_spec.tensors
            ))
            return base
        targets_for_layer: object = kept
    else:
        targets_for_layer = ctx.lora_targets
    return LoRAWrapperLayer(
        base, lora_targets=targets_for_layer,
        rank=ctx.lora_rank, alpha=ctx.lora_alpha,
        dims=layer_dims,
        adapter_compute_dtype=ctx.lora_adapter_compute_dtype,
        adapter_master_dtype=ctx.lora_adapter_master_dtype,
        adapter_grad_dtype=ctx.lora_adapter_grad_dtype,
        adapter_opt_state_dtype=ctx.lora_adapter_opt_state_dtype,
    )


# ---------------------------------------------------------------------------
# post_load_permute: Q/K halved->pair, QK-norm permutation, q_proj
# per-head [q|gate] -> flat [Q | gate], tied embed -> head.
# ---------------------------------------------------------------------------


def post_load_permute(am, hf_config, dims, hyperparams):
    """Three operations:

    1. **Full-attn ``w_q`` permutation**: HF stores per-head
       ``[q_h | gate_h]`` along the output axis (8*256*2 = 4096 with
       per-head interleaving). FT expects flat ``[Q_all | gate_all]``
       (first 2048 dims = Q across all heads, next 2048 dims = gate).
       Permute the output axis.
    2. **Q/K halved->pair RoPE permutation**: same as Llama / Qwen3
       dense. Applied per-head to the first ``rope_dim = head_dim *
       partial_rotary_factor`` channels of each head's slice.
    3. **Tied embedding mirror**: Qwen3.5 has ``tie_word_embeddings:
       True``; the head's ``w_head_proj`` slot exists in FT but no
       safetensors weight is present. Mirror ``embed.t()`` into the
       head buffer.
    """
    head_dim = int(dims["head_dim"])
    n_heads = int(dims["n_heads"])
    n_kv = int(dims["n_kv_heads"])
    n_layers = int(dims["n_layers"])
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    layer_types = hyperparams.get("layer_types") or []
    partial_rotary_factor = float(
        hyperparams.get("partial_rotary_factor", 0.25)
    )
    rope_dim = int(head_dim * partial_rotary_factor)

    def _halved_to_pair(dim: int, hd: int) -> torch.Tensor:
        """Per-(hd)-block halved->pair permutation. Applies only to
        the first ``rope_dim`` channels of each block; the trailing
        ``hd - rope_dim`` channels (no-RoPE pass-through) stay put."""
        out = torch.arange(dim, dtype=torch.int64)
        half = rope_dim // 2
        for h in range(dim // hd):
            base = h * hd
            for i in range(half):
                out[base + 2 * i] = base + i
                out[base + 2 * i + 1] = base + half + i
        return out

    q_perm = _halved_to_pair(attn_dim, head_dim)
    k_perm = _halved_to_pair(kv_dim, head_dim)
    qk_norm_perm = _halved_to_pair(head_dim, head_dim)

    def _split_perm_q_with_gate(host_t, n_heads: int, head_dim: int):
        """HF stores ``w_q`` with the doubled out-dim arranged per-head:
        each head's chunk is ``[q_h(head_dim) | gate_h(head_dim)]``,
        flattened to ``(in, n_heads * 2 * head_dim) = (in, attn_dim*2)``.

        FT expects flat ``[Q_all(attn_dim) | gate_all(attn_dim)]``.

        We additionally apply the halved->pair RoPE permutation to the
        Q half (gate half doesn't go through RoPE).

        ``host_t`` is the FT-loaded tensor (already TRANSPOSE-d to
        ``(in, n_heads * 2 * head_dim)``)."""
        in_dim, out_dim = host_t.shape
        assert out_dim == n_heads * 2 * head_dim
        # Reshape to (in, n_heads, 2*head_dim) where the inner 2*head_dim
        # is per-head [q | gate].
        v = host_t.reshape(in_dim, n_heads, 2 * head_dim)
        q_part = v[:, :, :head_dim].contiguous()       # (in, n_heads, head_dim)
        gate_part = v[:, :, head_dim:].contiguous()    # (in, n_heads, head_dim)
        # Apply RoPE halved->pair to Q only.
        q_flat = q_part.reshape(in_dim, n_heads * head_dim)
        q_flat = q_flat[:, q_perm].contiguous()
        gate_flat = gate_part.reshape(in_dim, n_heads * head_dim)
        # FT layout: [Q_all | gate_all]
        return torch.cat([q_flat, gate_flat], dim=-1)

    for L in range(n_layers):
        if L >= len(layer_types):
            continue
        host = am.buffers.host_params[L]
        if layer_types[L] != "full_attention":
            continue  # linear-attn layers don't have w_q / w_k / w_q_norm.

        # 1+2. w_q: split per-head [q|gate] -> flat [Q|gate], with
        # halved->pair on Q.
        if "w_q" in host:
            new_w_q = _split_perm_q_with_gate(host["w_q"], n_heads, head_dim)
            host["w_q"].copy_(new_w_q)

        # w_k: simple halved->pair.
        if "w_k" in host:
            host["w_k"].copy_(host["w_k"][:, k_perm])

        # QK-norm vectors (per head_dim).
        for nm in ("w_q_norm", "w_k_norm"):
            if nm in host:
                w = host[nm]
                # Some saves store as (head_dim,); we apply the head-
                # internal permutation.
                if w.dim() == 1 and w.shape[0] == head_dim:
                    host[nm].copy_(w[qk_norm_perm])

        # LoRA B mirrors w_q permutation if present (rare for w_q with
        # the doubled-gate layout; the LoRA wrapper's effective W'
        # already has the doubled output, so its B factor mirrors).
        # Note: LoRA on a gated q_proj is unusual; if used, we'd need
        # to apply the same per-head split + halved->pair on B. Skip
        # for now -- LoRA on Qwen3.5 attn defers to a later pass.

    # 3. Tied embedding mirror.
    tied = hyperparams.get("tie_word_embeddings")
    if tied is None:
        # Read from raw config if hyperparams didn't surface it.
        if isinstance(hf_config, dict):
            tied = hf_config.get("tie_word_embeddings")
        else:
            tied = getattr(hf_config, "tie_word_embeddings", False)
    if tied:
        head_w = am.buffers.host_head_params.get("w_head_proj")
        embed_w = am.buffers.host_embed_params.get("w_tok_embeddings")
        if head_w is not None and embed_w is not None:
            # Embed is (vocab, hidden); head is (hidden, vocab) post-
            # transpose.
            head_w.copy_(embed_w.t().to(head_w.dtype))

    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        if name in am.buffers.host_head_params:
            dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Block-builder registration.
# ---------------------------------------------------------------------------


def _register_builder() -> None:
    from flextrain.api import register_block_builder
    register_block_builder(
        ("Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration"),
        _qwen3_5_block_builder,
    )


_register_builder()


# Public block-builder symbol used by ``flextrain.from_dims`` via
# ``flextrain.io.arch.ARCH_MODULES``. Same function the registry above
# hands to from_pretrained, exposed at module level so the dims path
# can look it up by short name.
BLOCK_BUILDER = _qwen3_5_block_builder


# ---------------------------------------------------------------------------
# Multimodal factories (Phase 1: Qwen-VL vision tower, frozen, ConcatSplice).
#
# ``hf_config_to_vision_dims`` returns a dims dict when the checkpoint
# contains a vision_config; ``None`` for text-only saves. The api's
# ``_build_active_model`` calls this to decide whether to construct a
# :class:`MultimodalInputLayer` or fall through to the regular
# :class:`TokenEmbedLayer`.
# ---------------------------------------------------------------------------


def hf_config_to_vision_dims(hf_config) -> dict | None:
    """Extract vision-tower dims from an HF config, or return None.

    Returns the same dataclass-style ``dict`` that
    :func:`build_modality_encoders` consumes -- field names match
    :class:`~flextrain.nn.encoders.qwen_vl_vit.QwenVLVisionConfig`.
    """
    if isinstance(hf_config, dict):
        vc = hf_config.get("vision_config")
    else:
        vc = getattr(hf_config, "vision_config", None)
    if vc is None:
        return None
    g = (
        vc.get if isinstance(vc, dict)
        else lambda k, default=None: getattr(vc, k, default)
    )
    # ``model_type`` is informational; for Qwen3.5/3.6 it's "qwen3_5"
    # and for Qwen3-VL proper it's "qwen3_vl". Both use the same
    # Qwen-VL vision tower architecture so they share this builder.
    return {
        "depth": int(g("depth", g("num_hidden_layers", 24))),
        "hidden_size": int(g("hidden_size", 1024)),
        "intermediate_size": int(g("intermediate_size", 4096)),
        "num_heads": int(g("num_heads", g("num_attention_heads", 16))),
        "in_channels": int(g("in_channels", 3)),
        "patch_size": int(g("patch_size", 16)),
        "spatial_merge_size": int(g("spatial_merge_size", 2)),
        "temporal_patch_size": int(g("temporal_patch_size", 2)),
        "out_hidden_size": int(g("out_hidden_size", 2048)),
        "num_position_embeddings": int(g("num_position_embeddings", 2304)),
        "hidden_act": str(g("hidden_act", "gelu_pytorch_tanh")),
        "deepstack_visual_indexes": tuple(g("deepstack_visual_indexes", ()) or ()),
    }


def build_modality_encoders(
    vision_dims: dict,
    hyperparams: dict | None = None,
    *,
    compute_dtype=None,
    frozen: bool = True,
):
    """Construct the tuple of :class:`ModalityEncoder` instances for
    Qwen3.5/3.6 multimodal training.

    Phase 1 returns exactly one encoder (image, encoder_id=0). The
    tuple shape is used so multi-encoder configs (e.g. add an audio
    encoder later) plug in additively.
    """
    import torch
    from flextrain.nn.encoders import QwenVLVisionConfig, QwenVLVisionEncoder

    _ = hyperparams  # currently unused; Qwen-VL ViT has no per-layer hp.
    deepstack = tuple(int(i) for i in vision_dims.get("deepstack_visual_indexes", ()) or ())
    if deepstack:
        # Phase 2 fast-follow (Qwen3-VL proper has [8, 16, 24]).
        raise NotImplementedError(
            "Qwen-VL deepstack is a Phase 2 item; build_modality_encoders "
            f"got deepstack_visual_indexes={deepstack!r}. Set them to [] "
            "in the HF config (Qwen3.5/3.6 default) or wait for Phase 2."
        )
    cfg = QwenVLVisionConfig(
        depth=int(vision_dims["depth"]),
        hidden_size=int(vision_dims["hidden_size"]),
        intermediate_size=int(vision_dims["intermediate_size"]),
        num_heads=int(vision_dims["num_heads"]),
        in_channels=int(vision_dims.get("in_channels", 3)),
        patch_size=int(vision_dims["patch_size"]),
        spatial_merge_size=int(vision_dims["spatial_merge_size"]),
        temporal_patch_size=int(vision_dims["temporal_patch_size"]),
        out_hidden_size=int(vision_dims["out_hidden_size"]),
        num_position_embeddings=int(vision_dims["num_position_embeddings"]),
        hidden_act=str(vision_dims.get("hidden_act", "gelu_pytorch_tanh")),
        compute_dtype=compute_dtype or torch.bfloat16,
        attn_implementation=str(vision_dims.get("attn_implementation", "sdpa")),
    )
    return (
        QwenVLVisionEncoder(cfg, modality="image", encoder_id=0, frozen=frozen),
    )


def modality_splice_strategies(vision_dims: dict, hyperparams: dict | None = None):
    """Return one ``(splice_fwd, splice_bwd)`` pair per encoder built
    by :func:`build_modality_encoders`. Phase 1: ConcatSplice for the
    single image encoder."""
    _ = vision_dims, hyperparams  # unused for ConcatSplice
    from flextrain.nn.splices import concat_splice_bwd, concat_splice_fwd
    return ((concat_splice_fwd, concat_splice_bwd),)
