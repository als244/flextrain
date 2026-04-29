"""Shared helpers for ``flextrain.export``.

This module is intentionally small: it knows about HF directory layout
(tokenizer / config / generation_config files), how FlexTrain stores
its weights vs. how HF stores them, and a tiny shard-writer that splits
a flat ``{name: tensor}`` dict into safetensors files at a target size.

It does NOT know anything about LoRA semantics — that lives in
``_lora_adapter.py`` / ``_merged.py``.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Iterable, Mapping

import torch

# Files we copy verbatim from the original HF directory so the exported
# checkpoint is self-contained (vLLM / sGLang / transformers can load
# directly from it without referencing the source).
_HF_AUX_FILENAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
)


def copy_hf_aux_files(src_dir: str, dst_dir: str) -> list[str]:
    """Copy the tokenizer / config / generation_config files from
    ``src_dir`` to ``dst_dir``. Silent on missing files (not every model
    ships every aux file). Returns the list of files actually copied.
    """
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"export needs the original HF dir at {src_dir!r} so it can "
            f"copy tokenizer + config files; not a directory."
        )
    os.makedirs(dst_dir, exist_ok=True)
    copied: list[str] = []
    for fn in _HF_AUX_FILENAMES:
        src = os.path.join(src_dir, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))
            copied.append(fn)
    return copied


def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def write_sharded_safetensors(
    tensors: Mapping[str, torch.Tensor],
    out_dir: str,
    *,
    shard_size_bytes: int = 5 * (1 << 30),  # 5 GiB — HF default
    base_filename: str = "model",
) -> dict:
    """Write a (potentially sharded) HF-style safetensors checkpoint.

    Parameters
    ----------
    tensors
        ``{hf_tensor_name: torch.Tensor}``. Tensors are written in the
        order returned by ``tensors.items()``; callers should pass an
        ordered mapping if shard packing matters.
    out_dir
        Destination directory. Created if missing.
    shard_size_bytes
        Soft cap on each shard's payload. We greedily pack tensors into
        the current shard until adding the next one would exceed this
        budget; oversized single tensors get their own shard.
    base_filename
        Stem for the output files. ``"model"`` produces ``model.safetensors``
        for single-shard or ``model-00001-of-NNNNN.safetensors`` for
        multi-shard, matching HF's convention.

    Returns
    -------
    dict
        Manifest with ``{"shards": [filenames], "weight_map": {name: shard}}``.
        For single-shard exports the ``weight_map`` is omitted (no index
        file is written). For multi-shard, ``model.safetensors.index.json``
        is also written into ``out_dir``.
    """
    from safetensors.torch import save_file as _save_safetensors

    os.makedirs(out_dir, exist_ok=True)

    # Greedy bin-pack. We materialize tensors lazily to avoid holding
    # the full state-dict in RAM twice; callers should pass already-cpu
    # tensors.
    shards: list[dict[str, torch.Tensor]] = [{}]
    shard_sizes: list[int] = [0]
    weight_map: dict[str, int] = {}

    for name, t in tensors.items():
        sz = _tensor_nbytes(t)
        if shard_sizes[-1] > 0 and shard_sizes[-1] + sz > shard_size_bytes:
            shards.append({})
            shard_sizes.append(0)
        shards[-1][name] = t
        shard_sizes[-1] += sz
        weight_map[name] = len(shards) - 1

    n_shards = len(shards)
    if n_shards == 1:
        path = os.path.join(out_dir, f"{base_filename}.safetensors")
        _save_safetensors(shards[0], path)
        return {"shards": [os.path.basename(path)], "weight_map": None}

    # Multi-shard: HF naming is ``model-00001-of-00007.safetensors``.
    width = max(5, len(str(n_shards)))
    shard_filenames: list[str] = []
    name_to_shard_filename: dict[str, str] = {}
    for i, shard in enumerate(shards):
        shard_idx = i + 1
        fn = (
            f"{base_filename}-"
            f"{shard_idx:0{width}d}-of-{n_shards:0{width}d}.safetensors"
        )
        shard_filenames.append(fn)
        path = os.path.join(out_dir, fn)
        _save_safetensors(shard, path)
        for k in shard.keys():
            name_to_shard_filename[k] = fn

    # Compute total payload size (uncompressed tensor bytes — what HF puts
    # in ``metadata.total_size``).
    total_size = sum(shard_sizes)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": name_to_shard_filename,
    }
    with open(
        os.path.join(out_dir, f"{base_filename}.safetensors.index.json"), "w"
    ) as f:
        json.dump(index, f, indent=2, sort_keys=True)

    return {
        "shards": shard_filenames,
        "weight_map": name_to_shard_filename,
    }


def to_cpu_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Detach + move to CPU + materialize, so ``.contiguous()`` after
    a transpose actually produces a writable buffer for safetensors."""
    out = t.detach()
    if out.device.type != "cpu":
        out = out.cpu()
    return out.contiguous()


# ---------------------------------------------------------------------------
# RoPE halved<->pair permutation.
# ---------------------------------------------------------------------------


def halved_to_pair_perm(dim: int, head_dim: int, rope_dim: int) -> torch.Tensor:
    """The same per-head halved->pair permutation FlexTrain applies on
    *load*. Use this on export to invert any tensor whose load-side
    transform was the matching ``halved_to_pair`` (Q/K weights, plus
    a LoRA-B factor whose output dim aligns with attn / kv_dim).

    Parameters
    ----------
    dim
        Total length of the axis to permute (``attn_dim`` for Q,
        ``kv_dim`` for K).
    head_dim
        Size of one head along this axis.
    rope_dim
        Number of RoPE'd channels per head. For full-rotary models this
        equals ``head_dim``; for partial-rotary (e.g. Qwen3.5) it's a
        subset of the leading channels.

    Returns
    -------
    torch.LongTensor
        Permutation index ``perm`` of length ``dim``: ``out = src[..., perm]``
        applies the halved->pair direction. Re-applying the same perm
        twice does NOT roundtrip in general (it is not self-inverse for
        partial-rotary), but we expose ``invert_perm`` below.
    """
    out = torch.arange(dim, dtype=torch.int64)
    half = rope_dim // 2
    for h in range(dim // head_dim):
        base = h * head_dim
        for i in range(half):
            out[base + 2 * i] = base + i
            out[base + 2 * i + 1] = base + half + i
    return out


def invert_perm(perm: torch.Tensor) -> torch.Tensor:
    """Inverse permutation: ``inv[perm[i]] = i``."""
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    return inv


# ---------------------------------------------------------------------------
# Pulling weights out of an ActiveModel into a flat ``(scope, name) -> tensor``
# dict, mirroring what ``ActiveModel.save_hf`` does internally. Exposing this
# here keeps export code from reaching into engine internals directly.
# ---------------------------------------------------------------------------


def collect_host_params(am) -> dict[tuple[str, str], torch.Tensor]:
    src: dict[tuple[str, str], torch.Tensor] = {}
    for name, t in am.buffers.host_embed_params.items():
        src[("embed", name)] = t
    for name, t in am.buffers.host_head_params.items():
        src[("head", name)] = t
    for i, layer_host in enumerate(am.buffers.host_params):
        scope = f"layer_{i}"
        for name, t in layer_host.items():
            src[(scope, name)] = t
    return src


def is_lora_param(name: str) -> bool:
    return name.endswith("_lora_a") or name.endswith("_lora_b")


def lora_target_from_param_name(name: str) -> str:
    """`'w_q_lora_a'` -> `'w_q'`."""
    if name.endswith("_lora_a"):
        return name[: -len("_lora_a")]
    if name.endswith("_lora_b"):
        return name[: -len("_lora_b")]
    raise ValueError(f"not a LoRA param name: {name!r}")
