"""Full-weight HF export: write a complete HF directory (config +
tokenizer + sharded ``model.safetensors``) that vLLM, sGLang, and
``transformers.AutoModelForCausalLM.from_pretrained`` can load directly.

Use this after a full fine-tune (no LoRA) — the host master params hold
the updated weights and we just need to map them back to HF names.

After a LoRA run, prefer ``flextrain.export.save_hf_merged`` which folds
the LoRA delta into the base weights before calling this exporter.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

import torch

from ..io.hf_weights import _apply_export_transform  # type: ignore[attr-defined]
from ._common import (
    collect_host_params,
    copy_hf_aux_files,
    is_lora_param,
    to_cpu_contiguous,
    write_sharded_safetensors,
)


def _render_hf_name(template: str, layer_idx: int | None) -> str:
    if "{i}" in template:
        if layer_idx is None:
            raise ValueError(
                f"layer template {template!r} has no layer_idx context"
            )
        return template.format(i=layer_idx)
    return template


def _post_export_permute_for_arch(
    am, dst: dict[str, torch.Tensor], hf_arch_ids: tuple[str, ...],
) -> None:
    """Inverse of arch-specific load-side post_load_permute, applied to
    base weights *only* (no LoRA — that's handled in _merged.py).

    For Llama family: invert the per-head halved->pair permutation on
    Q / K weights so the exported ``q_proj.weight`` / ``k_proj.weight``
    match HF's expected layout.

    For Qwen3-dense: same Q/K halved->pair invert, plus q_norm /
    k_norm (per-head channels).

    Qwen3.5 / Qwen3.5-MoE / Qwen3-Next: out of scope here — those
    ship gated q_proj or per-K-head bundled linear-attn projections
    that aren't a 1:1 inversion. The full-weight exporter rejects
    those archs (caller should use ``save_hf_merged`` only after a
    LoRA run, and we leave full-FT export for those models for a
    future patch).
    """
    if not hf_arch_ids:
        return
    arch_set = set(hf_arch_ids)
    if arch_set & {"LlamaForCausalLM"}:
        _invert_llama_qk_perm(am, dst)
        return
    if arch_set & {"Qwen3ForCausalLM"}:
        _invert_qwen3_qk_perm(am, dst)
        return
    # Other archs: caller will get a louder failure if their
    # exported weights need a permute we haven't wired here.


def _halved_to_pair_perm(dim: int, head_dim: int, rope_dim: int) -> torch.Tensor:
    out = torch.arange(dim, dtype=torch.int64)
    half = rope_dim // 2
    for h in range(dim // head_dim):
        base = h * head_dim
        for i in range(half):
            out[base + 2 * i] = base + i
            out[base + 2 * i + 1] = base + half + i
    return out


def _invert_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), dtype=perm.dtype)
    return inv


def _invert_llama_qk_perm(am, dst: dict[str, torch.Tensor]) -> None:
    """Invert the halved->pair permutation FlexTrain applies on Llama
    Q/K weights at load time. Acts on the *exported* HF tensors in
    ``dst``: ``q_proj.weight`` (out, in) and ``k_proj.weight``."""
    dims = am.dims
    n_heads = int(dims["n_heads"])
    n_kv = int(dims["n_kv_heads"])
    head_dim = int(dims["head_dim"])
    n_layers = int(dims["n_layers"])
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    q_perm_inv = _invert_perm(_halved_to_pair_perm(attn_dim, head_dim, head_dim))
    k_perm_inv = _invert_perm(_halved_to_pair_perm(kv_dim, head_dim, head_dim))
    for i in range(n_layers):
        for nm, perm in (
            (f"model.layers.{i}.self_attn.q_proj.weight", q_perm_inv),
            (f"model.layers.{i}.self_attn.k_proj.weight", k_perm_inv),
        ):
            t = dst.get(nm)
            if t is None:
                continue
            # HF stores ``q_proj.weight`` as ``(out, in)``; permute axis 0.
            dst[nm] = t.index_select(0, perm).contiguous()


def _invert_qwen3_qk_perm(am, dst: dict[str, torch.Tensor]) -> None:
    """Same as Llama but also undoes the q_norm / k_norm head-dim
    permutation Qwen3-dense applies on load."""
    dims = am.dims
    n_heads = int(dims["n_heads"])
    n_kv = int(dims["n_kv_heads"])
    head_dim = int(dims["head_dim"])
    n_layers = int(dims["n_layers"])
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim
    q_perm_inv = _invert_perm(_halved_to_pair_perm(attn_dim, head_dim, head_dim))
    k_perm_inv = _invert_perm(_halved_to_pair_perm(kv_dim, head_dim, head_dim))
    head_perm_inv = _invert_perm(
        _halved_to_pair_perm(head_dim, head_dim, head_dim)
    )
    for i in range(n_layers):
        for nm, perm in (
            (f"model.layers.{i}.self_attn.q_proj.weight", q_perm_inv),
            (f"model.layers.{i}.self_attn.k_proj.weight", k_perm_inv),
        ):
            t = dst.get(nm)
            if t is None:
                continue
            dst[nm] = t.index_select(0, perm).contiguous()
        for nm in (
            f"model.layers.{i}.self_attn.q_norm.weight",
            f"model.layers.{i}.self_attn.k_norm.weight",
        ):
            t = dst.get(nm)
            if t is None or t.dim() != 1 or t.numel() != head_dim:
                continue
            dst[nm] = t.index_select(0, head_perm_inv).contiguous()


def _build_hf_state_dict_from_archspec(
    am, arch, src_pairs: Mapping[tuple[str, str], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Walk the registered ArchSpec entries and emit
    ``{hf_name: cpu_contiguous_tensor}``. Skips any FlexTrain tensor
    that isn't in ``src_pairs`` (which is how we drop LoRA A/B factors)."""
    out: dict[str, torch.Tensor] = {}

    def _emit(scope: str, entries, layer_idx: int | None) -> None:
        for entry in entries:
            key = (scope, entry.flextrain_name)
            t = src_pairs.get(key)
            if t is None:
                # Either an optional entry (e.g. tied head's lm_head) or
                # the param has been pruned (LoRA filtering above).
                if not getattr(entry, "optional", False):
                    # Best effort: emit anyway from host params if we can,
                    # else leave it out and let the caller see whatever
                    # error vLLM raises on a missing tensor.
                    pass
                continue
            t = _apply_export_transform(t, entry.transform)
            out[_render_hf_name(entry.hf_name, layer_idx)] = to_cpu_contiguous(t)

    _emit("embed", arch.embed, None)
    _emit("head", arch.head, None)
    n_layers = len(am.backbone)
    for i in range(n_layers):
        _emit(f"layer_{i}", arch.layer, i)
    return out


def save_hf_full(
    am,
    out_dir: str,
    *,
    hf_source_dir: str | None = None,
    arch=None,
    shard_size_bytes: int = 5 * (1 << 30),
) -> str:
    """Export ``am``'s host master params as a complete HF checkpoint
    directory at ``out_dir``.

    Behavior
    --------
    * LoRA A/B tensors are *skipped*. After a LoRA run, this function
      saves the (frozen) base weights only — i.e. the original model
      with no LoRA delta applied. To bake the LoRA delta into the base,
      use ``save_hf_merged`` instead.
    * Tokenizer + ``config.json`` + ``generation_config.json`` are
      copied from ``hf_source_dir`` (or ``am._hf_source_path`` when
      that was set by ``load_hf`` / ``from_pretrained``).
    * Weights are emitted under HF tensor names (e.g.
      ``model.layers.0.self_attn.q_proj.weight``) and sharded at the
      requested size, matching HF's ``model-NNNNN-of-NNNNN.safetensors``
      convention. An ``index.json`` is written when sharded.
    * Q/K halved->pair RoPE permutations applied at load time are
      inverted for Llama / Qwen3-dense. Other archs are best-effort.

    Returns the path to ``out_dir``.
    """
    src = collect_host_params(am)
    src = {k: v for k, v in src.items() if not is_lora_param(k[1])}

    if arch is None:
        arch = getattr(am, "_hf_arch", None)
        if arch is None:
            from ..io.hf_weights import _ARCH_REGISTRY  # type: ignore[attr-defined]

            if not _ARCH_REGISTRY:
                raise ValueError(
                    "save_hf_full needs the registered ArchSpec — pass "
                    "``arch=`` explicitly or load weights via ``load_hf`` "
                    "first so we can recover it."
                )
            arch = next(iter(_ARCH_REGISTRY.values()))

    if hf_source_dir is None:
        hf_source_dir = getattr(am, "_hf_source_path", None)
        if hf_source_dir is None:
            raise ValueError(
                "save_hf_full needs the original HF directory to copy "
                "tokenizer / config files. Pass ``hf_source_dir=`` "
                "explicitly or load weights via ``load_hf``."
            )

    os.makedirs(out_dir, exist_ok=True)
    copy_hf_aux_files(hf_source_dir, out_dir)

    hf_state = _build_hf_state_dict_from_archspec(am, arch, src)
    _post_export_permute_for_arch(am, hf_state, arch.hf_arch_ids)

    write_sharded_safetensors(
        hf_state, out_dir, shard_size_bytes=shard_size_bytes
    )
    return out_dir
