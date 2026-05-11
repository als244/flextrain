"""HuggingFace safetensors loader / exporter.

This is the *infrastructure* (safetensors iteration, tensor copy, name
lookup, dtype/transpose handling). The *architecture-specific* data (HF
tensor name -> FlexTrain field name) lives in ``flextrain/io/arch/<family>.py``.

Design rule
-----------
Per the plan, we do NOT use ``AutoModelForCausalLM`` to load weights. We
open safetensors shards directly via the ``safetensors`` package and copy
slice-by-slice into pre-allocated FlexTrain host buffers. This avoids
materializing an HF nn.Module tree we don't need.

Load path
---------
    from flextrain.io.hf_weights import load_hf_safetensors
    from flextrain.io.arch.llama import LLAMA_ARCH

    load_hf_safetensors(
        hf_path="meta-llama/Llama-3-8B",
        arch=LLAMA_ARCH,
        dest=dest_mapping,       # {(scope, name): host Tensor} the engine allocated
        num_layers=32,
    )

Export path
-----------
    export_hf_safetensors(
        out_dir="runs/llama3-finetuned",
        arch=LLAMA_ARCH,
        src=src_mapping,
        num_layers=32,
    )

Tensor name convention
----------------------
``dest`` maps ``(scope, name)`` where:
  * ``scope="embed"``         -> one of the input-layer fields.
  * ``scope="head"``          -> one of the output-layer fields.
  * ``scope="layer_{i}"``     -> backbone layer ``i``'s fields.

Per-tensor transforms
---------------------
Some HF tensors need post-processing on the way into FlexTrain (and the
inverse on the way out). We model these as a small ``Transform`` enum:

  * ``NONE``       : copy as-is (shape + dtype match).
  * ``TRANSPOSE``  : HF stores ``W`` as ``(out, in)``, FlexTrain uses
                     ``(in, out)`` for fused ``X @ W`` matmuls. Applied on
                     both load and export.
  * ``QKV_SPLIT``  : some checkpoints store fused QKV; we split them back.
                     (Not used by Llama; reserved for later architectures.)
"""

from __future__ import annotations

import enum
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as _save_safetensors
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "flextrain.io.hf_weights requires `safetensors`. "
        "Install via `pip install safetensors`."
    ) from e


# ---------------------------------------------------------------------------
# Per-tensor transform enum + applicator.
# ---------------------------------------------------------------------------


class Transform(str, enum.Enum):
    NONE = "NONE"
    TRANSPOSE = "TRANSPOSE"


def _apply_load_transform(t: torch.Tensor, transform: Transform) -> torch.Tensor:
    if transform is Transform.NONE:
        return t
    if transform is Transform.TRANSPOSE:
        # HF W is (out, in); FlexTrain wants (in, out). Transpose returns a
        # view; .contiguous() materializes for the subsequent copy_().
        return t.transpose(0, 1).contiguous()
    raise ValueError(f"unknown transform {transform!r}")


def _apply_export_transform(t: torch.Tensor, transform: Transform) -> torch.Tensor:
    # Export inverts load.
    return _apply_load_transform(t, transform)  # transpose is self-inverse for 2D


# ---------------------------------------------------------------------------
# Weight-map schema. One ArchSpec per architecture family.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightMapEntry:
    """One (flextrain_name -> hf_name_template, transform).

    ``hf_name`` may contain the placeholder ``{i}`` for a backbone-layer
    index; :func:`_render_hf_name` substitutes.

    ``hf_name_alternates`` is a tuple of alternate templates the loader
    will try if ``hf_name`` is absent from the checkpoint. First name
    present in the shards wins. Used by archs whose HF safetensor
    layout depends on the wrapping class — e.g. Gemma 3:
    ``Gemma3ForCausalLM`` (1B) saves weights under ``model.layers.*``
    while ``Gemma3ForConditionalGeneration`` (4B/12B) saves them under
    ``language_model.model.layers.*``. One ArchSpec, two prefixes.

    ``optional=True`` marks an entry that may be absent on disk for
    SOME layers (heterogeneous backbones — e.g. Qwen3-Next where
    ``self_attn.*`` exists only on full-attn layers and ``linear_attn.*``
    exists only on linear-attn layers). Strict mode skips the
    "must-be-consumed" check for such entries.
    """

    flextrain_name: str
    hf_name: str
    transform: Transform = Transform.NONE
    optional: bool = False
    hf_name_alternates: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArchSpec:
    """Per-architecture weight map.

    Attributes
    ----------
    hf_arch_ids
        ``config.architectures`` values that this spec handles (e.g.
        ``("LlamaForCausalLM",)``). Used by :func:`select_arch`.
    embed
        Weight-map entries under scope ``"embed"``. Typically one entry for
        the token-embedding table.
    head
        Weight-map entries under scope ``"head"``. Typically final-norm +
        LM head.
    layer
        Weight-map entries under scope ``"layer_{i}"``. Names contain
        ``{i}`` for the layer index.
    post_load_hook
        Optional callable invoked after per-tensor copies complete. Used
        by architectures whose HF weights don't map 1:1 to a single
        FlexTrain tensor — e.g. MoE models where each expert has
        separate ``experts.{e}.gate_proj``/``up_proj``/``down_proj``
        tensors that we need to STACK into ``w_up (E, 2*F, d)`` and
        ``w_down (E, d, F)``. The hook receives the HF path and the
        full ``dest`` mapping ``{(scope, name): host Tensor}`` and may
        read additional raw HF tensors via ``safe_open`` to populate
        the stacked FlexTrain tensors. Signature:
        ``(hf_path: str, dest: Mapping, num_layers: int) -> None``.
    pre_export_hook
        Symmetric inverse of ``post_load_hook``. Optional callable
        invoked by ``save_hf_full`` after the ArchSpec walk emits the
        per-tensor entries and BEFORE the final permute / write. Used
        to:
          1. Unstack FT 3-D MoE weights (``w_up (E, 2F, d)`` / ``w_down``)
             back to per-expert HF tensors that the loader expects.
          2. Unbundle FT linear-attn projections (``w_lin_qkvz`` /
             ``w_lin_ba``) back to HF's split per-K-head form
             (``in_proj_qkv`` / ``in_proj_z`` / ``in_proj_b`` /
             ``in_proj_a``).
          3. Subtract 1.0 from non-gated RMSNorm γ for arches whose HF
             RMSNorm forwards ``(1 + weight) * x_normed`` (Qwen3.5 /
             Qwen3.5-MoE / Qwen3-Next / Gemma2 / Gemma3) since FT
             stores the canonical γ = 1 + w_HF on master.
        The hook receives the active model, the MUTABLE HF state dict
        (``{hf_name: cpu Tensor}``), and the layer count. It mutates
        the dict in place: deletes any FT-stacked / FT-bundled tensors
        listed in ArchSpec.layer that the dump should NOT contain, and
        writes the per-expert / split / shifted HF tensors directly.
        Signature: ``(am, dst, num_layers) -> None``.
    """

    hf_arch_ids: tuple[str, ...]
    embed: tuple[WeightMapEntry, ...]
    head: tuple[WeightMapEntry, ...]
    layer: tuple[WeightMapEntry, ...]
    post_load_hook: Callable[[str, Mapping, int], None] | None = None
    pre_export_hook: Callable[[Any, MutableMapping, int], None] | None = None


# Registry: hf_arch_id -> ArchSpec. ``register_arch`` populates it; arch
# modules call that on import.
_ARCH_REGISTRY: dict[str, ArchSpec] = {}


def register_arch(spec: ArchSpec) -> None:
    for arch_id in spec.hf_arch_ids:
        if arch_id in _ARCH_REGISTRY:
            raise ValueError(f"ArchSpec for {arch_id!r} already registered")
        _ARCH_REGISTRY[arch_id] = spec


def select_arch(hf_config: Mapping) -> ArchSpec:
    """Pick the registered :class:`ArchSpec` matching this HF config.

    Accepts a ``transformers.PretrainedConfig`` or a plain dict with an
    ``architectures`` key (as in ``config.json``).
    """
    archs = getattr(hf_config, "architectures", None) or hf_config.get(
        "architectures"
    )
    if not archs:
        raise ValueError(
            "HF config has no `architectures` field -- can't pick weight map"
        )
    for a in archs:
        if a in _ARCH_REGISTRY:
            return _ARCH_REGISTRY[a]
    raise ValueError(
        f"no registered ArchSpec for any of {archs!r}. "
        f"Known: {sorted(_ARCH_REGISTRY)}"
    )


# ---------------------------------------------------------------------------
# Shard iteration.
# ---------------------------------------------------------------------------


def _find_shards(hf_path: str) -> list[str]:
    """Return sorted list of safetensors files in ``hf_path``. Accepts
    either a local directory or a single-file path."""
    if os.path.isfile(hf_path):
        return [hf_path]
    shards = sorted(glob.glob(os.path.join(hf_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"no .safetensors files found under {hf_path!r}"
        )
    return shards


def _load_weight_index(hf_path: str) -> Mapping[str, str] | None:
    """Read ``model.safetensors.index.json`` if present, mapping each
    tensor name to its shard filename. None if single-shard."""
    index_path = os.path.join(hf_path, "model.safetensors.index.json")
    if not os.path.isfile(index_path):
        return None
    with open(index_path) as f:
        return json.load(f).get("weight_map")


def _render_hf_name(template: str, layer_idx: int | None) -> str:
    if "{i}" in template:
        if layer_idx is None:
            raise ValueError(
                f"template {template!r} has {{i}} but no layer_idx supplied"
            )
        return template.format(i=layer_idx)
    return template


# ---------------------------------------------------------------------------
# Public: load / export.
# ---------------------------------------------------------------------------


def load_hf_safetensors(
    hf_path: str,
    arch: ArchSpec,
    dest: MutableMapping[tuple[str, str], torch.Tensor],
    num_layers: int,
    *,
    strict: bool = True,
    device: str = "cpu",
) -> list[str]:
    """Populate ``dest[(scope, name)]`` from safetensors at ``hf_path``.

    Parameters
    ----------
    hf_path
        Local directory of safetensors shards (typical HF layout).
    arch
        Registered :class:`ArchSpec` (call :func:`select_arch` to pick).
    dest
        Engine-allocated mapping from ``(scope, flextrain_name)`` to target
        tensor. Must be pre-allocated with correct shapes and dtypes; we
        copy-in-place with ``copy_()``.
    num_layers
        Backbone layer count. Used to instantiate the per-layer template.
    strict
        Raise on any missing HF tensor. Set False when loading partial
        checkpoints (e.g. LoRA adapters).
    device
        Passed to ``safe_open(..., device=...)``. Keep ``"cpu"`` for
        host-pinned allocation.

    Returns
    -------
    list[str]
        HF tensor names present in the shards that we DID NOT consume
        (diagnostic; should be empty for a full model).
    """
    shards = _find_shards(hf_path)
    weight_index = _load_weight_index(hf_path)

    # Build the full (hf_name -> (scope, flextrain_name, transform)) list.
    # ``optional_pairs`` collects HF names from entries marked
    # ``optional=True``; they are excluded from the strict
    # "must-have-been-consumed" check (heterogeneous backbones).
    #
    # For entries with ``hf_name_alternates``, we register one pair per
    # candidate name, all pointing at the same dest slot. The loader's
    # idempotent ``copy_()`` populates from whichever name happens to
    # be in the shards; only one is ever expected to be present.
    # ``alternate_groups`` tracks "at least one of these names must
    # have been consumed" so strict mode rejects truly-missing entries
    # without flagging the unused alternates as missing.
    pairs: list[tuple[str, tuple[str, str], Transform]] = []
    optional_hf_names: set[str] = set()
    alternate_groups: list[tuple[str, list[str]]] = []  # (label, names_for_strict_check)

    def _add_entry(entry: WeightMapEntry, scope: str, layer_idx: int | None) -> None:
        primary = _render_hf_name(entry.hf_name, layer_idx)
        candidates = [primary] + [
            _render_hf_name(alt, layer_idx) for alt in entry.hf_name_alternates
        ]
        for hf in candidates:
            pairs.append((hf, (scope, entry.flextrain_name), entry.transform))
            if entry.optional:
                optional_hf_names.add(hf)
        if not entry.optional:
            alternate_groups.append(
                (f"{scope}/{entry.flextrain_name}", candidates)
            )

    for entry in arch.embed:
        _add_entry(entry, "embed", None)
    for entry in arch.head:
        _add_entry(entry, "head", None)
    for i in range(num_layers):
        scope = f"layer_{i}"
        for entry in arch.layer:
            _add_entry(entry, scope, i)

    # Route each expected HF tensor to its shard file.
    by_shard: dict[str, list[tuple[str, tuple[str, str], Transform]]] = {}
    for hf_name, key, xform in pairs:
        shard = (
            os.path.join(hf_path, weight_index[hf_name])
            if weight_index and hf_name in weight_index
            else None
        )
        by_shard.setdefault(shard or "ALL", []).append((hf_name, key, xform))

    consumed: set[str] = set()
    seen_in_shards: set[str] = set()

    def _consume_shard(shard_path: str, wanted_in_this_shard) -> None:
        with safe_open(shard_path, framework="pt", device=device) as f:
            shard_keys = set(f.keys())
            seen_in_shards.update(shard_keys)
            for hf_name, (scope, fx_name), xform in wanted_in_this_shard:
                if hf_name not in shard_keys:
                    continue
                tensor = f.get_tensor(hf_name)
                tensor = _apply_load_transform(tensor, xform)
                target = dest.get((scope, fx_name))
                if target is None:
                    if strict:
                        raise KeyError(
                            f"dest missing slot ({scope!r}, {fx_name!r})"
                        )
                    continue
                if target.shape != tensor.shape:
                    raise ValueError(
                        f"{hf_name!r} -> ({scope}, {fx_name}): shape mismatch "
                        f"HF {tuple(tensor.shape)} vs dest {tuple(target.shape)}"
                    )
                target.copy_(tensor.to(target.dtype))
                consumed.add(hf_name)

    if weight_index is None:
        # Single shard or unindexed dir; scan each shard for any wanted name.
        for shard_path in shards:
            _consume_shard(shard_path, pairs)
    else:
        for shard_path, wanted in by_shard.items():
            if shard_path == "ALL":
                for sp in shards:
                    _consume_shard(sp, wanted)
            else:
                _consume_shard(shard_path, wanted)

    if strict:
        # A required entry is satisfied iff AT LEAST ONE of its
        # candidate names (primary + alternates) was consumed. Entries
        # marked optional via ``optional_hf_names`` are not in the
        # alternate-groups list.
        missing_groups = [
            (label, candidates)
            for label, candidates in alternate_groups
            if not any(c in consumed for c in candidates)
        ]
        if missing_groups:
            sample = [
                f"{lbl} (tried {cands[:2]}{'...' if len(cands)>2 else ''})"
                for lbl, cands in missing_groups[:5]
            ]
            raise KeyError(
                f"{len(missing_groups)} expected HF entries not satisfied. "
                f"First missing: {sample}"
            )

    # Arch-specific post-load hook (MoE expert stacking, etc.). Runs
    # AFTER the per-tensor TRANSPOSE/etc. transforms have populated
    # the regular ArchSpec-declared tensors. The hook may open the
    # safetensors again for tensors NOT listed in ArchSpec (e.g. the
    # 64 per-expert gate_proj / up_proj / down_proj in OLMoE).
    if arch.post_load_hook is not None:
        arch.post_load_hook(hf_path, dest, num_layers)

    leftover = sorted(seen_in_shards - consumed)
    return leftover


def export_hf_safetensors(
    out_dir: str,
    arch: ArchSpec,
    src: Mapping[tuple[str, str], torch.Tensor],
    num_layers: int,
    *,
    out_filename: str = "model.safetensors",
) -> str:
    """Write a single-shard safetensors file at ``out_dir/out_filename``
    holding the FlexTrain weights under HF names.

    Returns the path written. Multi-shard export is a future extension.
    """
    os.makedirs(out_dir, exist_ok=True)

    out: dict[str, torch.Tensor] = {}

    def _emit(scope: str, entries, layer_idx: int | None) -> None:
        for entry in entries:
            hf_name = _render_hf_name(entry.hf_name, layer_idx)
            tensor = src[(scope, entry.flextrain_name)]
            tensor = _apply_export_transform(tensor.detach(), entry.transform)
            out[hf_name] = tensor.contiguous().cpu()

    _emit("embed", arch.embed, None)
    _emit("head", arch.head, None)
    for i in range(num_layers):
        _emit(f"layer_{i}", arch.layer, i)

    path = os.path.join(out_dir, out_filename)
    _save_safetensors(out, path)
    return path
