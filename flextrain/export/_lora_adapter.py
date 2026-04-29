"""LoRA adapter export in HF PEFT format.

Writes ``adapter_config.json`` + ``adapter_model.safetensors`` to a
directory that:

* ``transformers``/PEFT: ``PeftModel.from_pretrained(base, adapter_dir)``
* vLLM:                  ``--enable-lora --lora-modules my=adapter_dir``
* sGLang:                ``--lora-paths my=adapter_dir``

Layout details
--------------
PEFT adapter tensor names look like::

    base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight
    base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight

with shapes ``lora_A: (r, in_features)`` and ``lora_B: (out_features, r)``.
FlexTrain stores ``w_q_lora_a: (in, r)`` and ``w_q_lora_b: (r, out)`` —
so the export transposes both factors. For Q/K projections that had
the halved->pair RoPE permutation applied at load time, we also invert
that permutation along ``lora_B``'s ``out_features`` axis so the
adapter slot lines up with HF's ``q_proj``/``k_proj``.

Scope
-----
This format describes a per-row LoRA over standard linear projections.
That maps cleanly onto Llama and Qwen3-dense (non-gated q_proj, no
linear-attn). Architectures that ship with non-standard Q layouts —
Qwen3.5 / Qwen3.5-MoE / Qwen3-Next — can't be expressed as a vanilla
PEFT adapter; for those, use ``save_hf_merged`` to fold the LoRA delta
into the base weights before export.

Per-expert LoRA on MoE blocks (3-D adapter tensors) is also out of
scope: serving engines treat each expert as a separate linear and don't
have a place to attach per-expert adapters. ``save_hf_merged`` handles
this case correctly.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Mapping

import torch

from ._common import (
    collect_host_params,
    halved_to_pair_perm,
    invert_perm,
    is_lora_param,
    lora_target_from_param_name,
    to_cpu_contiguous,
    write_sharded_safetensors,
)


# Architectures we support for raw-adapter export. Others must use
# ``save_hf_merged``.
_PEFT_COMPATIBLE_ARCHS: frozenset[str] = frozenset({
    "LlamaForCausalLM",
    "Qwen3ForCausalLM",
})


# Per-arch map from FlexTrain target name → (HF module suffix, "is_q",
# "is_k"). The HF module suffix is what goes in ``target_modules`` and
# what's used to build the PEFT tensor name.
#
# We derive this from the registered ArchSpec at runtime (so adding a
# new attn target only needs the ArchSpec entry); see
# ``_target_to_hf_module``. The static map below documents the public
# contract: callers can ask which targets we'll emit.
_HF_MODULE_FOR_FT: dict[str, str] = {
    # Attn
    "w_q":     "q_proj",
    "w_k":     "k_proj",
    "w_v":     "v_proj",
    "w_o":     "o_proj",
    # SwiGLU MLP (Llama uses w_1=gate, w_3=up, w_2=down)
    "w_1":     "gate_proj",
    "w_2":     "down_proj",
    "w_3":     "up_proj",
    # Qwen3-dense MoE-flavored MLP (same names but FT uses these)
    "w_gate":  "gate_proj",
    "w_up":    "up_proj",
    "w_down":  "down_proj",
}


def _target_to_hf_module(arch, ft_target_name: str) -> str:
    """Return the HF module suffix (``"q_proj"``) from a FlexTrain
    target name (``"w_q"``). Reads the registered ArchSpec and strips
    the trailing ``.weight``."""
    for entry in arch.layer:
        if entry.flextrain_name == ft_target_name:
            hf_name = entry.hf_name  # e.g. "model.layers.{i}.self_attn.q_proj.weight"
            if not hf_name.endswith(".weight"):
                raise ValueError(
                    f"export: HF entry {hf_name!r} for {ft_target_name!r} "
                    f"doesn't end in '.weight'; can't infer module name."
                )
            stem = hf_name[: -len(".weight")]
            # Strip "model.layers.{i}." prefix to get "self_attn.q_proj"
            # → take the LAST dotted component as the module name (HF
            # PEFT matches on suffix).
            module = stem.split(".")[-1]
            return module
    raise KeyError(
        f"target {ft_target_name!r} is not a known LoRA-able projection "
        f"for arch {arch.hf_arch_ids!r}; only standard linear projections "
        f"are supported by the PEFT adapter format."
    )


def _peft_tensor_name(arch, ft_target_name: str, layer_idx: int, ab: str) -> str:
    """Compose the PEFT-style adapter tensor name for one (layer, target,
    A|B) combination. ``ab`` is ``"A"`` or ``"B"``.

    PEFT names look like
    ``base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight``.
    """
    # Find the per-layer HF name template (with {i}) and replace the
    # trailing "weight" with "lora_X.weight".
    for entry in arch.layer:
        if entry.flextrain_name == ft_target_name:
            tmpl = entry.hf_name
            if not tmpl.endswith(".weight"):
                raise ValueError(f"unexpected HF name {tmpl!r}")
            stem = tmpl[: -len(".weight")]
            rendered = stem.format(i=layer_idx) + f".lora_{ab}.weight"
            # PEFT prepends ``base_model.model.``.
            return f"base_model.model.{rendered}"
    raise KeyError(ft_target_name)


def _maybe_invert_qk_perm_on_b(
    am, ft_target_name: str, b_t: torch.Tensor,
) -> torch.Tensor:
    """If the base ``w_q`` / ``w_k`` had the halved->pair permutation
    applied at load time, invert it on the matching axis of LoRA-B
    so the exported adapter aligns with HF's ``q_proj`` / ``k_proj``.

    FlexTrain ``w_q_lora_b`` is ``(r, out)`` where ``out == attn_dim``.
    We invert along axis 1 (the per-head halved-pair axis). After
    transpose-to-PEFT (axes -> ``(out, r)``) we'd be inverting axis 0
    instead, but we do the invert *before* transposing, so the caller
    can transpose afterwards.
    """
    dims = am.dims
    head_dim = int(dims["head_dim"])
    if ft_target_name == "w_q":
        out_dim = int(dims["n_heads"]) * head_dim
    elif ft_target_name == "w_k":
        out_dim = int(dims["n_kv_heads"]) * head_dim
    else:
        return b_t
    perm_inv = invert_perm(halved_to_pair_perm(out_dim, head_dim, head_dim))
    return b_t.index_select(1, perm_inv).contiguous()


def _is_2d(t: torch.Tensor) -> bool:
    return t.dim() == 2


def _gather_lora_pairs(
    am,
) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
    """Return ``{(layer_idx, ft_target_name): (A, B)}`` for every LoRA
    pair on a 2-D (non-MoE) target. 3-D adapter stacks are skipped (and
    callers should error if any are present)."""
    pairs: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for i, layer_host in enumerate(am.buffers.host_params):
        # Walk LoRA-A entries; matching B must exist by construction.
        for nm, t in layer_host.items():
            if not nm.endswith("_lora_a"):
                continue
            target = lora_target_from_param_name(nm)
            b_name = f"{target}_lora_b"
            b = layer_host.get(b_name)
            if b is None:
                continue
            if not (_is_2d(t) and _is_2d(b)):
                # 3-D MoE expert stack — skip; caller checks below.
                continue
            pairs[(i, target)] = (t, b)
    return pairs


def _has_3d_lora(am) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for i, layer_host in enumerate(am.buffers.host_params):
        for nm, t in layer_host.items():
            if (nm.endswith("_lora_a") or nm.endswith("_lora_b")) and t.dim() == 3:
                out.append((i, nm))
    return out


def save_lora_adapter(
    am,
    out_dir: str,
    *,
    base_model_name_or_path: str | None = None,
    arch=None,
    rank: int | None = None,
    alpha: float | None = None,
    bias: str = "none",
) -> str:
    """Write a PEFT-format LoRA adapter directory at ``out_dir``.

    Parameters
    ----------
    am
        :class:`ActiveModel` after a LoRA fine-tune.
    out_dir
        Destination directory (created if missing).
    base_model_name_or_path
        Goes into ``adapter_config.json``. This is what serving engines
        and PEFT use to locate the base weights when the adapter is
        loaded standalone. Defaults to ``am._hf_source_path``.
    arch
        Optional explicit ArchSpec; defaults to the one resolved from
        the loaded HF config.
    rank, alpha
        Override the LoRA hyperparameters written into the config. If
        ``None``, we read the rank from the actual A/B tensor shapes
        (and require all targets to share rank/alpha).

    Returns the path to ``out_dir``.

    Raises
    ------
    ValueError
        If the model has no LoRA tensors, if the arch isn't supported
        by the PEFT adapter format (use ``save_hf_merged`` for those),
        or if 3-D MoE expert adapters are present (also need merge).
    """
    if arch is None:
        arch = getattr(am, "_hf_arch", None)
        if arch is None:
            raise ValueError(
                "save_lora_adapter needs the registered ArchSpec — pass "
                "``arch=`` explicitly or load weights via ``load_hf`` "
                "first so we can recover it."
            )
    arch_set = set(arch.hf_arch_ids)
    if not (arch_set & _PEFT_COMPATIBLE_ARCHS):
        raise ValueError(
            f"PEFT adapter export is currently supported for "
            f"{sorted(_PEFT_COMPATIBLE_ARCHS)}; got {sorted(arch_set)}. "
            f"For other architectures, fold the LoRA delta into the base "
            f"weights via ``flextrain.export.save_hf_merged`` and serve "
            f"the merged model directly."
        )

    if base_model_name_or_path is None:
        base_model_name_or_path = getattr(am, "_hf_source_path", None) or ""

    pairs = _gather_lora_pairs(am)
    if not pairs:
        raise ValueError(
            "save_lora_adapter found no LoRA tensors on this ActiveModel. "
            "Did you train with ``lora_targets=``?"
        )
    bad = _has_3d_lora(am)
    if bad:
        raise ValueError(
            f"save_lora_adapter found per-expert (3-D) LoRA tensors at "
            f"{bad[:5]}... — PEFT/vLLM/sGLang don't support per-expert "
            f"adapters. Use ``save_hf_merged`` to fold them into the "
            f"base experts and serve the merged model."
        )

    # Cross-check rank/alpha consistency. We don't currently stash
    # alpha in the host_params dict, so we trust the caller's value or
    # fall back to ``rank`` (LoRA scale defaults to 1 in PEFT when
    # alpha == rank). Users who used non-default alpha should pass it.
    inferred_rank: int | None = None
    for (_, ft), (a_t, b_t) in pairs.items():
        # FT shapes: A (in, r), B (r, out). PEFT wants A (r, in), B (out, r).
        if a_t.shape[1] != b_t.shape[0]:
            raise ValueError(
                f"LoRA A/B rank mismatch on layer target {ft!r}: "
                f"A shape={tuple(a_t.shape)}, B shape={tuple(b_t.shape)}"
            )
        r = int(a_t.shape[1])
        if inferred_rank is None:
            inferred_rank = r
        elif inferred_rank != r:
            raise ValueError(
                f"LoRA targets have heterogeneous ranks "
                f"({inferred_rank} vs {r}); PEFT/vLLM expect a single "
                f"rank per adapter. Export each rank-group separately."
            )
    rank = rank if rank is not None else inferred_rank
    alpha = alpha if alpha is not None else float(rank)  # scale=1 default

    # Compose adapter_model state-dict in PEFT names + shapes.
    state: dict[str, torch.Tensor] = {}
    target_modules: set[str] = set()
    for (layer_idx, ft_target), (a_t, b_t) in pairs.items():
        try:
            module = _target_to_hf_module(arch, ft_target)
        except KeyError:
            # A target the ArchSpec doesn't know about — skip
            # silently is misleading; raise so the user sees it.
            raise ValueError(
                f"layer {layer_idx} has a LoRA on FT target {ft_target!r} "
                f"but the ArchSpec has no matching HF entry. Add a "
                f"WeightMapEntry for it or use save_hf_merged."
            )
        target_modules.add(module)

        # Apply RoPE-permute inversion to B for q/k targets so the
        # exported B aligns with HF's q_proj.weight / k_proj.weight axis.
        if ft_target in ("w_q", "w_k"):
            b_t = _maybe_invert_qk_perm_on_b(am, ft_target, b_t)

        # FT (in, r) -> PEFT (r, in)
        peft_a = a_t.transpose(0, 1).contiguous()
        # FT (r, out) -> PEFT (out, r)
        peft_b = b_t.transpose(0, 1).contiguous()

        state[_peft_tensor_name(arch, ft_target, layer_idx, "A")] = (
            to_cpu_contiguous(peft_a)
        )
        state[_peft_tensor_name(arch, ft_target, layer_idx, "B")] = (
            to_cpu_contiguous(peft_b)
        )

    os.makedirs(out_dir, exist_ok=True)
    write_sharded_safetensors(
        state, out_dir, base_filename="adapter_model"
    )

    # PEFT config. We hard-code the LoRA fields that vLLM / sGLang /
    # PEFT need. Notable choices:
    #   * ``fan_in_fan_out=False`` — HF projections store W as (out, in),
    #     so PEFT's default fan-out is correct.
    #   * ``modules_to_save=null`` — we're not saving any base layers
    #     intact (e.g. embed/lm_head); that's what save_hf_merged is for.
    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path,
        "bias": bias,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "lora_alpha": float(alpha),
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": int(rank),
        "revision": None,
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
    }
    with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2, sort_keys=True)

    # Helpful sidecar: also copy tokenizer + chat template so a serving
    # engine can locate them via the adapter dir if they want.
    src_dir = getattr(am, "_hf_source_path", None) or base_model_name_or_path
    if src_dir and os.path.isdir(src_dir):
        from ._common import copy_hf_aux_files
        copy_hf_aux_files(src_dir, out_dir)

    return out_dir
