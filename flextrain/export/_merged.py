"""Merge LoRA into base weights, then export as a full HF checkpoint.

This is the universal LoRA-export path: it works for any architecture
the loader supports (Llama, Qwen3, Qwen3.5, Qwen3-MoE, Qwen3.5-MoE,
Qwen3-Next, OLMoE, ...) because the merge is performed in FlexTrain's
own tensor layout, before any HF mapping. After merging, the standard
``save_hf_full`` exporter produces a self-contained HF dir that
serving engines can load with no LoRA support required.

Trade-offs vs. ``save_lora_adapter``:

* (+) Universal: works for gated q_proj, MoE per-expert adapters, and
  anything else the loader handles.
* (+) No serving-time overhead: the engine treats the model as a plain
  base model.
* (-) Output is the size of the full base, not just the rank-r delta.
* (-) Cannot hot-swap multiple adapters at serve time (vLLM ``--lora-modules``
  / sGLang ``--lora-paths`` won't apply).

Math
----
For a 2-D target ``W (in, out)`` with ``A (in, r)`` and ``B (r, out)``::

    W' = W + scale * A @ B

For a 3-D MoE expert stack ``W (E, in, out)`` with ``A (E, in, r)`` and
``B (E, r, out)``, we merge per-expert::

    W'[e] = W[e] + scale * A[e] @ B[e]   for e in 0..E-1
"""
from __future__ import annotations

import os
from typing import Mapping

import torch

from ._common import is_lora_param, lora_target_from_param_name


def _scale_for(am, target_name: str) -> float:
    """Read the LoRA scale for a target from the wrapped layer.

    The LoRA wrapper holds the canonical alpha/rank → scale mapping;
    we go through ``am.backbone[i]`` to find it, falling back to the
    A/B tensor rank with a default alpha == rank (scale = 1.0) if a
    layer isn't a LoRA wrapper (e.g. mixed wrapped/unwrapped layers
    via the per-layer LoRA target filter)."""
    # We resolve scale per layer below; this helper is a placeholder
    # that records the contract.
    raise NotImplementedError


def _per_layer_scales(am, layer_idx: int) -> dict[str, float]:
    """Return ``{ft_target_name: scale}`` for one backbone layer."""
    layer = am.backbone[layer_idx]
    targets = getattr(layer, "targets", None)
    if not targets:
        return {}
    return {t.target_name: float(t.scale) for t in targets}


def merge_lora_into_base(am) -> int:
    """Apply ``W += scale * A @ B`` in place on every host param that
    has a matching LoRA pair, then zero the A/B tensors. Returns the
    number of targets merged.

    After this call, the LoRA-A/B host tensors are still allocated
    (we don't realloc the buffer manager) but contain zeros. The
    next ``save_hf_full`` will skip them on its own (the LoRA filter
    drops anything ending in ``_lora_a`` / ``_lora_b``); the zero
    init also makes the ActiveModel safe to keep training from this
    point if desired (delta starts at zero again).
    """
    n_layers = len(am.backbone)
    merged = 0
    for L in range(n_layers):
        host = am.buffers.host_params[L]
        scales = _per_layer_scales(am, L)
        for nm in list(host.keys()):
            if not nm.endswith("_lora_a"):
                continue
            target = lora_target_from_param_name(nm)
            b_name = f"{target}_lora_b"
            A = host.get(nm)
            B = host.get(b_name)
            W = host.get(target)
            if A is None or B is None or W is None:
                continue
            scale = scales.get(target, 1.0)
            if A.dim() == 2 and B.dim() == 2 and W.dim() == 2:
                # W: (in, out)  A: (in, r)  B: (r, out)
                delta = (A.float() @ B.float()) * scale
                W.add_(delta.to(W.dtype))
                A.zero_()
                B.zero_()
                merged += 1
            elif A.dim() == 3 and B.dim() == 3 and W.dim() == 3:
                # Per-expert MoE LoRA. LoRA factor convention (PEFT-aligned):
                #   A (E, in, r), B (E, r, out)
                # so the natural delta is bmm(A, B) -> (E, in, out).
                #
                # Option-B MoE storage transposed the routed-expert weights
                # so they live as W (E, out, in). Add delta to W after
                # transposing the bmm result to match W's orientation.
                # Done on CPU in fp32 to avoid loading huge MoE expert
                # stacks onto the GPU here.
                delta = torch.bmm(A.float(), B.float()) * scale  # (E, in, out)
                if W.shape[-2:] == delta.shape[-2:]:
                    W.add_(delta.to(W.dtype))
                elif W.shape[-2:] == delta.shape[-2:][::-1]:
                    # Option-B layout: W is transposed relative to delta.
                    W.add_(delta.transpose(-1, -2).contiguous().to(W.dtype))
                else:
                    raise ValueError(
                        f"layer {L} target {target!r}: cannot align "
                        f"LoRA delta {tuple(delta.shape)} with W "
                        f"{tuple(W.shape)} via either direct add or transpose."
                    )
                A.zero_()
                B.zero_()
                merged += 1
            else:
                raise ValueError(
                    f"unexpected LoRA tensor shapes on layer {L} "
                    f"target {target!r}: W={tuple(W.shape)}, "
                    f"A={tuple(A.shape)}, B={tuple(B.shape)}"
                )
    am._refresh_gpu_residents()
    return merged


def save_hf_merged(
    am,
    out_dir: str,
    *,
    hf_source_dir: str | None = None,
    arch=None,
    shard_size_bytes: int = 5 * (1 << 30),
    keep_lora_after_merge: bool = False,
) -> str:
    """Fold the LoRA delta into the base weights and write a full HF
    directory at ``out_dir`` (sharded ``model.safetensors`` + tokenizer
    + config + ``generation_config``). The output is an ordinary HF
    checkpoint — vLLM, sGLang, and ``transformers.from_pretrained``
    load it as a base model with no LoRA support required.

    By default this *modifies the host master params* in place: after
    the call, the in-memory model holds the merged weights and the
    LoRA A/B factors are zeroed. To preserve the in-memory LoRA state
    (so you can keep training the same adapter), pass
    ``keep_lora_after_merge=True`` — but note that this currently
    requires re-loading the base weights and is only useful in the
    "export then keep training" pattern. For the common "export at
    end of training" use case, leave the default.

    Returns the path to ``out_dir``.
    """
    if keep_lora_after_merge:
        # Snapshot the base + adapters, restore after merge+save.
        # We keep this a simple-but-correct copy: callers who care
        # about RAM during long export sessions should disable this.
        snapshot_base: list[dict[str, torch.Tensor]] = []
        snapshot_lora: list[dict[str, torch.Tensor]] = []
        for layer_host in am.buffers.host_params:
            base_snap, lora_snap = {}, {}
            for nm, t in layer_host.items():
                if is_lora_param(nm):
                    lora_snap[nm] = t.detach().clone()
                else:
                    base_snap[nm] = t.detach().clone()
            snapshot_base.append(base_snap)
            snapshot_lora.append(lora_snap)

    n_merged = merge_lora_into_base(am)
    if n_merged == 0:
        # Not necessarily an error: caller may have run a full FT and
        # just want the standard exporter. Forward-compatible with
        # ``save_hf_full(am, ...)``.
        pass

    from ._hf_full import save_hf_full

    out_path = save_hf_full(
        am,
        out_dir,
        hf_source_dir=hf_source_dir,
        arch=arch,
        shard_size_bytes=shard_size_bytes,
    )

    if keep_lora_after_merge:
        # Restore in-memory state so the user can keep training.
        for L, layer_host in enumerate(am.buffers.host_params):
            for nm, t in layer_host.items():
                src = snapshot_lora[L].get(nm)
                if src is None:
                    src = snapshot_base[L].get(nm)
                if src is not None:
                    t.copy_(src)
        am._refresh_gpu_residents()

    return out_path
