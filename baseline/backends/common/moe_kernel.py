"""Configure HuggingFace's native MoE-kernel dispatch (no custom adapter).

HF transformers exposes
`transformers.integrations.moe.ExpertsInterface` with
``{"batched_mm", "grouped_mm", "sonicmoe", ...}`` registered, and a
one-line model method ``model.set_experts_implementation("sonicmoe")``
(introduced in
https://github.com/huggingface/transformers/pull/45433) that re-binds
the experts module's forward to the named kernel.

This module just wraps that call so each HF backend (trl_deepspeed,
deepspeed_arctic, trl_fsdp, megatrain) can opt in via ``--moe-kernel-backend sonic``
without keeping a custom adapter on our side. The previous local
``SonicMoEAdapter`` (router + experts replacement, legacy ModuleList
layout, no DTensor support) has been removed.
"""

from __future__ import annotations

from typing import Literal

MoEKernelMode = Literal["hf", "auto", "sonic"]

_KERNEL_NAME = "sonicmoe"


def apply_moe_kernel_backend(model, mode: MoEKernelMode) -> str:
    """Switch ``model`` to the named experts kernel.

    Returns the kernel actually in use (``"hf"`` or ``"sonicmoe"``).

    ``mode`` semantics:
      * ``"hf"`` — keep the model's default experts implementation.
      * ``"auto"`` — try sonicmoe; on any failure (transformers too old,
        not a MoE model, kernel can't load on this GPU/CUDA combo) fall
        back to the default and log the reason.
      * ``"sonic"`` — strict: raise on any failure.

    Detection: if the model exposes ``set_experts_implementation``
    (HF transformers PR #45433+ with the modular Experts API), we call
    it. Models without that method (older transformers, dense models,
    legacy MoE layout) keep their default.
    """
    if mode == "hf":
        print("moe_kernel_backend=hf (default; --moe-kernel-backend not set)", flush=True)
        return "hf"

    setter = getattr(model, "set_experts_implementation", None)
    if setter is None:
        msg = (
            "model does not expose set_experts_implementation; "
            "either it isn't a modular MoE model, or the installed "
            "transformers predates the ExpertsInterface refactor "
            "(HF PR #45433)."
        )
        if mode == "sonic":
            raise SystemExit(f"--moe-kernel-backend sonic requested but {msg}")
        print(f"moe_kernel_backend=hf reason={msg}", flush=True)
        return "hf"

    try:
        setter(_KERNEL_NAME)
    except Exception as exc:  # noqa: BLE001 — surface kernel-load failures cleanly
        if mode == "sonic":
            raise SystemExit(
                f"--moe-kernel-backend sonic requested but "
                f"set_experts_implementation('{_KERNEL_NAME}') raised: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        print(
            f"moe_kernel_backend=hf reason=set_experts_implementation_failed "
            f"({type(exc).__name__}: {exc})",
            flush=True,
        )
        return "hf"

    print(f"moe_kernel_backend=sonicmoe (HF native, set via model.set_experts_implementation)", flush=True)
    return "sonicmoe"
