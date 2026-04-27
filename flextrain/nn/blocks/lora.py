"""LoRA (Low-Rank Adaptation) block.

A frozen base linear `W` plus trainable low-rank delta `B @ A` with
scaling. Forward: ``y = x @ W + (x @ A) @ B * (alpha / r)``.

Memory characteristics:
* Base ``W (d_in, d_out)`` is **frozen** — no grads, no optimizer state.
  See :class:`flextrain.core.layer.TensorSpec.frozen`.
* ``A (d_in, r)`` and ``B (r, d_out)`` are trained as normal AdamW
  parameters (rank ``r`` is typically 8-32, so these are tiny vs ``W``).
* Engine still pays for ``W``'s compute slot on GPU and master copy on
  host (W is needed at every fwd) — but no grad / opt-state buffers.

This block is the building primitive; layers that want LoRA on their
attention or FFN projections compose with it. See
:class:`flextrain.nn.layers.lora_llama.LoRALlamaBlock` for an example
of a Llama block with LoRA on Q and V projections (the conventional
PEFT default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import ActivationField
from flextrain.core.layer import (
    ChunkMeta, ComputeCost, LayerContext, ParamSpec, TensorSpec,
)


@dataclass(frozen=True)
class LoRALinearConfig:
    """Configuration for a single LoRA-wrapped linear layer.

    Notation
    --------
    base shape: ``(d_in, d_out)``    (frozen)
    A shape:    ``(d_in, r)``        (trainable)
    B shape:    ``(r, d_out)``       (trainable)
    output:     ``y = x @ W + (x @ A) @ B * scale``   with scale = alpha / r

    ``alpha`` is the standard LoRA scaling factor (default = r, so
    scale = 1.0). ``init_std_a`` follows the PEFT convention of
    Kaiming-uniform on A and zero on B (so the LoRA delta starts
    at zero on first forward).
    """

    name: str                # base param name (e.g. "w_q")
    d_in_dim_name: str = "d_model"
    d_out_dim_name: str = "attn_dim"
    rank: int = 16
    alpha: float = 16.0
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def a_name(self) -> str:
        return f"{self.name}_lora_a"

    @property
    def b_name(self) -> str:
        return f"{self.name}_lora_b"


def lora_param_spec(cfg: LoRALinearConfig) -> tuple[TensorSpec, ...]:
    """Return the (frozen base + A + B) TensorSpec tuple for this LoRA
    config. Compose with the layer's other params via ``ParamSpec.merge``.
    """
    return (
        TensorSpec(
            name=cfg.name,
            shape_fn=lambda d, c=cfg: (d[c.d_in_dim_name], d[c.d_out_dim_name]),
            compute_dtype=cfg.compute_dtype,
            master_dtype=cfg.master_dtype,
            grad_dtype=cfg.grad_dtype,
            frozen=True,                    # base is frozen
        ),
        TensorSpec(
            name=cfg.a_name,
            shape_fn=lambda d, c=cfg: (d[c.d_in_dim_name], c.rank),
            compute_dtype=cfg.compute_dtype,
            master_dtype=cfg.master_dtype,
            grad_dtype=cfg.grad_dtype,
            optimizer="adamw",              # rank is small; adam is fine
        ),
        TensorSpec(
            name=cfg.b_name,
            shape_fn=lambda d, c=cfg: (c.rank, d[c.d_out_dim_name]),
            compute_dtype=cfg.compute_dtype,
            master_dtype=cfg.master_dtype,
            grad_dtype=cfg.grad_dtype,
            optimizer="adamw",
        ),
    )


def lora_init(
    weights: dict[str, torch.Tensor],
    cfg: LoRALinearConfig,
    *,
    init_std_a: float = 0.02,
    seed: int | None = None,
) -> None:
    """PEFT-style init: A ~ N(0, init_std_a), B = 0. So the LoRA
    delta starts as zero and the base behaves identically to a no-LoRA
    forward at step 0."""
    if seed is not None:
        torch.manual_seed(seed)
    weights[cfg.a_name].normal_(mean=0.0, std=init_std_a)
    weights[cfg.b_name].zero_()


def lora_linear_fwd(
    x: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cfg: LoRALinearConfig,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward: ``y = x @ W + (x @ A) @ B * scale``.

    This is a free function (not a class method) so it composes
    with existing block code that just does ``out = x @ W``. Replace
    the matmul with this call.
    """
    base = weights[cfg.name]
    A = weights[cfg.a_name]
    B = weights[cfg.b_name]
    if out is None:
        out = torch.matmul(x, base)
    else:
        torch.matmul(x, base, out=out)
    # LoRA delta: (x @ A) @ B * scale.
    xa = torch.matmul(x, A)
    out.addmm_(xa, B, alpha=cfg.scale)
    return out


def lora_linear_bwd(
    dy: torch.Tensor,                    # (T, d_out) upstream grad of y
    x: torch.Tensor,                     # (T, d_in) saved fwd input
    weights: Mapping[str, torch.Tensor],
    grads: MutableMapping[str, torch.Tensor],
    cfg: LoRALinearConfig,
) -> torch.Tensor:
    """Backward through the LoRA-wrapped linear.

    Returns dL/dx, accumulates dL/dA and dL/dB into ``grads`` (no grad
    for the frozen base).

    Math:
        y = x @ W + (x @ A) @ B * s         where s = alpha / r
        xa = x @ A                          (T, r)

        dL/dx_base   = dy @ W.T             # ignored — base is frozen
        dL/dx_lora   = (dy @ B.T) * s @ A.T (T, d_in)

        dL/dW        = none                 # base is frozen

        dL/dA        = (dy @ B.T).T @ x * s = x.T @ dy @ B.T * s
        dL/dB        = xa.T @ dy * s
    """
    A = weights[cfg.a_name]
    B = weights[cfg.b_name]

    # dy_b = dy @ B.T  (T, r)
    dy_b = torch.matmul(dy, B.transpose(-1, -2))
    # dL/dx contribution from LoRA path:
    dx_lora = torch.matmul(dy_b * cfg.scale, A.transpose(-1, -2))

    # xa for dL/dB:
    xa = torch.matmul(x, A)
    # dL/dB = xa.T @ dy * scale, accumulated. Cast to grad accumulator
    # dtype (often fp32) to keep precision over many steps.
    g_b = grads.get("g_" + cfg.b_name)
    if g_b is not None:
        contrib = (
            xa.transpose(-1, -2).to(g_b.dtype)
            @ dy.to(g_b.dtype)
            * cfg.scale
        )
        g_b.add_(contrib)
    # dL/dA = x.T @ (dy @ B.T) * scale, accumulated.
    g_a = grads.get("g_" + cfg.a_name)
    if g_a is not None:
        contrib = (
            x.transpose(-1, -2).to(g_a.dtype)
            @ dy_b.to(g_a.dtype)
            * cfg.scale
        )
        g_a.add_(contrib)

    return dx_lora


__all__ = [
    "LoRALinearConfig",
    "lora_param_spec",
    "lora_init",
    "lora_linear_fwd",
    "lora_linear_bwd",
]
