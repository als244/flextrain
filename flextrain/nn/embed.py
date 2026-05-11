"""TokenEmbedLayer: the ``InputLayer`` for Transformer architectures.

Ports ``orig/awsm_transformer/embed.py`` onto the :class:`InputLayer`
Protocol from :mod:`flextrain.core.layer`.

Semantics mirror orig exactly
-----------------------------
* ``forward(token_ids, weights)`` -- fancy-indexed copy from
  ``w_tok_embeddings``. Returns a fresh ``(num_tokens, d_model)``
  tensor on the device of the weights (the engine places it into the
  per-chunk "transition" slot).
* ``backward(dx, token_ids, grad_weights)`` -- calls
  ``flextrain_embedding_bwd`` which does a scatter-add into the embedding
  gradient rows. ``scale=1.0`` matches orig (the sqrt(d_model) factor
  is commented out on both sides).

The embedding has no per-(chunk, layer) activation slot: its "activation"
is the token_ids tensor, which ``ChunkMeta`` already owns. We therefore
declare an empty :class:`ActivationSchema` with ``max_tier=0`` -- the
engine will see zero bytes to offload for this layer. This is consistent
with orig's special-casing (no entry in ``cpu_act_slots`` for the embed
layer; see ``orig/active_model.py:1230-1235``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import ActivationSchema
from flextrain.core.layer import (
    ChunkMeta,
    ComputeCost,
    LayerContext,
    ParamSpec,
    TensorSpec,
)
from flextrain.ops import flextrain_embedding_bwd


@dataclass(frozen=True)
class TokenEmbedConfig:
    """Minimal embedding-layer config.

    Llama / Qwen / Mistral / OLMoE all share the same embedding surface
    (a ``(vocab_size, d_model)`` table, no bias, no positional additions
    since RoPE is per-block). Future architectures that need additional
    behavior (Gemma's sqrt(d_model) scaling, tied-embedding indicator,
    ...) extend this class rather than branching inside the layer.
    """

    vocab_size: int
    d_model: int
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None  # defaults to compute_dtype
    grad_dtype: torch.dtype | None = None  # defaults to compute_dtype
    # Gemma 2 / Gemma 3 scale the embedding output by ``sqrt(d_model)``
    # before feeding it into the first decoder layer. Other arches
    # (Llama, Qwen, Mistral) use 1.0 (no scaling). ``None`` ≡ 1.0.
    embed_scale: float | None = None


class TokenEmbedLayer:
    """Token-embedding ``InputLayer``.

    Parameters
    ----------
    cfg
        :class:`TokenEmbedConfig` with vocab size, d_model, dtypes.

    Attributes
    ----------
    schema
        Empty :class:`ActivationSchema` (max_tier=0, no fields). The
        engine queries this to allocate 0 bytes of home activation
        storage for the embed layer.
    param_spec
        :class:`ParamSpec` with a single tensor ``w_tok_embeddings`` of
        shape ``(vocab_size, d_model)``. The engine allocates host
        master + grad buffers + optimizer state from this spec alone.
    """

    # Conceptually we don't have a layer_id (embed runs once, outside the
    # for-layer loop), but keeping the attribute makes the Protocol
    # unified with :class:`Layer`.
    layer_id: int = -1

    def __init__(self, cfg: TokenEmbedConfig) -> None:
        self.cfg = cfg
        self.schema = ActivationSchema(fields=(), max_tier=0)

        def _table_shape(dims: Mapping[str, int]) -> tuple[int, ...]:
            # Preferentially read from layer-level dims if they match; fall
            # back to cfg values. Keeps the spec usable for both model-wide
            # dims dicts and minimal unit-test dicts.
            v = dims.get("vocab_size", cfg.vocab_size)
            d = dims.get("d_model", cfg.d_model)
            return (v, d)

        self.param_spec = ParamSpec(
            tensors=(
                TensorSpec(
                    name="w_tok_embeddings",
                    shape_fn=_table_shape,
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
            )
        )

    # ------------------------------------------------------------------
    # InputLayer Protocol
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        chunk: ChunkMeta,  # noqa: ARG002  -- not consumed; kept for Protocol
        weights: Mapping[str, torch.Tensor],
        ctx: LayerContext,  # noqa: ARG002  -- not consumed; kept for Protocol
    ) -> torch.Tensor:
        """Return ``weights["w_tok_embeddings"][token_ids, :]`` (a copy),
        optionally scaled by ``cfg.embed_scale`` (Gemma's
        ``sqrt(d_model)`` knob).

        Mirrors ``orig/awsm_transformer/embed.py:12-18``.
        """
        table = weights["w_tok_embeddings"]
        out = table[token_ids, :]
        if self.cfg.embed_scale is not None and self.cfg.embed_scale != 1.0:
            out = out.mul_(self.cfg.embed_scale)
        return out

    def backward(
        self,
        dx: torch.Tensor,
        token_ids: torch.Tensor,
        chunk: ChunkMeta,  # noqa: ARG002  -- not consumed; kept for Protocol
        weights: Mapping[str, torch.Tensor],  # noqa: ARG002  -- not consumed
        grads: MutableMapping[str, torch.Tensor],
        ctx: LayerContext,  # noqa: ARG002  -- not consumed; kept for Protocol
    ) -> None:
        """Scatter-add ``dx * embed_scale`` into ``grads["g_tok_embeddings"]``.

        Skipped under LoRA (the embed table is frozen, ``grads`` does
        not contain ``g_tok_embeddings``).

        Mirrors ``orig/awsm_transformer/embed.py:20-32``.
        """
        g = grads.get("g_tok_embeddings")
        if g is None:
            return
        scale = (
            float(self.cfg.embed_scale)
            if self.cfg.embed_scale is not None else 1.0
        )
        flextrain_embedding_bwd(dx, token_ids, g, scale=scale)

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        """Gather (forward) + scatter-add (backward) are both bandwidth-
        bound, not FLOP-bound. We report zero FLOPs so the DP solver sees
        the embedding's compute time as ~0 (which is correct: it never
        dominates a layer's compute time).
        """
        _ = chunk
        return ComputeCost(total_fwd_flops=0, avoided_recompute_flops=(0,))
