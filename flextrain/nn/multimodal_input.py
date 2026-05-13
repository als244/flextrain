"""MultimodalInputLayer -- composes a TokenEmbedLayer with one or more
ModalityEncoders + splice strategies.

Implements the :class:`~flextrain.core.layer.InputLayer` Protocol plus
the optional ``setup_round`` / ``finalize_round`` hooks. The engine
calls:

1. ``setup_round(prepared, ctx)`` -- once at the start of each round,
   BEFORE the per-chunk embed-forward loop. Gathers ``pixel_values``
   from all sequences in the round, DMAs to GPU, invokes each encoder's
   ``forward_round``, stashes the result in ``self._round_cache``.
2. ``forward(token_ids, chunk, weights, ctx)`` -- once per chunk.
   Runs ``TokenEmbedLayer.forward`` then applies each encoder's splice
   strategy in order, returning the merged residual stream.
3. ``backward(dx, token_ids, chunk, weights, grads, ctx)`` -- once per
   chunk. Applies each splice strategy's backward (reverse order),
   routing placeholder-position dx into the per-encoder grad
   accumulator (Phase 3) and zeroing those rows in ``d_text_emb`` to
   protect the embed-table row. Finally calls
   ``TokenEmbedLayer.backward`` on the masked ``d_text_emb``.
4. ``finalize_round(prepared, ctx)`` -- once at the end of each round,
   AFTER the per-chunk embed-backward loop. Phase 1 with frozen
   encoders is a no-op; Phase 3 calls each encoder's
   ``backward_round`` to accumulate encoder param grads from the
   per-round accumulator.

Param surface
-------------
``param_spec`` is the MERGE of ``text_embed.param_spec`` and every
encoder's ``param_spec``. Encoder tensors are
``TensorSpec(frozen=True)`` in Phase 1; the engine's existing
frozen-skip code in :func:`flextrain.engine.buffers.param_spec_byte_size`
silently skips grad / opt-state allocation for those entries.

Param-name collision avoidance: each encoder names its tensors with
the prefix ``f"{modality}{encoder_id}_"`` (enforced by
:func:`flextrain.nn.encoders.qwen_vl_vit.qwen_vl_vit_param_spec`); the
text-embed table is just ``"w_tok_embeddings"``. The merged dict has
unique keys by construction.

Sequence / ChunkMeta contract
-----------------------------
The data adapter populates each ``Sequence.modality_inputs[modality]``
with a list of :class:`~flextrain.core.modality.ImageInputCPU`. Chunk
preparation populates the per-chunk metadata that each splice
strategy reads from ``chunk.meta.extra`` -- see
:mod:`flextrain.nn.splices.concat` for the contract.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, MutableMapping, Sequence as _Seq

import torch

from flextrain.core.activation_schema import ActivationSchema
from flextrain.core.layer import (
    ChunkMeta,
    ComputeCost,
    LayerContext,
    ParamSpec,
)
from flextrain.core.modality import (
    ImageEmbeddings,
    ImageGradInputs,
    ImageInputCPU,
    ImageInputs,
)
from flextrain.nn.embed import TokenEmbedLayer


# Splice strategy = pair of pure functions, see flextrain/nn/splices/.
SpliceFwd = Callable[
    [torch.Tensor, ImageEmbeddings, Any, int],
    torch.Tensor,
]
SpliceBwd = Callable[
    [torch.Tensor, ImageEmbeddings, "ImageGradInputs | None", Any, int],
    torch.Tensor,
]


class MultimodalInputLayer:
    """Composite :class:`InputLayer` for multimodal training.

    Phase 1: image-only, frozen encoders (Qwen-VL family).
    """

    # Conceptually the layer has no layer_id (runs once per chunk before
    # the layer loop, same as :class:`TokenEmbedLayer`).
    layer_id: int = -1

    def __init__(
        self,
        text_embed: TokenEmbedLayer,
        encoders: _Seq[Any],
        splice_strategies: _Seq[tuple[SpliceFwd, SpliceBwd]],
    ) -> None:
        if len(encoders) == 0:
            raise ValueError(
                "MultimodalInputLayer needs at least one ModalityEncoder. "
                "For pure text-only training, use TokenEmbedLayer directly."
            )
        if len(encoders) != len(splice_strategies):
            raise ValueError(
                f"encoder count {len(encoders)} != splice strategy count "
                f"{len(splice_strategies)}; one strategy per encoder."
            )
        self.text_embed = text_embed
        self.encoders = tuple(encoders)
        self.splice_strategies = tuple(splice_strategies)

        # Same schema as TokenEmbedLayer -- empty, no per-chunk slot.
        self.schema = ActivationSchema(fields=(), max_tier=0)
        # Merged param_spec: text + all encoders.
        merged_specs = [self.text_embed.param_spec] + [
            e.param_spec for e in self.encoders
        ]
        self.param_spec = ParamSpec.merge(merged_specs)

        # Per-round state. Populated in ``setup_round``, consumed by
        # per-chunk forward/backward, cleared at ``finalize_round``.
        self._round_cache: dict[int, ImageEmbeddings] = {}
        self._round_inputs: dict[int, ImageInputs] = {}
        self._round_grad_accum: dict[int, ImageGradInputs] = {}

    # ------------------------------------------------------------------
    # Convenience: total vision-layer count (consumed by
    # ``ActiveModel.load_hf`` to thread num_vision_layers through to
    # the loader). Sum across encoders -- the arch loader's
    # ``vision_layer`` entries cycle through ``i in range(N)`` where N
    # is the encoder's depth. For multi-encoder configs the arch
    # loader's entries must use distinct prefixes; the loader sees a
    # flat ``num_vision_layers`` and the per-encoder entries enable
    # ``optional=True`` so missing entries don't fault.
    #
    # Phase 1 has one encoder so this is just encoders[0].num_vision_layers.
    # ------------------------------------------------------------------

    @property
    def num_vision_layers(self) -> int:
        if not self.encoders:
            return 0
        return max(
            int(getattr(e, "num_vision_layers", 0)) for e in self.encoders
        )

    # ------------------------------------------------------------------
    # InputLayer Protocol -- per-chunk forward / backward
    # ------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> torch.Tensor:
        """Run text embed + per-encoder splice for one chunk.

        The text-embed forward is unchanged from :class:`TokenEmbedLayer`;
        each splice strategy applies after, replacing placeholder rows
        in the text-emb tensor with the cached encoder rows.
        """
        text_emb = self.text_embed.forward(token_ids, chunk, weights, ctx)
        # Apply each encoder's splice strategy in order.
        for encoder, (splice_fwd, _splice_bwd) in zip(
            self.encoders, self.splice_strategies,
        ):
            cached = self._round_cache.get(encoder.encoder_id)
            if cached is None:
                # No data of this modality this round; splice is a no-op.
                continue
            text_emb = splice_fwd(text_emb, cached, chunk, encoder.encoder_id)
        return text_emb

    def backward(
        self,
        dx: torch.Tensor,
        token_ids: torch.Tensor,
        chunk: ChunkMeta,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        ctx: LayerContext,
    ) -> None:
        """Per-chunk embed backward.

        For each encoder (reversed): apply splice_bwd, which (Phase 3)
        accumulates placeholder-position dx into the encoder grad
        accumulator and (Phase 1) zeros those rows so the embed-table
        scatter-add doesn't double-count them. Then call
        :meth:`TokenEmbedLayer.backward` on the masked dx.
        """
        for encoder, (_splice_fwd, splice_bwd) in zip(
            reversed(self.encoders), reversed(self.splice_strategies),
        ):
            cached = self._round_cache.get(encoder.encoder_id)
            if cached is None:
                continue
            accum = self._round_grad_accum.get(encoder.encoder_id)  # may be None
            dx = splice_bwd(dx, cached, accum, chunk, encoder.encoder_id)
        # Text embed scatter-add for non-placeholder rows.
        self.text_embed.backward(dx, token_ids, chunk, weights, grads, ctx)

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        # The per-chunk splice is bandwidth-bound (small scatter); the
        # encoder forward isn't a per-chunk cost (it ran once per round
        # in setup_round). Mirror :meth:`TokenEmbedLayer.compute_cost`
        # and report zero -- the DP solver doesn't see the input layer.
        _ = chunk
        return ComputeCost(total_fwd_flops=0, avoided_recompute_flops=(0,))

    # ------------------------------------------------------------------
    # Round hooks -- optional InputLayer Protocol methods. Engine guards
    # the call with ``hasattr(self.embed, "setup_round")``.
    # ------------------------------------------------------------------

    def setup_round(self, prepared: Any, ctx: LayerContext) -> None:
        """Aggregate per-round modality inputs and run encoder forwards.

        ``prepared`` is :class:`~flextrain.engine.schedule.PreparedRound`.
        We extract every sequence's ``modality_inputs["image"]``, build
        a packed device-side :class:`ImageInputs`, then invoke each
        encoder's ``forward_round`` and cache the result.
        """
        # Determine the device from the existing embed buffers if we
        # can; fall back to cuda:0 (the compute stream's device).
        device = ctx.stream.device  # torch.cuda.Stream.device since PT 2.0+
        # Walk sequences in this round and collect their image inputs.
        per_encoder_inputs: dict[str, list[ImageInputCPU]] = {}
        seqs = self._gather_round_sequences(prepared)
        for seq in seqs:
            mod_in = getattr(seq, "modality_inputs", None) or {}
            for modality, items in mod_in.items():
                per_encoder_inputs.setdefault(modality, []).extend(items or [])

        # Drop any state from a prior round.
        self._round_cache.clear()
        self._round_inputs.clear()
        self._round_grad_accum.clear()

        for encoder in self.encoders:
            modality = encoder.modality
            items: list[ImageInputCPU] = per_encoder_inputs.get(modality, [])
            if not items:
                continue
            packed = self._pack_image_inputs(items, device)
            self._round_inputs[encoder.encoder_id] = packed
            # Build the per-modality weight slice (engine passes the
            # full gpu_embed_params dict here via the engine context;
            # we filter to just this encoder's prefix).
            # In practice the encoder forward reads by full name from
            # the same dict the engine builds at call time, so we look
            # it up via the input layer's weights dict at call sites.
            # For setup_round there is no engine-side weights argument
            # to forward(), so the encoder reads from
            # ``ctx`` -- we route the weights dict via a slot on
            # ``ctx`` if present, otherwise raise (Phase 1 keeps the
            # API simple: the engine threads the gpu_embed_params dict
            # as ``ctx._mm_weights``).
            weights = getattr(ctx, "_mm_weights", None)
            if weights is None:
                raise RuntimeError(
                    "MultimodalInputLayer.setup_round: ctx is missing "
                    "the ``_mm_weights`` attribute. The engine must "
                    "attach the gpu_embed_params dict before calling "
                    "setup_round (see ActiveModel._setup_round)."
                )
            embeds = encoder.forward_round(packed, weights, ctx)
            self._round_cache[encoder.encoder_id] = embeds

    def finalize_round(self, prepared: Any, ctx: LayerContext) -> None:
        """Drop per-round caches. Phase 3: also call each encoder's
        ``backward_round`` to accumulate encoder param grads."""
        _ = prepared
        for encoder in self.encoders:
            accum = self._round_grad_accum.get(encoder.encoder_id)
            inputs = self._round_inputs.get(encoder.encoder_id)
            if accum is not None and inputs is not None:
                # Phase 3: real backward pass.
                weights = getattr(ctx, "_mm_weights", None)
                # The arch-loader threads gpu_embed_grads as
                # ``ctx._mm_grads``; Phase 1 frozen encoders never need
                # this. The placeholder code below is a no-op until
                # Phase 3 wires up the trainable path.
                if weights is None:
                    raise RuntimeError(
                        "finalize_round: ctx missing ``_mm_weights``."
                    )
                grads_dict = getattr(ctx, "_mm_grads", None) or {}
                encoder.backward_round(accum, inputs, weights, grads_dict, ctx)
        # Free per-round state.
        self._round_cache.clear()
        self._round_inputs.clear()
        self._round_grad_accum.clear()

    # ------------------------------------------------------------------
    # Helpers (private)
    # ------------------------------------------------------------------

    def _gather_round_sequences(self, prepared: Any) -> list[Any]:
        """Walk ``prepared.chunks`` to assemble the set of
        :class:`Sequence` objects contributing to this round.

        ``chunk.seqs`` is a list of
        :class:`~flextrain.engine.schedule.ChunkSeqRef` (NOT
        :class:`Sequence` directly) -- each ref carries a ``.seq``
        attribute pointing to the underlying Sequence plus the
        chunk-local / seq-local ranges. We unwrap to the actual
        Sequence and de-duplicate by identity (a long sequence can
        contribute to multiple chunks via large-seq splitting).
        """
        seen: set[int] = set()
        ordered: list[Any] = []
        for chunk in getattr(prepared, "chunks", []) or []:
            seqs = getattr(chunk, "seqs", None) or []
            for ref in seqs:
                # ChunkSeqRef.seq is the actual Sequence; older
                # callers (e.g. unit tests) may pass a flat list of
                # Sequences directly, so fall back if no ``.seq``.
                s = getattr(ref, "seq", ref)
                key = id(s)
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(s)
        return ordered

    def _pack_image_inputs(
        self, items: list[ImageInputCPU], device: torch.device,
    ) -> ImageInputs:
        """Pack a list of per-image CPU bundles into a device-side
        :class:`ImageInputs`.

        Concatenates ``pixel_values`` along dim 0 (each item is
        ``(n_patches_i, patch_dim)``), builds prefix-sum offsets, and
        stacks ``grid_thw`` to ``(n_images, 3)``.
        """
        pix_list = [it.pixel_values.to(device, non_blocking=True) for it in items]
        pix_offsets_host = [0]
        for pv in pix_list:
            pix_offsets_host.append(pix_offsets_host[-1] + pv.shape[0])
        pix_offsets = torch.tensor(
            pix_offsets_host, dtype=torch.int32, device=device,
        )
        pixel_values = torch.cat(pix_list, dim=0).contiguous()
        grid_thw = torch.stack(
            [it.grid_thw.to(device, non_blocking=True).to(torch.int32) for it in items],
            dim=0,
        )
        return ImageInputs(
            pixel_values=pixel_values,
            pix_offsets=pix_offsets,
            grid_thw=grid_thw,
        )
