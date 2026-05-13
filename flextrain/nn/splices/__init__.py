"""Splice strategies for :class:`MultimodalInputLayer`.

A splice strategy combines a chunk's text embeddings with the
per-round encoder cache (image / audio / video embeddings) into the
final residual stream that flows into the backbone.

Two known shapes:

* :mod:`flextrain.nn.splices.concat` -- Qwen-VL pattern.
  Vision-token positions in ``input_ids`` are pre-expanded to N slots
  (driven by ``image_grid_thw`` per image); encoder rows scatter onto
  those slots; 3-D MRoPE positions are carried via
  ``chunk.seq_positions``. Phase 1 ships this one.
* :mod:`flextrain.nn.splices.substitution` (Phase 2) -- Gemma3 /
  Gemma4 pattern. Each placeholder token is a single position; encoder
  produces a fixed N-token block that scatters onto N consecutive
  placeholders. 1-D RoPE.

A "strategy" is a pair of pure functions ``(splice_fwd, splice_bwd)``
keyed by name; :class:`MultimodalInputLayer` stores one pair per
encoder.
"""

from .concat import concat_splice_bwd, concat_splice_fwd

__all__ = [
    "concat_splice_bwd",
    "concat_splice_fwd",
]
