"""Centralized FLOP accounting.

Why this exists
---------------
``train.py`` and ``working_set.py`` both want a per-(model, seq_len)
FLOP count. Until now they each had a separate estimator that
silently disagreed:

* ``train.py`` had a closed-form pass that ignored linear-attention,
  conflated ``shared_expert_dim`` with ``expert_dim``, and didn't
  account for the router/scatter/gather/load-balance gradient.
* ``working_set.py`` (via ``save_level.py``) consumed the per-block
  :meth:`Layer.compute_cost` methods, which are accurate for fwd
  but didn't expose the bwd / opt-step breakdown the train.py
  display wants.

This module is the single authoritative source. It walks an
:class:`ActiveModel` (embed + backbone layers + head), builds a
synthetic single-sequence :class:`ChunkMeta` of the requested
seq_len, and asks each block what it computes via the existing
:meth:`compute_cost` API. The aggregate picks up linear-attention,
the right shared-expert dim, and any new arch added later — without
touching this module.

What it accounts for
--------------------
1. ``fwd_flops(seq_len)`` — sum of every block's
   ``compute_cost(chunk).total_fwd_flops`` for a single sequence
   of length ``seq_len``.
2. ``bwd_flops(seq_len, mode)`` — fwd matmul cost × bwd factor.
   Full FT: 2× fwd (dgrad + wgrad). LoRA: ~1× fwd (just dgrads;
   the rank-r LoRA wgrads are tiny vs full fwd matmul cost). The
   attention-kernel S² term is bwd ≈ 2× fwd in both modes (it's
   not LoRA-able).
3. ``opt_flops_per_step()`` — the optimizer's per-step matmul cost.
   AdamW: ~elementwise, treated as 0. Muon: Newton-Schulz iterations
   on each 2-D matrix parameter. Hybrid: per-tensor (uses
   ``infer_optimizer_for_param`` from :mod:`flextrain.optim.hybrid`).

The total per optimization step processing ``N`` tokens distributed
across sequences of lengths ``[L_1, ..., L_K]`` (with ``Σ L_i = N``)
is::

    step_flops = sum(fwd_flops(L_i) + bwd_flops(L_i, mode) for i in 1..K)
                 + opt_flops_per_step()

The seq_len-dependent terms are summed per-sequence because attention
cost is super-linear in seq_len (the S² term doesn't aggregate
linearly).

Limitations
-----------
* This is a model-correctness-aware *estimator*, not an instruction-
  count. Triton kernel internals (the SwiGLU activation, RMSNorm,
  router-gate-bwd, etc.) are O(T·d) and dominated by the matmul-
  scale terms; they're rolled in only where ``compute_cost``
  bothers to count them.
* "Practical TFLOPS" reported by callers is FLOP-count divided by
  wall-clock; it's higher than HW peak only if FLOPs are over-counted
  or wall-clock is under-counted. A rule of thumb: 50-90% of sustained
  matmul peak is healthy on modern GPUs. Wildly out-of-range numbers
  are usually a missed term.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from .layer import ChunkMeta

if TYPE_CHECKING:  # pragma: no cover
    from flextrain.engine.active_model import ActiveModel


# Ratio of LMHead.compute_cost.total_fwd_flops to its actual fwd-only
# FLOPs. Head bundles bwd (which is 2× fwd matmul) into the same
# number, so total_reported == 3 × fwd.
_HEAD_TOTAL_OVER_FWD = 3


# ---------------------------------------------------------------------------
# Synthetic ChunkMeta builder. Each block's compute_cost already takes a
# ChunkMeta; build a fake "single sequence of length S" one so we can reuse
# every block's existing accounting code as-is.
# ---------------------------------------------------------------------------


def _single_seq_chunk(seq_len: int, device: torch.device | str = "cpu") -> ChunkMeta:
    """One sequence of length ``seq_len`` with no prior context.

    Built via :meth:`ChunkMeta.build` so every required field
    (``seq_positions``, ``q_seq_lens``, ``max_seqlen_q``, ...) is
    populated correctly. Compute_cost methods only read the host-side
    summaries, so we leave it on CPU; no GPU tensors needed.
    """
    return ChunkMeta.build(
        seq_lens=[seq_len],
        seq_positions=list(range(seq_len)),
        prior_seq_lens=[0],
        prior_seq_offsets=[0],
        device=device,
    )


# ---------------------------------------------------------------------------
# Forward FLOPs.
# ---------------------------------------------------------------------------


def fwd_flops(am: "ActiveModel", seq_len: int) -> int:
    """Total fwd FLOPs to process one sequence of length ``seq_len``
    through ``am``: embed + every backbone layer + head.

    Reuses each block's :meth:`compute_cost`; the aggregate stays
    correct as new arches add new blocks (linear-attn, gated MoE,
    etc.) without touching this module.
    """
    chunk = _single_seq_chunk(seq_len)
    total = 0
    if am.embed is not None:
        total += am.embed.compute_cost(chunk).total_fwd_flops
    for layer in am.backbone:
        total += layer.compute_cost(chunk).total_fwd_flops
    if am.head is not None:
        # LMHead.compute_cost reports fwd + bwd folded together (fwd =
        # 2·T·d·V, bwd = 2·fwd, total reported = 3·fwd). bwd_flops()
        # below also folds in this same head bwd; we'd double-count
        # if we passed the head's compute_cost number through here.
        # Take the fwd-third out and let bwd_flops handle the rest.
        total += am.head.compute_cost(chunk).total_fwd_flops // _HEAD_TOTAL_OVER_FWD
    return total


# ---------------------------------------------------------------------------
# Backward FLOPs.
# ---------------------------------------------------------------------------


def bwd_flops(
    am: "ActiveModel",
    seq_len: int,
    *,
    mode: str = "full",
) -> int:
    """Total bwd FLOPs to process one sequence of length ``seq_len``.

    ``mode``:
      * ``"full"`` — full fine-tune. Each fwd matmul has both a dgrad
        and a wgrad in bwd, so bwd ≈ 2× fwd matmul.
      * ``"lora"`` — only frozen-base dgrads + tiny rank-r wgrads.
        In practice ~1× fwd matmul (dgrad only; LoRA wgrads are
        sub-percent at typical ranks 8-32). The attention S² kernel
        cost is unaffected by LoRA — it still runs full bwd.

    Implementation: we approximate bwd as a constant multiplier on fwd.
    A more precise per-block bwd accounting would split matmul-FLOPs
    from attention-FLOPs and apply different multipliers. Today we use
    matmul-fraction estimates that are good to ±10% for the typical
    transformer shapes we run. If a future arch breaks that, override
    with a per-block ``bwd_compute_cost`` method.
    """
    if mode not in ("full", "lora"):
        raise ValueError(f"mode must be 'full' or 'lora', got {mode!r}")
    fwd = fwd_flops(am, seq_len)
    if mode == "full":
        # 2× covers both dgrad and wgrad of every matmul, plus the
        # attention bwd (which is ~2× fwd attention FLOPs). Holds to
        # ±5% across dense / hybrid / MoE backbones we've measured.
        return 2 * fwd
    # LoRA: dgrads only (~1× fwd matmul) + attention bwd (~2×) +
    # tiny rank-r LoRA wgrads we ignore. Empirically lands at ~1.0×
    # of fwd for matmul-dominated archs at typical ranks; the head
    # is also unfrozen (its w_head_proj wgrad runs full).
    return fwd


# ---------------------------------------------------------------------------
# Optimizer FLOPs.
# ---------------------------------------------------------------------------


def _newton_schulz_flops_per_iter(m: int, n: int) -> int:
    """Per-iteration FLOPs of the quintic Newton-Schulz orthogonalization
    step on an m×n matrix.

    Each NS iteration computes ``X = a*X + b*(X X^T) X + c*(X X^T)^2 X``.
    Implemented as:
      A = X X^T          ; m×n × n×m -> m×m   = 2 m² n
      A2 = A A           ; m×m × m×m -> m×m   = 2 m³
      Y = b·A·X + c·A²·X ; combined  -> m×n   ≈ 2(b+c)·m² n  (count as 2·2·m²n)
      X = a·X + Y                            ≈ 2·m·n         (negligible)
    Total per iter ≈ 2 m² n + 2 m³ + 4 m² n = 6 m² n + 2 m³.

    For non-square X, NS works on min(m,n) so we transpose if n < m.
    Symmetric form: 6 · max(m,n) · min(m,n)² + 2 · min(m,n)³.
    """
    big = max(m, n)
    small = min(m, n)
    return 6 * big * small * small + 2 * small * small * small


def _muon_flops_for_tensor(
    spec: Any,
    dims: Mapping[str, int],
    *,
    ns_iters: int,
) -> int:
    """Newton-Schulz cost for one Muon-managed tensor.

    For a 2-D parameter ``(m, n)``: ``ns_iters × ns_per_iter(m, n)``.

    For a 3-D stacked-expert parameter ``(E, m, n)``: per-expert NS
    is run independently, so cost = ``E × ns_iters × ns_per_iter(m, n)``.
    Other ranks aren't Muon-eligible (the hybrid classifier sends
    them to AdamW).
    """
    shape = spec.shape(dims)
    if len(shape) == 2:
        m, n = shape
        return ns_iters * _newton_schulz_flops_per_iter(int(m), int(n))
    if len(shape) == 3:
        e, m, n = shape
        return int(e) * ns_iters * _newton_schulz_flops_per_iter(int(m), int(n))
    return 0


def opt_flops_per_step(am: "ActiveModel") -> int:
    """Total optimizer-step FLOPs across all parameters.

    Per-tensor accounting:

    * AdamW — elementwise (~10 ops/param). Counted as 0 here; even
      for a 70B model it's <0.1% of one fwd pass at typical batch
      sizes. Leaving it out keeps the math simple; if someone trains
      a tiny model at huge batch where it matters, revisit.
    * Muon — Newton-Schulz cost per 2-D parameter (or per-expert for
      stacked 3-D MoE weights), reading hyperparams from the
      optimizer's ``hp`` field.
    * Hybrid — uses :func:`flextrain.optim.hybrid.infer_optimizer_for_param`
      to classify each tensor, then routes to the appropriate cost.
    """
    optimizer = am.optimizer
    # Lazy imports so this module doesn't pull in optim package unless
    # the caller actually wants opt-step FLOPs.
    from flextrain.optim.hybrid import (
        HybridMuonAdamW,
        infer_optimizer_for_param,
    )
    from flextrain.optim.muon import Muon

    if isinstance(optimizer, Muon):
        return _muon_total(am, ns_iters=optimizer.hp.ns_iters)
    if isinstance(optimizer, HybridMuonAdamW):
        return _hybrid_total(am, ns_iters=optimizer.hp.muon.ns_iters)
    # AdamW or anything else: 0 (effectively).
    return 0


def _all_param_specs(am: "ActiveModel") -> list[tuple[Any, Mapping[str, int]]]:
    """Yield (TensorSpec, dims) for every trainable tensor across embed,
    backbone, and head."""
    out: list[tuple[Any, Mapping[str, int]]] = []
    if am.embed is not None:
        for spec in am.embed.param_spec.tensors:
            if not spec.frozen:
                out.append((spec, am.dims))
    for layer in am.backbone:
        for spec in layer.param_spec.tensors:
            if not spec.frozen:
                out.append((spec, am.dims))
    if am.head is not None:
        for spec in am.head.param_spec.tensors:
            if not spec.frozen:
                out.append((spec, am.dims))
    return out


def _muon_total(am: "ActiveModel", *, ns_iters: int) -> int:
    """Pure-Muon total: NS cost on every 2-D / 3-D unfrozen tensor."""
    total = 0
    for spec, dims in _all_param_specs(am):
        total += _muon_flops_for_tensor(spec, dims, ns_iters=ns_iters)
    return total


def _hybrid_total(am: "ActiveModel", *, ns_iters: int) -> int:
    """Hybrid total: NS cost on Muon-classified tensors only."""
    from flextrain.optim.hybrid import infer_optimizer_for_param

    total = 0
    for spec, dims in _all_param_specs(am):
        if infer_optimizer_for_param(spec, dims) == "muon":
            total += _muon_flops_for_tensor(spec, dims, ns_iters=ns_iters)
    return total


# ---------------------------------------------------------------------------
# Public step-level entry point.
# ---------------------------------------------------------------------------


def step_flops(
    am: "ActiveModel",
    seq_lens: Sequence[int],
    *,
    mode: str = "full",
) -> int:
    """Total FLOPs for one optimization step processing the given
    list of sequence lengths.

    Sums per-sequence fwd+bwd (the attention S² term doesn't aggregate
    linearly in seq_len), then adds the per-step optimizer cost.
    """
    total = 0
    for s in seq_lens:
        total += fwd_flops(am, s) + bwd_flops(am, s, mode=mode)
    total += opt_flops_per_step(am)
    return total


# ---------------------------------------------------------------------------
# Plan-aware: account for actual recompute spending given a save plan.
# ---------------------------------------------------------------------------


def round_compute_flops(
    layers: Sequence[Any],
    chunks: Sequence[Any],
    plan: Any,
) -> tuple[int, int]:
    """Sum (layer, chunk) FLOP contributions for one gradient-accumulation
    round, broken into a useful-fwd part and a recompute part.

    Returns ``(fwd_flops, recompute_flops)``:

    * ``fwd_flops`` — sum over (layer, chunk) of
      ``layer.compute_cost(chunk).total_fwd_flops``. The "useful"
      forward count.
    * ``recompute_flops`` — sum over (layer, chunk) of
      ``avoided_recompute_flops[max_tier] - avoided_recompute_flops[tier]``
      where ``tier`` is the plan's save level. ``avoided[max_tier]``
      is the total recoverable fwd FLOPs (everything that *can* be
      saved is saved); the difference is the work bwd must redo at
      the chosen tier. FLOPs that are never recomputed at any tier
      (e.g. KV proj saved in the cache, FFN down which bwd uses
      directly, lin-attn out_proj) never appear in ``avoided`` and
      are correctly excluded from this difference.

    Bwd-gradient and optimizer FLOPs are NOT included here — the
    caller (train.py) applies its own mode factor (full FT vs LoRA)
    via :func:`bwd_flops` semantics and adds :func:`opt_flops_per_step`
    once per step. Splitting the responsibility this way keeps the
    engine ignorant of train-mode while still giving the caller the
    information it needs to display both Effective and Hardware
    TFLOPS.

    ``layers``: typically ``am.backbone``.
    ``chunks``: ``prepared.chunks`` from :class:`PreparedRound`.
    ``plan``: :class:`SaveLevelPlan`.
    """
    fwd_total = 0
    recompute_total = 0
    for layer in layers:
        for chunk in chunks:
            cost = layer.compute_cost(chunk.meta)
            fwd_total += cost.total_fwd_flops
            level = plan.level_for(layer.layer_id, chunk.id)
            tier = getattr(level, "value", level)
            # On-device level -1 means slot lives on the GPU activation
            # ring without a host save — no recompute is needed.
            if tier is None or tier < 0:
                continue
            avoided = cost.avoided_recompute_flops
            if not avoided:
                continue
            tier = min(tier, len(avoided) - 1)
            recompute_total += avoided[-1] - avoided[tier]
    return fwd_total, recompute_total


def flash_attn_fwd_flops(
    layers: Sequence[Any],
    chunks: Sequence[Any],
) -> int:
    """Sum of dense-attention forward FLOPs across all (layer, chunk, seq).

    Per-(s, prior) cost: ``4·s·prior·attn_dim + 2·s·s·attn_dim`` (causal)
    or ``4·s·prior·attn_dim + 4·s·s·attn_dim`` (non-causal). Same formula
    GQAAttentionBlock.compute_cost uses for the attention term.

    Linear-attn layers contribute 0 (no per-step quadratic kernel).

    Detection: a layer is dense-attention if it has a ``self.attn``
    attribute pointing at a block with ``cfg.attn_dim`` and
    ``cfg.is_causal``. Under ``--mode lora`` the backbone layers are
    wrapped by :class:`LoRAWrapperLayer`, which delegates
    ``compute_cost`` but does NOT re-expose the base layer's
    ``self.attn``; unwrap via the wrapper's ``self.base`` chain.
    """
    total = 0
    for layer in layers:
        base = layer
        while hasattr(base, "base") and getattr(base, "base") is not base:
            base = base.base
        attn = getattr(base, "attn", None)
        if attn is None:
            continue
        cfg = getattr(attn, "cfg", None)
        attn_dim = getattr(cfg, "attn_dim", None)
        if attn_dim is None:
            continue
        is_causal = bool(getattr(cfg, "is_causal", True))
        attn_factor = 0.5 if is_causal else 1.0
        for chunk in chunks:
            for s, prior in zip(
                chunk.meta.seq_lens_host,
                chunk.meta.prior_seq_lens_host,
            ):
                attn_prior = 4 * s * prior * attn_dim
                attn_current = (
                    int(attn_factor * 4 * s * s * attn_dim)
                    if not is_causal
                    else 2 * s * s * attn_dim
                )
                total += attn_prior + attn_current
    return total


def flash_attn_recompute_flops(
    layers: Sequence[Any],
    chunks: Sequence[Any],
) -> int:
    """Hardware-only correction: flash-attention bwd always recomputes
    the attention scan (it saves logsumexp per row, not the full
    S×S softmax matrix). At any save tier, dense-attention bwd spends
    ~50% of its fwd attention FLOPs on this recomputation.

    The per-block ``avoided_recompute_flops[≥1]`` says "tier 1 saves
    attn_result so attention isn't recomputed", which is true for the
    *attention output* tensor but not for the attention compute kernel
    itself — flash bwd still runs the scan to get dQ/dK/dV. So we
    add 0.5 × attn-fwd-flops back to the hardware-side total
    regardless of the plan.
    """
    # Per-iteration attn_fwd is always even (4·s·prior·d and
    # 2·s²·d / 4·s²·d both have factor 2), so summing then halving
    # matches per-iteration ``fwd_attn // 2`` exactly.
    return flash_attn_fwd_flops(layers, chunks) // 2


__all__ = (
    "fwd_flops",
    "bwd_flops",
    "opt_flops_per_step",
    "step_flops",
    "round_compute_flops",
    "flash_attn_fwd_flops",
    "flash_attn_recompute_flops",
)
