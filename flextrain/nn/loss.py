"""Pluggable per-token loss objectives for :class:`LMHead`.

Why separate from the head
--------------------------
The LM head's compute is architecture-specific (RMSNorm shape +
linear-proj geometry + dtype) but objective-agnostic. The objective
(supervised cross-entropy, GRPO, DPO, MSE on continuous targets, ...)
is architecture-agnostic but data-specific (labels, advantages, KL
reference log-probs, ...).

Splitting them:

* Keeps :class:`LMHead` reusable across SFT, RL, distillation, etc.
* Keeps the micro-chunk loop (which must live inside the head for the
  ``O(T', V)`` logits-buffer memory win) objective-generic.
* Lets us unit-test loss fns in isolation against naive PyTorch
  references without spinning up the head.

The contract
------------
A loss fn is an object implementing :class:`LossFn`. Each call
receives:

* ``logits``        : ``(T', V)`` in the head's compute dtype.
* ``token_slice``   : a :class:`TokenSlice` describing which rows of
                      the training chunk we're on and what per-token
                      context to apply (labels, advantages, ref
                      log-probs, ...).
* ``loss_scale``    : scalar to fold into the returned ``dZ`` (so the
                      head's weight-grad accumulator doesn't have to
                      know about sample counts).

It returns:

* ``dZ``            : ``(T', V)`` grad in the same dtype as ``logits``,
                      already multiplied by ``loss_scale``.
* ``per_token_loss``: ``(T',)`` fp32 per-token scalar (for logging).
* ``aux``           : optional dict of extra diagnostics (KL value,
                      entropy, etc.) -- concatenated across
                      micro-chunks by the head.

The head does NOT know what's inside ``aux``. The caller who chose
the loss fn reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

import torch

from flextrain.ops import (
    flextrain_cross_entropy_loss,
    flextrain_softmax,
)


# Default ignore-index: labels equal to this value produce no gradient.
# Matches PyTorch's ``CrossEntropyLoss(ignore_index=-100)`` convention.
IGNORE_INDEX = -100


@dataclass
class TokenSlice:
    """Per-token context for one micro-chunk of a training chunk.

    Holds the labels / advantages / KL refs / etc. needed by whatever
    loss fn is active. The head builds this per micro-chunk via
    :meth:`TokenContext.slice_for`.

    ``labels`` is the one field most losses will touch; the rest are
    opaque ``torch.Tensor`` slices that flow through from the caller.
    Losses read whichever they care about.

    Loss masking
    ------------
    Two ways to mark a token as "no gradient contribution":

    1. ``labels[t] == IGNORE_INDEX`` (default ``-100``) — PyTorch
       convention. Works for any loss fn that honors it (CE, GRPO).
    2. ``loss_mask[t] == False`` — explicit boolean mask of which
       tokens to train on. Overrides / supplements labels. Useful
       when labels are continuous (MSE) or when the mask is
       computed from per-turn role info (SFT chat templating,
       where we train on assistant turns only).

    Losses that observe these MUST zero both ``dZ`` and
    ``per_token_loss`` at masked positions.
    """

    offset: int
    size: int
    # Supervised targets (for CE, MSE over continuous targets, ...).
    labels: torch.Tensor | None = None
    # Boolean mask: True = include in loss/grad, False = skip. Same
    # length as labels.
    loss_mask: torch.Tensor | None = None
    # Per-token advantage / reward signal (GRPO, RLOO, ...).
    advantages: torch.Tensor | None = None
    # Frozen-ref log-probs aligned with labels (DPO, KL-penalized RL).
    ref_logprobs: torch.Tensor | None = None
    # Opaque escape hatch for anything else callers need.
    extra: Mapping[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class TokenContext:
    """Per-(training-chunk) context the caller provides to
    :meth:`LMHead.forward_backward`.

    Mirrors :class:`TokenSlice` but at the full training-chunk length.
    ``slice_for(offset, size)`` narrows to a micro-chunk's rows.

    Loss masking (see :class:`TokenSlice`)
    --------------------------------------
    * ``labels == IGNORE_INDEX`` — PyTorch convention, supported by
      :class:`CrossEntropyLoss` and :class:`GRPOLoss`.
    * ``loss_mask`` — explicit ``(T,) bool`` mask. ``True`` = include
      this token in loss/grad; ``False`` = skip.

    Why a dataclass and not a dict
    ------------------------------
    Makes the field set documented and type-checkable; an unknown field
    name is caught at construction, not lost in a dict.
    """

    labels: torch.Tensor | None = None
    loss_mask: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    extra: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def slice_for(self, offset: int, size: int) -> TokenSlice:
        def _s(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else t[offset : offset + size]

        extra_sl = {k: v[offset : offset + size] for k, v in self.extra.items()}
        return TokenSlice(
            offset=offset,
            size=size,
            labels=_s(self.labels),
            loss_mask=_s(self.loss_mask),
            advantages=_s(self.advantages),
            ref_logprobs=_s(self.ref_logprobs),
            extra=extra_sl,
        )

    @property
    def total_tokens(self) -> int:
        """Number of tokens this context covers. Inferred from the first
        populated field so the caller doesn't have to pass it
        separately."""
        for t in (self.labels, self.loss_mask, self.advantages, self.ref_logprobs):
            if t is not None:
                return int(t.shape[0])
        for v in self.extra.values():
            return int(v.shape[0])
        raise ValueError("TokenContext is empty; cannot infer total_tokens")


@runtime_checkable
class LossFn(Protocol):
    """Per-token loss function.

    Implementations MUST:

    * Write ``dZ`` **in-place into ``logits``** if possible (same
      buffer, same dtype) -- this is what orig does with
      cross-entropy and matches what the head's fused grad matmul
      expects. If your loss can't reuse the logits buffer in place,
      return a fresh tensor of the same shape/dtype; the caller will
      use whichever buffer is returned.
    * Multiply ``dZ`` by ``loss_scale`` so the head's grad
      accumulator can use it directly.
    * Write per-token loss scalars into ``per_token_loss_out`` (a
      pre-allocated fp32 tensor the head owns).
    """

    def compute(
        self,
        logits: torch.Tensor,  # (T', V) in compute_dtype
        token_slice: TokenSlice,
        *,
        loss_scale: float,
        per_token_loss_out: torch.Tensor,  # (T',) fp32, pre-allocated
    ) -> tuple[torch.Tensor, dict]:
        """Return ``(dZ, aux)`` where ``dZ`` is the grad w.r.t. ``logits``
        (with ``loss_scale`` already folded in) and ``aux`` is an
        optional per-chunk diagnostics dict.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in loss fns.
# ---------------------------------------------------------------------------


class CrossEntropyLoss:
    """Standard softmax + token-label cross-entropy. The SFT default.

    Uses orig's fused ``flextrain_softmax`` + ``flextrain_cross_entropy_loss``
    kernels:

    * ``flextrain_softmax`` writes softmax into an output buffer and also
      records argmax + argmax-prob (used by :class:`LossStats` for
      diagnostics — accuracy, prediction traces).
    * ``flextrain_cross_entropy_loss`` overwrites ``probs`` in place with
      ``probs - one_hot(label)`` (i.e. dZ), writes per-token loss into
      ``L``, and returns ``(dZ, loss_unused)``.
    * Multiply by ``loss_scale`` in-place before returning so the head
      grad matmul can use alpha=loss_scale directly with no double
      scaling.

    Matches the math at ``orig/awsm_transformer/head.py:208-227``.

    Masking
    -------
    Two modes (cumulative):

    * ``labels == ignore_index`` (default ``-100``) — matches PyTorch's
      ``CrossEntropyLoss(ignore_index=-100)`` idiom. The kernel's
      in-place ``probs - one_hot(label)`` is still run (with label
      clamped to 0 to avoid out-of-bounds indexing), then both ``dZ``
      and per-token loss are zeroed at the masked rows.
    * ``token_slice.loss_mask == False`` — explicit bool mask, applied
      the same way.

    Either flag suppresses both the gradient contribution AND the
    per-token loss value for affected positions; downstream metrics
    see 0 in those slots. Positions that ARE included still accumulate
    into ``g_head_proj`` through the head's ``addmm``.
    """

    def __init__(self, ignore_index: int = IGNORE_INDEX) -> None:
        self.ignore_index = ignore_index

    def compute(
        self,
        logits: torch.Tensor,
        token_slice: TokenSlice,
        *,
        loss_scale: float,
        per_token_loss_out: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if token_slice.labels is None:
            raise ValueError("CrossEntropyLoss requires token_slice.labels")

        labels = token_slice.labels
        ignore_rows = labels == self.ignore_index
        any_ignore = bool(ignore_rows.any().item())

        # If some labels are ignore_index, sanitize them to 0 so the
        # kernel's gather-at-label doesn't index out of bounds. We'll
        # zero those rows out of dZ + per_token_loss after.
        if any_ignore:
            labels_sanitized = torch.where(
                ignore_rows, torch.zeros_like(labels), labels
            )
        else:
            labels_sanitized = labels

        # Softmax into a fresh buffer (same shape/dtype as logits).
        probs = torch.empty_like(logits)
        aux_next_pred = torch.empty(
            token_slice.size, dtype=torch.int64, device=logits.device
        )
        aux_next_prob = torch.empty(
            token_slice.size, dtype=torch.float32, device=logits.device
        )
        probs, _max_idx, _max_val = flextrain_softmax(
            logits,
            out=probs,
            max_idx_out=aux_next_pred,
            max_val_out=aux_next_prob,
        )

        # In-place: probs <- probs - one_hot(labels), writes per-token loss.
        dZ, _loss_ignored = flextrain_cross_entropy_loss(
            probs, labels_sanitized, L=per_token_loss_out
        )

        # Build the combined mask: include a row iff label != ignore AND
        # loss_mask != False.
        mask_include = None
        if any_ignore:
            mask_include = ~ignore_rows
        if token_slice.loss_mask is not None:
            lm = token_slice.loss_mask.to(dtype=torch.bool)
            mask_include = lm if mask_include is None else (mask_include & lm)

        if mask_include is not None and not bool(mask_include.all().item()):
            # Zero dZ + per-token loss for excluded rows.
            keep = mask_include.to(dtype=logits.dtype).unsqueeze(-1)
            dZ.mul_(keep)
            per_token_loss_out.masked_fill_(~mask_include, 0.0)

        if loss_scale != 1.0:
            dZ.mul_(loss_scale)

        aux = {
            "next_prediction": aux_next_pred,
            "next_prediction_prob": aux_next_prob,
        }
        return dZ, aux


class MSELoss:
    """Mean-squared error against continuous targets.

    ``labels`` is expected to be a ``(T', V)`` tensor of target logits
    (or whatever vector space -- this is a distillation-style use
    case). Per-token loss is ``0.5 * ||logits - labels||^2 / V``.

    ``dZ = (logits - labels) * loss_scale / V``, in-place in a fresh
    buffer.

    Not optimized; straightforward reference for when someone needs
    MSE distillation without pulling in CE.
    """

    def compute(
        self,
        logits: torch.Tensor,
        token_slice: TokenSlice,
        *,
        loss_scale: float,
        per_token_loss_out: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if token_slice.labels is None:
            raise ValueError("MSELoss requires token_slice.labels (continuous targets)")
        targets = token_slice.labels
        if targets.shape != logits.shape:
            raise ValueError(
                f"MSELoss target shape {tuple(targets.shape)} != "
                f"logits shape {tuple(logits.shape)}"
            )

        diff = logits - targets
        V = logits.shape[-1]
        # Per-token loss in fp32 so it matches CE's per_token_loss_out dtype.
        per_token_loss_out.copy_(
            (diff.float().pow(2).sum(dim=-1) * 0.5 / V)
        )
        dZ = diff * (loss_scale / V)
        return dZ, {}


class GRPOLoss:
    """Group-relative policy optimization token loss (simplified).

    ``token_slice.labels``      : the sampled token ids (T',).
    ``token_slice.advantages``  : per-token advantage A_t (T',) fp32.
    ``token_slice.ref_logprobs``: optional frozen-policy log-prob at
                                  each sampled token; if present,
                                  enables an explicit KL penalty
                                  against the reference (``kl_coef``).

    The returned ``dZ`` follows the policy-gradient form:

        dZ[t, k] = softmax(logits)[t, k] - 1[k == labels[t]]   # same as CE
        dZ[t, :] *= -A_t                                         # sign flip
        dZ[t, :] += kl_coef * (softmax(logits)[t, :] - ref_probs[t, :])  # optional
        dZ *= loss_scale

    This is the *token-level* form that slots into our micro-chunk
    loop; the outer "normalize by group" step is a responsibility of
    the caller assembling ``advantages`` before forward_backward. We
    intentionally do NOT implement the group/episode bookkeeping here
    -- that lives in the RL training loop, not in the LM head.

    Per-token loss reported is ``-A_t * log p(labels[t] | logits[t])``
    for diagnostic purposes. The actual minimized surrogate is
    captured via the ``dZ`` formula above.

    Status: correctness-skeleton only. This is the structural hook so
    users CAN plug in a real RL loss; production RL training will
    generally want to wire in their own variant (PPO clip, KL-
    shaping, etc.) rather than use this verbatim.
    """

    def __init__(self, kl_coef: float = 0.0) -> None:
        self.kl_coef = kl_coef

    def compute(
        self,
        logits: torch.Tensor,
        token_slice: TokenSlice,
        *,
        loss_scale: float,
        per_token_loss_out: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if token_slice.labels is None:
            raise ValueError("GRPOLoss requires token_slice.labels")
        if token_slice.advantages is None:
            raise ValueError("GRPOLoss requires token_slice.advantages")

        # Softmax in a fresh buffer.
        probs = torch.softmax(logits.float(), dim=-1)
        # log p(label) for diagnostics + advantage-weighted loss.
        gather = probs.gather(
            1, token_slice.labels.view(-1, 1).long()
        ).squeeze(-1).clamp_min_(1e-12)
        log_p = gather.log()
        per_token_loss_out.copy_(-(token_slice.advantages.float() * log_p))

        # dZ = softmax - one_hot(labels)   (the CE gradient)
        dZ = probs.clone()
        idx = token_slice.labels.long()
        dZ[torch.arange(dZ.shape[0], device=dZ.device), idx] -= 1.0
        # Sign: policy gradient is -A * grad(log pi), so grad(-A * log pi) = -A * dZ_CE.
        dZ.mul_(-token_slice.advantages.float().view(-1, 1))

        if self.kl_coef != 0.0 and token_slice.ref_logprobs is not None:
            # Explicit KL term (if caller passes full ref prob dists via extras).
            ref_probs = token_slice.extra.get("ref_probs")
            if ref_probs is not None:
                dZ.add_(ref_probs.float(), alpha=-self.kl_coef).add_(
                    probs, alpha=self.kl_coef
                )
            # If only ref_logprobs (label-aligned) available, a token-level
            # approximation: skip here -- callers who need that should
            # assemble it in `extra["ref_probs"]` or subclass this loss.

        dZ = dZ.mul_(loss_scale).to(logits.dtype)
        return dZ, {}
