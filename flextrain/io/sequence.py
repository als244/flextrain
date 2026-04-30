"""The :class:`Sequence` object the engine consumes.

The engine's scheduler (``flextrain/engine/schedule.py``) duck-types on
this shape: a ``tokens`` int64 tensor, a ``targets`` int64 tensor, a
``per_token_loss`` cpu fp32 buffer, and ``__len__``. This module
makes it explicit so data-source adapters in ``flextrain/io/streams/``
can produce the same thing.

Ported (with minor cleanup) from ``orig/sequence.py``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import torch


class Sequence:
    """One training sequence.

    Attributes
    ----------
    tokens
        ``(N,)`` int64 token ids on CPU.
    targets
        ``(N,)`` int64 target token ids on CPU. For standard next-token
        prediction this is ``tokens`` shifted by 1.
    per_token_loss
        ``(N,)`` fp32 on CPU (pinned when possible) where the engine
        writes per-token loss after the head runs. Caller reads this
        post-step for logging / metrics.
    loss_mask
        Optional ``(N,)`` bool on CPU. ``False`` = don't train on this
        position (e.g. prompt tokens in an SFT example). Engine passes
        to :class:`~flextrain.nn.loss.CrossEntropyLoss` via
        :class:`~flextrain.nn.loss.TokenContext`.
    advantages, ref_logprobs
        Optional per-token RL signals (GRPO, DPO, ...). Forwarded to
        the pluggable loss fn.
    seq_id
        Diagnostic identifier (defaults to a uuid).
    loss_function
        Optional tag the caller can attach to drive per-sequence loss
        dispatching.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        loss_mask: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        ref_logprobs: torch.Tensor | None = None,
        seq_id: Any = None,
        loss_function: Any = None,
        pin_per_token_loss: bool = True,
    ) -> None:
        self.seq_id = seq_id if seq_id is not None else str(uuid.uuid4())
        self.tokens = tokens
        # Default to next-token shift if no explicit targets given.
        self.targets = (
            targets if targets is not None else torch.roll(tokens, -1)
        )
        self.loss_mask = loss_mask
        self.advantages = advantages
        self.ref_logprobs = ref_logprobs
        self.loss_function = loss_function
        # ``active_token_count`` = number of positions that contribute
        # to the loss = positions where ``targets != -100``. Cached
        # once at construction (host-side, free for callers to access).
        # Callers building a step's batch pass
        # ``loss_scale_factor = 1.0 / sum(s.active_token_count for s
        # in seqs)`` to the engine so dZ is scaled to the
        # ``mean-over-active-tokens`` convention (matches HF /
        # PyTorch ``CrossEntropyLoss(ignore_index=-100)``).
        self.active_token_count = int((self.targets != -100).sum().item())
        try:
            self.per_token_loss = torch.zeros(
                len(tokens), dtype=torch.float32,
                device="cpu", pin_memory=pin_per_token_loss,
            )
        except RuntimeError:
            # pin_memory may fail in test contexts without CUDA.
            self.per_token_loss = torch.zeros(
                len(tokens), dtype=torch.float32, device="cpu"
            )
        self.create_time = time.time()
        self.start_train_time: float | None = None
        self.complete_train_time: float | None = None

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.tokens[index]
