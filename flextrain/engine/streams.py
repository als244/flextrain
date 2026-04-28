"""CUDA stream + event bookkeeping for the training engine.

What this module owns
---------------------
* The four CUDA streams orig creates in ``ActiveModel.__init__``
  (``compute``, ``inbound``, ``outbound``, ``inbound_fwd_context``).
* NVTX naming for those streams (for nsys-profile readability).
* Typed event maps that replace orig's bare dicts.

What this module does NOT own
-----------------------------
* Any device buffers (lives in :mod:`flextrain.engine.buffers`).
* The actual forward / backward compute (lives in
  :mod:`flextrain.nn`).
* The scheduling decisions — those are in
  :mod:`flextrain.engine.active_model`; this module just gives them
  typed handles to the bookkeeping state.

Why typed wrappers over bare ``dict``
--------------------------------------
Orig carries ten event maps in
``active_model.py:113-131``, keyed variously by ``layer_id`` (int),
``slot_idx`` (int), or ``(layer_id, chunk_id)`` (tuple). A lookup
error silently returns ``None`` (dict default), which becomes a hang
at the next ``wait_event`` because CUDA happily waits on a "never
fires" sentinel. Wrapping each map in a small class with a
``record`` / ``clear`` / ``wait_from`` method surfaces typos at
definition time and documents the lifecycle.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from typing import Iterator

import torch


# ---------------------------------------------------------------------------
# NVTX stream naming. Best-effort; if libnvToolsExt isn't present we just
# skip the naming and keep running.
# ---------------------------------------------------------------------------


try:
    _NVTX_LIB: ctypes.CDLL | None = ctypes.CDLL("libnvToolsExt.so")
except OSError:
    _NVTX_LIB = None


# Declare ``nvtxNameCuStreamA(CUstream, const char*)`` argtypes explicitly.
# Without this, ctypes defaults to ``c_int`` (32-bit) for the stream handle
# and truncates the pointer-sized ``cudaStream_t`` value torch hands us via
# ``stream.cuda_stream``. The truncated pointer dereferences to garbage
# inside NVTX -> SIGSEGV. The bug is silent without a profiler attached
# because NVTX is a no-op stub in that case (it never dereferences); nsys /
# nv-nsight loads the real NVTX library which actually walks the handle.
if _NVTX_LIB is not None:
    try:
        _NVTX_LIB.nvtxNameCuStreamA.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        _NVTX_LIB.nvtxNameCuStreamA.restype = None
    except AttributeError:  # pragma: no cover
        # Symbol missing -> NVTX too old. Disable naming entirely.
        _NVTX_LIB = None


def _name_stream(stream: torch.cuda.Stream, name: str) -> None:
    if _NVTX_LIB is None:
        return
    try:
        _NVTX_LIB.nvtxNameCuStreamA(
            ctypes.c_void_p(stream.cuda_stream), name.encode("utf-8")
        )
    except Exception:  # pragma: no cover
        # NVTX naming is purely cosmetic; never let a failure break training.
        pass


# ---------------------------------------------------------------------------
# StreamBundle: the four (or five) streams one engine instance uses.
# ---------------------------------------------------------------------------


@dataclass
class StreamBundle:
    """All CUDA streams the engine needs for one device.

    Mirrors ``orig/active_model.py:100-111`` one-for-one. The
    ``secondary_compute`` stream is optional — orig creates it only
    when any layer is MoE (``orig/train.py:192-198``). In v2 we make
    it optional and let the caller pass it in; dense-only training
    leaves it ``None``.
    """

    compute: torch.cuda.Stream
    inbound: torch.cuda.Stream
    outbound: torch.cuda.Stream
    inbound_fwd_context: torch.cuda.Stream
    secondary_compute: torch.cuda.Stream | None = None

    @classmethod
    def create(
        cls, device: torch.device | str, *, with_secondary: bool = False
    ) -> "StreamBundle":
        """Allocate and NVTX-name the four primary streams."""
        compute = torch.cuda.Stream(device=device)
        inbound = torch.cuda.Stream(device=device)
        outbound = torch.cuda.Stream(device=device)
        inbound_fwd_context = torch.cuda.Stream(device=device)
        secondary = (
            torch.cuda.Stream(device=device) if with_secondary else None
        )

        _name_stream(compute, "Compute")
        _name_stream(inbound, "Inbound")
        _name_stream(outbound, "Outbound")
        _name_stream(inbound_fwd_context, "Inbound Fwd Context")
        if secondary is not None:
            _name_stream(secondary, "Secondary Compute")

        return cls(
            compute=compute,
            inbound=inbound,
            outbound=outbound,
            inbound_fwd_context=inbound_fwd_context,
            secondary_compute=secondary,
        )

    def synchronize_all(self) -> None:
        """Block the CPU until every stream in this bundle is idle.
        Used at fwd_bwd end and during :meth:`ActiveModel.step`'s
        cleanup."""
        self.compute.synchronize()
        self.inbound.synchronize()
        self.outbound.synchronize()
        self.inbound_fwd_context.synchronize()
        if self.secondary_compute is not None:
            self.secondary_compute.synchronize()


# ---------------------------------------------------------------------------
# Typed event maps. All three share the same "key -> Event" shape but
# differ in key type, so we parametrize with a thin base.
# ---------------------------------------------------------------------------


@dataclass
class _BaseEventMap:
    """``key -> torch.cuda.Event`` with a few conveniences.

    Subclasses narrow the key type. Keeping this base class simple
    avoids generic-Protocol contortions while still giving us one
    place to fix lifecycle bugs.
    """

    events: dict = field(default_factory=dict)

    def clear(self) -> None:
        self.events.clear()

    def __contains__(self, key) -> bool:
        return key in self.events

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator:
        return iter(self.events)

    def get(self, key):
        return self.events.get(key)

    def record_on(self, key, stream: torch.cuda.Stream) -> torch.cuda.Event:
        """Record a fresh event on ``stream`` and store under ``key``.
        Overwrites any prior event for this key — engine is responsible
        for calling ``clear()`` or reassigning at round boundaries.

        The recorded event is returned so callers can use it locally
        without a second dict lookup.
        """
        ev = stream.record_event()
        self.events[key] = ev
        return ev

    def wait_on(self, key, stream: torch.cuda.Stream) -> None:
        """Make ``stream`` wait on the event for ``key``. Raises
        ``KeyError`` if the event hasn't been recorded yet (unlike
        orig's dict ``.get(None)`` silently leading to a hang).
        """
        try:
            ev = self.events[key]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__}: no event recorded for key {key!r}. "
                "The scheduler is waiting on something that was never fired."
            ) from None
        if ev is None:
            # Orig's convention: a value of None marks "already consumed,
            # no wait needed" (see active_model.py:1308,1329). Preserve.
            return
        stream.wait_event(ev)

    def mark_consumed(self, key) -> None:
        """Set the entry to ``None`` to indicate that the event has
        been waited on and should not be re-waited. Mirrors orig's
        ``self.weight_inbound_events[k] = None`` pattern at
        ``active_model.py:1329``."""
        self.events[key] = None


class LayerEventMap(_BaseEventMap):
    """``layer_id: int -> Event``.

    Used for per-layer resources: param fetches, grad fetches, opt-state
    fetches.
    """


class SlotEventMap(_BaseEventMap):
    """``slot_idx: int -> Event``.

    Used for the activation ring — one event per ring slot, fires when
    the slot is safe for the compute stream to overwrite.
    """


class LayerChunkEventMap(_BaseEventMap):
    """``(layer_id, chunk_id): tuple[int, int] -> Event``.

    Used for per-(layer, chunk) state: home-slot availability, inbound
    activation prefetch completion.
    """


# ---------------------------------------------------------------------------
# EventBook: the engine's bag of event maps, all in one place.
# ---------------------------------------------------------------------------


@dataclass
class EventBook:
    """Every event map ``ActiveModel`` needs, named.

    Replaces the ten bare-dict event fields at
    ``orig/active_model.py:113-131``. The :meth:`clear_per_round`
    method exists because orig resets a subset of these at the top of
    every gradient-accumulation round (``active_model.py:1180-1183``)
    and forgetting any would cause cross-round stale waits.
    """

    weight_inbound: LayerEventMap = field(default_factory=LayerEventMap)
    grad_weight_inbound: LayerEventMap = field(default_factory=LayerEventMap)
    opt_inbound: LayerEventMap = field(default_factory=LayerEventMap)

    act_slot_ready: SlotEventMap = field(default_factory=SlotEventMap)

    home_act_slot_available: LayerChunkEventMap = field(
        default_factory=LayerChunkEventMap
    )
    inbound_act_slot_ready: LayerChunkEventMap = field(
        default_factory=LayerChunkEventMap
    )

    # dev_act_slot_mapping — not an event map, but the engine routes it
    # alongside the event bookkeeping (it names which GPU ring slot
    # currently holds a given (layer, chunk)'s activations). Keeping
    # here keeps the lifecycle in one place.
    dev_act_slot_mapping: dict = field(default_factory=dict)

    def clear_per_round(self) -> None:
        """Reset per-round event state. Matches
        ``active_model.py:1180-1183``."""
        self.home_act_slot_available.clear()
        self.inbound_act_slot_ready.clear()
        self.dev_act_slot_mapping.clear()
        self.grad_weight_inbound.clear()


__all__ = [
    "EventBook",
    "LayerChunkEventMap",
    "LayerEventMap",
    "SlotEventMap",
    "StreamBundle",
]
