"""Host-memory abstraction for the training engine.

Why an abstraction
------------------
The engine's master params, grads, and optimizer state all live in
"host memory" — memory the GPU DMA engine can pull from. The default
binding is local-node RAM pinned via ``cudaHostRegister``, but:

* **Scale-out systems** may want to keep master weights on a remote
  node (over RDMA, NVLink-over-Fabric, CXL, ...) when local host RAM
  is smaller than the model's master state.
* **Checkpointing on exotic storage** (persistent memory, NVMe-over-
  fabrics) looks similar to the engine: "give me a tensor the GPU
  can copy to/from at PCIe-class bandwidth."
* **Testing** benefits from a meta-backend that never actually
  allocates.

Rather than hardcode ``cudaHostRegister(torch.zeros(...))`` at every
allocation site, we go through a :class:`HostMemoryBackend`. The
engine calls :meth:`allocate_tensor`; the backend returns a
``torch.Tensor`` that behaves like pinned host memory to the GPU DMA.

Default: :class:`LocalPinnedHostBackend`. This is a drop-in for
``orig/active_model.py:270-273``. Remote / persistent backends would
subclass and override the two hooks.

What the abstraction does NOT solve
------------------------------------
* It does NOT pretend remote memory has the same latency as local
  pinned memory. A remote backend would carry its own bandwidth
  characteristics and the engine's prefetch pipeline would need to
  be retuned. We're only making the *allocation surface* uniform.
* It does NOT handle migration between backends (e.g. live-move from
  local to remote under memory pressure). That's a scheduler policy
  question, not an allocator question.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from typing import Protocol, runtime_checkable

import torch


# ---------------------------------------------------------------------------
# Low-level: cudaHostRegister/Unregister wrappers. Used by the local
# backend; remote backends substitute their own registration API.
# ---------------------------------------------------------------------------


try:
    _cudart: ctypes.CDLL | None = ctypes.CDLL(
        ctypes.util.find_library("cudart") or "libcudart.so"
    )
except OSError:  # pragma: no cover
    _cudart = None


# CUDA error codes we treat as benign on registration:
#   712 = cudaErrorHostMemoryAlreadyRegistered.
# Can happen when two BufferManagers (or two test modules) allocate
# separate torch tensors that happen to land in memory the driver
# already has pinned from a prior (still-live) allocation. The
# effective behavior is identical to "already registered" — the new
# tensor can be used for DMA — so we silently accept.
_CUDA_ERR_ALREADY_REGISTERED = 712


def _cuda_host_register(tensor: torch.Tensor) -> bool:
    """Register ``tensor``'s storage for GPU DMA. Returns True on a
    fresh registration, False if the driver reports the range is
    already mapped.
    """
    if _cudart is None or tensor.numel() == 0:
        return False
    nbytes = tensor.numel() * tensor.element_size()
    ret = _cudart.cudaHostRegister(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_size_t(int(nbytes)),
        ctypes.c_uint(0),
    )
    if ret == _CUDA_ERR_ALREADY_REGISTERED:
        # Clear the sticky CUDA error that cudaGetLastError would
        # otherwise surface on the next CUDA API call (torch checks
        # this after every op). Failing to clear here causes later
        # torch.zeros(..., device='cuda') to raise with the same
        # error code.
        _cudart.cudaGetLastError()
        return False
    if ret != 0:
        raise RuntimeError(
            f"cudaHostRegister failed (ret={ret}) for tensor of "
            f"{nbytes} bytes at {tensor.data_ptr():#x}"
        )
    return True


def _cuda_host_unregister(tensor: torch.Tensor) -> None:
    if _cudart is None or tensor.numel() == 0:
        return
    _cudart.cudaHostUnregister(ctypes.c_void_p(tensor.data_ptr()))


# ---------------------------------------------------------------------------
# Backend Protocol.
# ---------------------------------------------------------------------------


@runtime_checkable
class HostMemoryBackend(Protocol):
    """Host-memory allocator surface the engine consumes.

    Implementations return ``torch.Tensor`` objects the engine can
    bind to host-side dicts and issue ``tensor.copy_(..., non_blocking=
    True)`` against.

    Lifecycle
    ---------
    * :meth:`allocate_tensor(shape, dtype)`: return a new tensor (zero-
      initialized on local; undefined on remote — callers should zero
      if they need it).
    * :meth:`release(tensor)`: free. The default local backend
      unregisters with cudaHostUnregister here.
    * :meth:`available_bytes()` — optional (for sizing heuristics).

    Implementations SHOULD guarantee:
    * The returned tensor's storage is stable for its lifetime (the
      backend won't migrate it without warning).
    * ``tensor.copy_(dev_tensor, non_blocking=True)`` transfers at
      PCIe-class bandwidth (i.e. registered somehow).
    """

    name: str

    def allocate_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Allocate a host tensor of ``shape`` and ``dtype``."""
        ...

    def release(self, tensor: torch.Tensor) -> None:
        """Free / unregister ``tensor``. Safe to call on the same
        tensor more than once (second call is a no-op)."""
        ...


# ---------------------------------------------------------------------------
# Default backend: local-node pinned RAM.
# ---------------------------------------------------------------------------


# Process-wide set of cudaHostRegister'd data pointers. Shared across
# all LocalPinnedHostBackend instances so two backends in one process
# (e.g. across two test modules) don't double-register the same
# torch-caching-allocator-recycled address.
_PROCESS_REGISTERED_PTRS: set[int] = set()


class LocalPinnedHostBackend:
    """Default: allocate plain ``torch.zeros`` in host RAM, then
    register the buffer with the CUDA driver for efficient DMA.

    Why not ``pin_memory=True``?
    ----------------------------
    PyTorch's ``pin_memory=True`` uses ``cudaHostAlloc``, which only
    reserves in powers of 2. A 100 GiB training state budget would
    get rounded up to 128 GiB — several GB wasted for nothing. Orig's
    ``cudaHostRegister`` trick (``active_model.py:266-273``) avoids
    this. We preserve it.

    Registration tracking
    ---------------------
    The set of registered data_ptrs is **process-wide**
    (:data:`_PROCESS_REGISTERED_PTRS`), not per-instance. This
    matters because torch's caching allocator recycles host storage
    across ``torch.zeros`` calls; a second backend allocating a
    tensor that lands on storage the first backend already pinned
    would otherwise hit
    ``cudaErrorHostMemoryAlreadyRegistered`` (712) AND leave the
    CUDA driver in a state where subsequent ``copy_(..., non_blocking
    =True)`` fails with ``cudaErrorInvalidValue``. Tracking at
    process scope avoids the re-register entirely.
    """

    name = "local_pinned"

    def __init__(self) -> None:
        self._owned: set[int] = set()  # ptrs this instance is responsible for

    def allocate_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        t = torch.zeros(shape, dtype=dtype)
        ptr = t.data_ptr()
        if ptr in _PROCESS_REGISTERED_PTRS:
            # Already pinned by a prior allocation (ours or another
            # backend's). Leave the CUDA pin in place and just use the
            # new tensor — it shares the same storage, so DMA works.
            return t
        if _cuda_host_register(t):
            _PROCESS_REGISTERED_PTRS.add(ptr)
            self._owned.add(ptr)
        return t

    def release(self, tensor: torch.Tensor) -> None:
        ptr = tensor.data_ptr()
        if ptr in self._owned:
            _cuda_host_unregister(tensor)
            _PROCESS_REGISTERED_PTRS.discard(ptr)
            self._owned.discard(ptr)


def unregister_all_process_pinned_memory() -> None:
    """Drain every cudaHostRegister'd pointer tracked by any
    :class:`LocalPinnedHostBackend` in this process.

    Useful between test-runs: `BufferManager.destroy()` only touches
    tensors the backend was aware of, but tests that construct and
    drop buffer managers without calling ``destroy()`` leak
    registrations. This is a process-wide cleanup.

    Callers must pass a ``ptr -> tensor`` way to reach each tensor;
    since we only have ptrs, we emit an ``cudaHostUnregister`` on the
    raw pointer. That's safe: cudaHostUnregister only requires the
    host pointer, not a size.
    """
    if _cudart is None:
        return
    for ptr in list(_PROCESS_REGISTERED_PTRS):
        _cudart.cudaHostUnregister(ctypes.c_void_p(ptr))
        _cudart.cudaGetLastError()
    _PROCESS_REGISTERED_PTRS.clear()


# ---------------------------------------------------------------------------
# Null backend: allocate but don't register. Useful for tests on
# machines without CUDA or when the engine is in a smoke-test mode.
# ---------------------------------------------------------------------------


class UnpinnedHostBackend:
    """Unregistered local host memory — for tests. ``copy_()`` will
    still work but synchronously (no DMA queuing benefit)."""

    name = "unpinned"

    def allocate_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype)

    def release(self, tensor: torch.Tensor) -> None:  # noqa: ARG002
        pass


# ---------------------------------------------------------------------------
# Convenience default.
# ---------------------------------------------------------------------------


def default_host_backend() -> HostMemoryBackend:
    """Return a backend that matches orig behavior (local pinned)."""
    return LocalPinnedHostBackend()


__all__ = [
    "HostMemoryBackend",
    "LocalPinnedHostBackend",
    "UnpinnedHostBackend",
    "default_host_backend",
]
