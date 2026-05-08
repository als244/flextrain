"""Execution engine: orchestrator + stream/buffer/schedule bookkeeping.

The engine is a port of ``orig/active_model.py`` (1852 LOC) split into four
focused modules:

* :mod:`flextrain.engine.buffers`   -- GPU / host buffer lifecycle (params,
                                       grads, activation ring + home buffer,
                                       transitions, KV context windows).
                                       Replaces the tangled buffer setup in
                                       ``active_model.py:37-476``.
* :mod:`flextrain.engine.streams`   -- CUDA stream + event bookkeeping
                                       (inbound / outbound / compute /
                                       secondary compute). Small.
* :mod:`flextrain.engine.schedule`  -- Table 2 (paper) pre/post-action
                                       dispatch. Maps each (chunk, layer,
                                       phase) into "wait on X, trigger Y"
                                       patterns.
* :mod:`flextrain.engine.active_model` -- :class:`ActiveModel`: the outer
                                          fwd/bwd/step orchestrator a
                                          training script invokes.

NOT YET IMPLEMENTED
-------------------
The compute-path refactor (blocks + LlamaBlock + embed + head) needs to land
first -- the engine calls into Layer.forward / Layer.backward, and those
methods are still stubs in Phase 2. Until then this package publishes the
API shapes so downstream code (io/, optim/, tests) has something to import.

Review docs/internal/PLAN.md "Phase 3 -- Engine port" for the landing order.
"""

from .buffers import BufferManager, KVContextWindow, ScratchPool
from .host_memory import (
    HostMemoryBackend,
    LocalPinnedHostBackend,
    UnpinnedHostBackend,
    default_host_backend,
    unregister_all_process_pinned_memory,
)
from .schedule import (
    ChunkPolicy,
    ChunkSeqRef,
    PreparedRound,
    TrainingChunk,
    prepare_training_chunks,
    split_sequences,
)
from .streams import (
    EventBook,
    LayerChunkEventMap,
    LayerEventMap,
    SlotEventMap,
    StreamBundle,
)

__all__ = [
    "BufferManager",
    "ChunkPolicy",
    "ChunkSeqRef",
    "EventBook",
    "HostMemoryBackend",
    "KVContextWindow",
    "LayerChunkEventMap",
    "LayerEventMap",
    "LocalPinnedHostBackend",
    "PreparedRound",
    "ScratchPool",
    "SlotEventMap",
    "StreamBundle",
    "TrainingChunk",
    "UnpinnedHostBackend",
    "default_host_backend",
    "prepare_training_chunks",
    "split_sequences",
    "unregister_all_process_pinned_memory",
]
