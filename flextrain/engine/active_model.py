"""ActiveModel: AdaWS training orchestrator.

Ports ``orig/active_model.py`` (1852 LOC) into a typed, slot-aware
engine with pluggable loss functions, heterogeneous backbones, and a
swappable :class:`HostMemoryBackend`.

Module layout
-------------
The engine's internals are four modules:

* :mod:`flextrain.engine.streams`   — StreamBundle + typed EventBook.
* :mod:`flextrain.engine.buffers`   — BufferManager (params, grads,
                                       opt state, activation ring,
                                       host act buffer, KV context).
* :mod:`flextrain.engine.schedule`  — split_sequences +
                                       prepare_training_chunks
                                       (+ ChunkPolicy for non-causal).
* :mod:`flextrain.engine.active_model` — THIS module. :class:`ActiveModel`
                                          owns :meth:`fwd_bwd` and
                                          :meth:`step` and stitches the
                                          other three together.

Semantics preserved from orig
-----------------------------
* Per-(chunk, layer) save-level planning via DP solver. Paper §3.4.
* Forward traversal: ``for layer: for chunk``.
* Backward traversal: ``for layer (reverse): for seq_group (reverse):
  for chunk-within-group (reverse)``.
* Activation ring rotates forward during fwd, reverse during bwd.
* KV context window is refreshed during bwd from the PRIOR seq group's
  saved K/V (this is the trickiest bit — see :meth:`_update_fwd_context`).
* Last ``n_gpu_act_slots`` chunks keep their activations on the GPU
  ring; others get sent home and prefetched back in reverse order
  during backward.
* Gradient accumulation across rounds: first round zeros grads, later
  rounds accumulate (``addmm(beta=1.0)`` inside layer backward paths).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping as MutableMappingT, Sequence as _Seq

import numpy as np
import torch

from flextrain.core.layer import (
    ChunkMeta,
    InputLayer,
    Layer,
    LayerContext,
    LossStats,
    MoEChunkConfig,
    OutputLayer,
)
from flextrain.core.save_level import (
    HardwareCost,
    SaveLevel,
    SaveLevelPlan,
    build_dp_tables,
    plan_from_solution,
)
from flextrain.core.activation_schema import (
    ActivationSlot,
    send_home,
    fetch_home,
)
from flextrain.core.working_set import WorkingSetConfig
from flextrain.nn.loss import CrossEntropyLoss, LossFn, TokenContext
from flextrain.optim.base import Optimizer

from .buffers import BufferManager, ScratchPool
from .host_memory import HostMemoryBackend
from .schedule import (
    ChunkPolicy,
    PreparedRound,
    TrainingChunk,
    prepare_training_chunks,
    split_sequences,
)
from .streams import EventBook, StreamBundle


# ---------------------------------------------------------------------------
# StepStats: what fwd_bwd returns to the trainer loop.
# ---------------------------------------------------------------------------


@dataclass
class StepStats:
    """Summary statistics for one full fwd_bwd call (= one optimization
    step's worth of gradient accumulation).

    Attributes
    ----------
    total_tokens
        Sum across all rounds' chunks. For logging.
    total_loss
        Sum of per-token losses across every chunk, every round
        (fp32). Caller typically divides by total_tokens to get mean.
    rounds
        Number of gradient-accumulation rounds the input was split
        into.
    """

    total_tokens: int
    total_loss: float
    rounds: int


# ---------------------------------------------------------------------------
# ActiveModel.
# ---------------------------------------------------------------------------


@dataclass
class ActiveModel:
    """AdaWS training orchestrator.

    Construction
    ------------
    embed
        :class:`InputLayer` (token embedding). Must have a ``param_spec``
        with one tensor (``w_tok_embeddings``).
    backbone
        Sequence of :class:`Layer`, ordered by depth. Each must set
        ``layer_id = position`` (0-indexed) for event-map keys.
    head
        :class:`OutputLayer` (LM head).
    optimizer
        :class:`~flextrain.optim.Optimizer` (AdamW, Muon, ...).
    working_set
        :class:`WorkingSetConfig` from
        :func:`~flextrain.core.working_set.determine_working_set_config`.
    hw_cost
        :class:`HardwareCost` for DP-solver time conversions.
    dims
        Model-wide dims mapping (``d_model``, ``n_heads``, ``n_kv_heads``,
        ``head_dim``, ``expert_dim``, ``vocab_size``, ...). Passed to
        :class:`TensorSpec`/:class:`ActivationField` shape functions.
    chunk_policy
        :class:`ChunkPolicy.CAUSAL` or ``NON_CAUSAL``. Defaults to
        CAUSAL (every architecture we currently support).
    force_saved_act_level
        Optional debug override: force every (layer, chunk) to this
        tier. Defaults to ``None`` (use the DP solver).
    host_backend
        Optional :class:`HostMemoryBackend` (default: local pinned
        RAM). Swap in a remote / persistent backend for scale-out.
    device
        CUDA device string.
    """

    embed: InputLayer
    backbone: _Seq[Layer]
    head: OutputLayer
    optimizer: Optimizer
    working_set: WorkingSetConfig
    hw_cost: HardwareCost
    dims: Mapping[str, Any]
    chunk_policy: ChunkPolicy = ChunkPolicy.CAUSAL
    force_saved_act_level: int | None = None
    host_backend: HostMemoryBackend | None = None
    device: str | torch.device = "cuda:0"

    # Post-init fields.
    buffers: BufferManager = field(init=False)
    streams: StreamBundle = field(init=False)
    events: EventBook = field(init=False)
    scratch: ScratchPool = field(init=False)
    step_count: int = 0
    _zero_grad: bool = True  # True at round 0 of each step; False after first round.

    def __post_init__(self) -> None:
        device = torch.device(self.device)

        # Build buffer manager from layer specs.
        layer_specs = [layer.param_spec for layer in self.backbone]
        layer_schemas = [layer.schema for layer in self.backbone]

        self.buffers = BufferManager(
            working_set=self.working_set,
            dims=self.dims,
            layer_param_specs=layer_specs,
            layer_schemas=layer_schemas,
            embed_param_spec=self.embed.param_spec,
            head_param_spec=self.head.param_spec,
            opt_spec=self.optimizer.state_spec,
            device=device,
            host_backend=self.host_backend,
        )

        # MoE layers (future) can pass with_secondary=True; dense defaults.
        with_secondary = any(
            getattr(layer, "uses_secondary_stream", False)
            for layer in self.backbone
        )
        self.streams = StreamBundle.create(
            device=device, with_secondary=with_secondary
        )
        self.events = EventBook()
        self.scratch = ScratchPool(device)

        # Initial param prefetch into the first N_P GPU slots.
        self._initial_param_prefetch()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _initial_param_prefetch(self) -> None:
        """Kick off the host->device copy for the first ``N_P`` layers'
        params. After this call ``self.events.weight_inbound`` has an
        event for every layer resident in the param ring.

        Mirrors ``orig/active_model.py:247-252``.
        """
        with torch.cuda.stream(self.streams.inbound):
            # Embed params (resident, one-shot copy).
            if self.embed is not None:
                for name, dev_t in self.buffers.gpu_embed_params.items():
                    dev_t.copy_(
                        self.buffers.host_embed_params[name], non_blocking=True
                    )
            # Head params (resident, one-shot copy).
            if self.head is not None:
                for name, dev_t in self.buffers.gpu_head_params.items():
                    dev_t.copy_(
                        self.buffers.host_head_params[name], non_blocking=True
                    )
            # Backbone: first N_P layers.
            N_P = self.working_set.n_gpu_layers
            for slot_idx in range(min(N_P, len(self.backbone))):
                layer_id = slot_idx  # first N_P layers map to slots 0..N_P-1
                self.buffers.fetch_layer_params(
                    layer_id, slot_idx, non_blocking=True
                )
                self.events.weight_inbound.record_on(
                    layer_id, self.streams.inbound
                )

        # Synchronize once at the end of construction. Follows orig's
        # pattern (active_model.py:281-282).
        torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Public: fwd_bwd + step.
    # ------------------------------------------------------------------

    def fwd_bwd(
        self,
        sequences: _Seq,
        *,
        loss_scale_factor: float | None = None,
        total_tokens_per_step: int | None = None,
        loss_fn: LossFn | None = None,
        verbose: bool = False,
    ) -> StepStats:
        """One optimization step's fwd + bwd (may span multiple gradient
        accumulation rounds).

        Parameters
        ----------
        sequences
            Iterable of :class:`Sequence` (duck-typed: needs ``tokens``,
            ``targets``, ``per_token_loss``, ``__len__``).
        loss_scale_factor
            Folded into the head's grad matmul. Callers typically pass
            ``1.0 / total_tokens_per_step`` to get "sum ≡ mean" grads.
            Defaults to ``1.0`` (no scaling).
        total_tokens_per_step
            Forwarded to backbone layer ``backward()`` for any
            per-layer grad scaling that needs it (e.g. MoE load-balance
            loss). Not currently consumed by Llama layers.
        loss_fn
            Optional :class:`LossFn` (default :class:`CrossEntropyLoss`).
        verbose
            Log round / save-level / loss per round.

        Returns
        -------
        :class:`StepStats`. The caller reads ``total_loss`` /
        ``total_tokens`` for per-step averaging.
        """
        if loss_scale_factor is None:
            loss_scale_factor = 1.0
        if loss_fn is None:
            loss_fn = CrossEntropyLoss()

        rounds, _ = split_sequences(
            sequences,
            target_round_tokens=self.working_set.target_round_tokens,
            max_total_round_tokens=self.working_set.max_total_round_tokens,
            max_chunk_size=self.working_set.max_chunk_size,
            max_training_chunks=self.working_set.max_training_chunks,
            policy=self.chunk_policy,
        )

        total_loss = 0.0
        total_tokens = 0

        for round_idx, round_seqs in enumerate(rounds):
            prepared = prepare_training_chunks(
                round_seqs,
                max_chunk_size=self.working_set.max_chunk_size,
                device=self.device,
                policy=self.chunk_policy,
            )
            # Allocate per-chunk MoE scratch (no-op if no MoE layers).
            self._allocate_moe_chunk_scratch(prepared)

            # Clear per-round event state so stale events from prior
            # rounds don't confuse the waits.
            self.events.clear_per_round()


            plan = self._plan_save_levels(prepared)
            if verbose:
                self._log_round(round_idx, len(rounds), prepared, plan)

            # Sync at top of round so last round's tail DMAs are done
            # before we start overwriting state. Matches orig:1203.
            self.streams.compute.synchronize()

            self._setup_round(prepared, plan)
            self._forward_pass(prepared, plan)
            round_loss, round_tokens = self._head_pass(
                prepared,
                loss_scale=loss_scale_factor,
                loss_fn=loss_fn,
            )
            self._backward_pass(
                prepared, plan, total_tokens_per_step=total_tokens_per_step
            )
            self._embed_backward(prepared)

            total_loss += round_loss
            total_tokens += round_tokens
            self._zero_grad = False  # grads accumulated

        # Make sure any trailing grad-offload DMAs are done before
        # returning; the caller will read per-sequence loss or call
        # step().
        self.streams.compute.synchronize()
        self.streams.outbound.synchronize()

        return StepStats(
            total_tokens=total_tokens,
            total_loss=total_loss,
            rounds=len(rounds),
        )

    def step(self, step_num: int | None = None) -> int:
        """Apply the optimizer update to every parameter.

        Semantics mirror ``orig/active_model.py:1632-1850`` but the
        per-layer update is driven by the single
        :class:`~flextrain.optim.base.Optimizer` object (no per-layer
        step methods). The opt-state ring is staged in the GPU
        activation buffer (paper §3.3) via
        :meth:`BufferManager.swap_to_optimizer_state`.

        Parameters
        ----------
        step_num
            Optimizer step counter, forwarded to the optimizer (AdamW
            needs it for bias correction; Muon ignores). Defaults to
            ``self.step_count + 1`` (pre-increment — matches orig's
            ``opt_hyperparams['step_num'] += 1`` BEFORE calling step,
            see ``orig/train.py:267``).

        Returns
        -------
        0 on success; nonzero propagates an optimizer failure (NaN /
        Inf grads).
        """
        if step_num is None:
            step_num = self.step_count + 1

        # No matter how this step goes (success or early return on NaN),
        # the next fwd_bwd must start fresh. Set the flag here so a
        # short-circuit doesn't leave grads stale.
        self._zero_grad = True

        # ---- 1. Embed + head step (resident on GPU; no ring rotation) ----
        if self.embed is not None:
            ret = self._step_resident(
                self.embed.param_spec,
                self.buffers.gpu_embed_params,
                self.buffers.gpu_embed_grads,
                host_master=self.buffers.host_embed_params,
                host_opt=self.buffers.host_embed_opt.host,
                step_num=step_num,
                mirror_host=True,
            )
            if ret:
                return ret
        if self.head is not None:
            ret = self._step_resident(
                self.head.param_spec,
                self.buffers.gpu_head_params,
                self.buffers.gpu_head_grads,
                host_master=self.buffers.host_head_params,
                host_opt=self.buffers.host_head_opt.host,
                step_num=step_num,
                mirror_host=True,
            )
            if ret:
                return ret

        # ---- 2. Swap GPU activation ring into opt-state ring ----
        N_O = self.working_set.n_gpu_opt_layers
        self.buffers.swap_to_optimizer_state(n_gpu_opt_layers=N_O)

        # ---- 3. Initial opt-state prefetch for first N_O layers ----
        #     + assumption: first N_P layers' weights and N_G layers'
        #     grads are still resident from the end of the last
        #     fwd_bwd (orig:1697-1713).
        # In v2 we re-prefetch to be robust: first N_P param slots get
        # layers 0..N_P-1; first N_G grad slots get layers 0..N_G-1;
        # first N_O opt slots get layers 0..N_O-1.
        num_layers = len(self.backbone)
        N_P = self.working_set.n_gpu_layers
        N_G = self.working_set.n_gpu_grads

        self.events.weight_inbound.clear()
        self.events.grad_weight_inbound.clear()
        self.events.opt_inbound.clear()

        with torch.cuda.stream(self.streams.inbound):
            for i in range(min(N_P, num_layers)):
                layer_id = self.backbone[i].layer_id
                self.buffers.fetch_layer_params(
                    layer_id, i, non_blocking=True
                )
                self.events.weight_inbound.record_on(
                    layer_id, self.streams.inbound
                )
            for i in range(min(N_G, num_layers)):
                layer_id = self.backbone[i].layer_id
                self.buffers.fetch_layer_grads(
                    layer_id, i, non_blocking=True
                )
                self.events.grad_weight_inbound.record_on(
                    layer_id, self.streams.inbound
                )
            for i in range(min(N_O, num_layers)):
                layer_id = self.backbone[i].layer_id
                self.buffers.fetch_layer_opt(
                    layer_id, i, non_blocking=True
                )
                self.events.opt_inbound.record_on(
                    layer_id, self.streams.inbound
                )

        # ---- 4. Per-layer optimizer update + ring rotation ----
        cur_w = 0
        cur_g = 0
        cur_o = 0

        for k_ind in range(num_layers):
            layer = self.backbone[k_ind]
            lid = layer.layer_id

            # Wait on all three inbound events.
            self.events.weight_inbound.wait_on(lid, self.streams.compute)
            self.events.grad_weight_inbound.wait_on(lid, self.streams.compute)
            self.events.opt_inbound.wait_on(lid, self.streams.compute)

            weights = self.buffers.gpu_param_slot(cur_w, layer.param_spec)
            grads = self.buffers.gpu_grad_slot(cur_g, layer.param_spec)
            opt_state = self.buffers.gpu_opt_slot(cur_o, lid)

            with torch.cuda.stream(self.streams.compute):
                ret = self.optimizer.step(
                    layer.param_spec,
                    weights,
                    grads,
                    opt_state,
                    step_num=step_num,
                )
            if ret:
                print(f"[FlexTrain] optimizer step failed for layer {lid}")
                # Restore the activation ring before returning so the
                # next fwd_bwd doesn't see stale opt-state views.
                self.buffers.restore_activation_ring()
                return ret

            # Mirror updated weights + opt state back to host on outbound.
            self.streams.outbound.wait_stream(self.streams.compute)
            with torch.cuda.stream(self.streams.outbound):
                self.buffers.offload_layer_params(
                    lid, cur_w, non_blocking=True
                )
                self.buffers.offload_layer_opt(
                    lid, cur_o, non_blocking=True
                )

            # Prefetch the next-unstaged layer of each resource on inbound.
            # CRUCIAL: the fetches must run INSIDE ``with
            # torch.cuda.stream(inbound)`` — otherwise ``copy_`` uses
            # the default stream, breaks the wait_stream(outbound) sync,
            # and races with the still-in-flight offload of the same
            # ring slot (offload reads stale data).
            self.streams.inbound.wait_stream(self.streams.outbound)
            with torch.cuda.stream(self.streams.inbound):
                next_w = k_ind + N_P
                if next_w < num_layers:
                    next_lid = self.backbone[next_w].layer_id
                    self.buffers.fetch_layer_params(
                        next_lid, cur_w, non_blocking=True
                    )
                    self.events.weight_inbound.record_on(
                        next_lid, self.streams.inbound
                    )
                    self.events.weight_inbound.mark_consumed(lid)

                next_g = k_ind + N_G
                if next_g < num_layers:
                    next_lid = self.backbone[next_g].layer_id
                    self.buffers.fetch_layer_grads(
                        next_lid, cur_g, non_blocking=True
                    )
                    self.events.grad_weight_inbound.record_on(
                        next_lid, self.streams.inbound
                    )
                    self.events.grad_weight_inbound.mark_consumed(lid)

                next_o = k_ind + N_O
                if next_o < num_layers:
                    next_lid = self.backbone[next_o].layer_id
                    self.buffers.fetch_layer_opt(
                        next_lid, cur_o, non_blocking=True
                    )
                    self.events.opt_inbound.record_on(
                        next_lid, self.streams.inbound
                    )
                    self.events.opt_inbound.mark_consumed(lid)

            cur_w = (cur_w + 1) % N_P
            cur_g = (cur_g + 1) % N_G
            cur_o = (cur_o + 1) % N_O

        # ---- 5. Close out opt-ring, restore activation-ring views ----
        self.streams.compute.synchronize()
        self.streams.outbound.synchronize()
        self.streams.inbound.synchronize()
        self.buffers.restore_activation_ring()

        # ---- 6. Reload the first N_P layers' weights into slots 0..N_P-1
        #     so next fwd_bwd's weight_inbound waits find them. ----
        self.events.weight_inbound.clear()
        with torch.cuda.stream(self.streams.inbound):
            for slot_idx in range(min(N_P, num_layers)):
                layer_id = self.backbone[slot_idx].layer_id
                self.buffers.fetch_layer_params(
                    layer_id, slot_idx, non_blocking=True
                )
                self.events.weight_inbound.record_on(
                    layer_id, self.streams.inbound
                )
        self.streams.inbound.synchronize()

        self.step_count += 1
        return 0

    def _step_resident(
        self,
        param_spec,
        weights: Mapping[str, torch.Tensor],
        grads: Mapping[str, torch.Tensor],
        *,
        host_master: Mapping[str, torch.Tensor],
        host_opt: MutableMappingT,
        step_num: int,
        mirror_host: bool,
    ) -> int:
        """Step an embed / head layer whose weights are always resident.
        Their opt-state lives host-side; we stage it to GPU temporarily.

        The activation ring is NOT touched here (embed/head step runs
        BEFORE we swap the ring to opt-state mode).
        """
        # Allocate a transient GPU opt-state dict matching the shape /
        # dtype of the host opt-state. This is tiny (embed + head
        # are small) so we just torch.empty_like on the device side
        # and copy from host. Uses default stream (simplest) and we
        # just sync at the end to be safe.
        dev_opt: dict[str, torch.Tensor] = {}
        for name, host_t in host_opt.items():
            d = torch.empty(
                host_t.shape, dtype=host_t.dtype, device=self.device
            )
            d.copy_(host_t)  # blocking on default stream — tiny tensors
            dev_opt[name] = d

        ret = self.optimizer.step(
            param_spec, weights, grads, dev_opt, step_num=step_num
        )
        if ret:
            return ret

        # Mirror updated state back to host (blocking — these are tiny).
        if mirror_host:
            for name, t in weights.items():
                host_master[name].copy_(t)
            for name, t in dev_opt.items():
                host_opt[name].copy_(t)
        torch.cuda.synchronize()
        return 0

    # ------------------------------------------------------------------
    # HF load/save.
    # ------------------------------------------------------------------

    def load_hf(
        self,
        hf_path: str,
        *,
        arch: Any | None = None,
        strict: bool = True,
    ) -> list[str]:
        """Load HF safetensors into host master-param buffers.

        Parameters
        ----------
        hf_path
            Path to an HF safetensors directory (contains
            ``*.safetensors`` + optional ``model.safetensors.index.json``)
            OR a single ``.safetensors`` file.
        arch
            Optional :class:`~flextrain.io.hf_weights.ArchSpec`. If
            omitted, we try to auto-select by reading ``config.json``
            at ``hf_path`` and looking up the architecture id.
        strict
            If ``True`` (default), raise if any expected HF tensor is
            missing.

        Returns
        -------
        list[str]
            HF tensor names present in the shards that we did NOT
            consume (diagnostic; should be empty for a canonical
            checkpoint).

        After the host copy completes we refresh every GPU slot that
        should be resident for the first fwd_bwd (embed + head +
        first ``N_P`` backbone layers).
        """
        import json
        import os

        from flextrain.io.hf_weights import (
            load_hf_safetensors,
            select_arch,
        )
        # Import per-arch modules so ``select_arch`` finds them.
        from flextrain.io import arch as _arch_pkg  # noqa: F401

        if arch is None:
            cfg_path = os.path.join(hf_path, "config.json")
            if not os.path.isfile(cfg_path):
                raise FileNotFoundError(
                    f"load_hf needs either an explicit ``arch=`` or a "
                    f"config.json at {hf_path!r}"
                )
            with open(cfg_path) as f:
                hf_config = json.load(f)
            arch = select_arch(hf_config)

        num_layers = len(self.backbone)
        dest: dict[tuple[str, str], torch.Tensor] = {}
        for name, t in self.buffers.host_embed_params.items():
            dest[("embed", name)] = t
        for name, t in self.buffers.host_head_params.items():
            dest[("head", name)] = t
        for i, layer_host in enumerate(self.buffers.host_params):
            scope = f"layer_{i}"
            for name, t in layer_host.items():
                dest[(scope, name)] = t

        leftover = load_hf_safetensors(
            hf_path=hf_path,
            arch=arch,
            dest=dest,
            num_layers=num_layers,
            strict=strict,
            device="cpu",
        )

        # Refresh resident GPU slots from the fresh host values.
        self._refresh_gpu_residents()
        return leftover

    def save_hf(
        self,
        out_dir: str,
        *,
        arch: Any | None = None,
        out_filename: str = "model.safetensors",
    ) -> str:
        """Export host master params to a single-shard HF safetensors
        file at ``out_dir/out_filename``. Returns the path written.

        If ``arch`` is ``None``, we inspect the backbone / embed / head
        and try the first registered :class:`ArchSpec`. For one-off
        exports you should pass the matching ``ArchSpec`` explicitly.
        """
        from flextrain.io.hf_weights import (
            export_hf_safetensors,
            _ARCH_REGISTRY,
        )
        from flextrain.io import arch as _arch_pkg  # noqa: F401

        if arch is None:
            if not _ARCH_REGISTRY:
                raise ValueError(
                    "no ArchSpec registered; pass ``arch=`` explicitly"
                )
            # Pick the first (and typically only) registered arch.
            arch = next(iter(_ARCH_REGISTRY.values()))

        num_layers = len(self.backbone)
        src: dict[tuple[str, str], torch.Tensor] = {}
        for name, t in self.buffers.host_embed_params.items():
            src[("embed", name)] = t
        for name, t in self.buffers.host_head_params.items():
            src[("head", name)] = t
        for i, layer_host in enumerate(self.buffers.host_params):
            scope = f"layer_{i}"
            for name, t in layer_host.items():
                src[(scope, name)] = t

        return export_hf_safetensors(
            out_dir=out_dir,
            arch=arch,
            src=src,
            num_layers=num_layers,
            out_filename=out_filename,
        )

    def _refresh_gpu_residents(self) -> None:
        """Re-copy host master params into the GPU resident slots
        (embed, head, first ``N_P`` backbone layers). Call after any
        operation that mutates host params (HF load, direct test
        overwrite) so the GPU state is consistent for the next
        fwd_bwd.
        """
        with torch.cuda.stream(self.streams.inbound):
            for name, dev_t in self.buffers.gpu_embed_params.items():
                dev_t.copy_(
                    self.buffers.host_embed_params[name], non_blocking=True
                )
            for name, dev_t in self.buffers.gpu_head_params.items():
                dev_t.copy_(
                    self.buffers.host_head_params[name], non_blocking=True
                )
            N_P = self.working_set.n_gpu_layers
            for slot_idx in range(min(N_P, len(self.backbone))):
                layer_id = self.backbone[slot_idx].layer_id
                self.buffers.fetch_layer_params(
                    layer_id, slot_idx, non_blocking=True
                )
                self.events.weight_inbound.record_on(
                    layer_id, self.streams.inbound
                )
        # Synchronize before the next fwd_bwd so callers don't need to
        # think about stream ordering.
        self.streams.inbound.synchronize()

    # ==================================================================
    # Internal: round setup (act-slot allocation + embed).
    # ==================================================================

    def _plan_save_levels(self, prepared: PreparedRound) -> SaveLevelPlan:
        """Run the save-level DP for this round.

        Fast path: if ``n_home_act_slots == 0`` (the entire round fits
        on the GPU activation ring), every pair is set to the
        on-device sentinel and we skip the DP. Mirrors ``orig/active_model.py:544-553``.
        """
        num_layers = len(self.backbone)
        num_chunks = len(prepared.chunks)
        total_pairs = num_layers * num_chunks
        n_gpu_act_slots = self.buffers.n_gpu_act_slots
        n_home_act_slots = max(0, total_pairs - n_gpu_act_slots)

        if n_home_act_slots == 0:
            return SaveLevelPlan.all_on_device(
                layer_ids=[layer.layer_id for layer in self.backbone],
                chunk_ids=[c.id for c in prepared.chunks],
            )

        # Run DP via the same C solver orig uses.
        tables = build_dp_tables(
            layers=self.backbone,
            chunk_metas=[c.meta for c in prepared.chunks],
            dims=self.dims,
            hw=self.hw_cost,
        )

        # Headline bookkeeping: min_required = sum of compute_times
        # minus max achievable avoid (= sum of values at each task's
        # own max tier). Matches orig:620-621.
        max_achievable = float(
            tables.values[np.arange(tables.T), tables.max_tier_per_task].sum()
        )
        min_required_ms = float(tables.compute_times.sum() - max_achievable)

        from transmission_scheduler import TransmissionScheduler  # type: ignore[import-not-found]

        solver = TransmissionScheduler()
        _opt_avoid, choices = solver.solve(
            tables.compute_times,
            tables.transfer_durations,
            tables.values,
            n_gpu_act_slots,
        )

        plan = plan_from_solution(
            tables,
            choices,
            n_gpu_act_slots,
            min_required_recompute_time_ms=min_required_ms,
            max_optional_recompute_time_avoided_ms=max_achievable,
        )

        # Optional override: force every pair to a fixed tier (debug).
        if self.force_saved_act_level is not None:
            forced = int(self.force_saved_act_level)
            new_choices: dict[tuple[int, int], SaveLevel] = {}
            for (lid, cid), lvl in plan.choices.items():
                # If pair lives in the on-device tail, keep it; otherwise
                # use the forced tier (clamped to that layer's max).
                if lvl.is_on_device:
                    new_choices[(lid, cid)] = lvl
                else:
                    layer = self.backbone[lid]
                    clamped = min(forced, layer.schema.max_tier)
                    new_choices[(lid, cid)] = SaveLevel(clamped)
            plan = SaveLevelPlan(
                choices=new_choices,
                estimated_recompute_time_ms=plan.estimated_recompute_time_ms,
                estimated_fwd_time_ms=plan.estimated_fwd_time_ms,
            )

        return plan

    def _allocate_moe_chunk_scratch(
        self, prepared: PreparedRound
    ) -> None:
        """Allocate per-chunk MoE scratch tensors and stash them in each
        ``chunk.meta.extra``. No-op if no backbone layer has a
        ``moe_chunk_config`` attribute set.

        Two tensors per (MoE-layer, chunk) pair:

        * ``token_index_mapping``: ``(total_q, top_k)`` int32 on device.
          Written by ``flextrain_moe_sort`` during fwd, read by
          ``flextrain_moe_scatter`` / ``flextrain_moe_router_gate_bwd`` during bwd.
        * ``expert_counts_host``: ``(num_experts,)`` int32 pinned CPU.
          Written by ``flextrain_copy_expert_counts`` during fwd (via a
          kernel that writes directly into mapped host memory), read
          by the expert-loop dispatch at fwd AND bwd.

        Mirrors ``orig/active_model.py::make_chunk_metadata``
        (line 1246-1254).
        """
        # Fast path: no MoE layers anywhere in the backbone.
        moe_layers: list[tuple[int, MoEChunkConfig]] = []
        for layer in self.backbone:
            mc = getattr(layer, "moe_chunk_config", None)
            if mc is not None:
                moe_layers.append((layer.layer_id, mc))
        if not moe_layers:
            return

        for chunk in prepared.chunks:
            token_index_mapping: dict[int, torch.Tensor] = {}
            expert_counts_host: dict[int, torch.Tensor] = {}
            total_q = chunk.meta.total_q
            for lid, mc in moe_layers:
                token_index_mapping[lid] = torch.zeros(
                    total_q, mc.top_k,
                    dtype=torch.int32, device=self.device,
                )
                expert_counts_host[lid] = torch.zeros(
                    mc.num_experts,
                    dtype=torch.int32, device="cpu", pin_memory=True,
                )
            chunk.meta.extra["moe_token_index_mapping"] = token_index_mapping
            chunk.meta.extra["moe_expert_counts_host"] = expert_counts_host

    def _setup_round(
        self, prepared: PreparedRound, plan: SaveLevelPlan
    ) -> None:
        """Allocate host activation slots for this round and run embed
        on every chunk (seeding the transition table).
        """
        # 1. Reset host-act cursor; allocate a slot for each
        #    (layer, chunk) pair with a home level >= 0.
        self.buffers.reset_host_act_cursor()
        self._host_act_slots: dict[tuple[int, int], ActivationSlot | None] = {}
        for layer in self.backbone:
            for chunk in prepared.chunks:
                lvl = plan.level_for(layer.layer_id, chunk.id)
                if lvl.is_on_device:
                    self._host_act_slots[(layer.layer_id, chunk.id)] = None
                    continue
                slot, _used = self.buffers.host_act_slot(
                    layer.schema,
                    num_tokens=chunk.meta.total_q,
                    level=lvl.value,
                )
                self._host_act_slots[(layer.layer_id, chunk.id)] = slot

        # 2. Record the initial act-slot-ready events: all slots are
        #    "safe to overwrite" at round start.
        for slot_idx in range(self.buffers.n_gpu_act_slots):
            self.events.act_slot_ready.record_on(
                slot_idx, self.streams.inbound
            )

        # 3. Embed every chunk into the transition table on the compute stream.
        embed_ctx = self._layer_context()
        self.buffers.transitions.clear()
        with torch.cuda.stream(self.streams.compute):
            for chunk in prepared.chunks:
                emb = self.embed.forward(
                    chunk.token_ids,
                    chunk.meta,
                    self.buffers.gpu_embed_params,
                    embed_ctx,
                )
                self.buffers.transitions[chunk.id] = emb

    # ==================================================================
    # Internal: forward pass.
    # ==================================================================

    def _forward_pass(
        self, prepared: PreparedRound, plan: SaveLevelPlan
    ) -> None:
        """Run one forward pass over all layers * all chunks.

        Mirrors ``orig/active_model.py:1240-1358``. Advances the
        activation ring one slot per (layer, chunk) pair (forward
        direction). Sends activations home at tier >= 0; keeps them on
        the device ring at tier == -1 (the last N_gpu_act_slots pairs).

        After this method:
          * ``self.buffers.transitions[chunk_id]`` holds the post-last-
            layer residual-stream output per chunk.
          * ``self.events.home_act_slot_available`` is populated for
            every saved pair.
          * ``self.events.inbound_act_slot_ready`` holds events for the
            on-device tail pairs.
          * ``self.events.dev_act_slot_mapping`` maps on-device pairs
            to their current GPU ring slot (an :class:`ActivationSlot`).
          * ``self._next_act_slot_prefetch`` names the (layer_ind,
            chunk_id) whose activations should be prefetched FIRST
            during backward (= the latest-in-forward pair that was
            sent home).
        """
        N_P = self.working_set.n_gpu_layers
        N_G = self.working_set.n_gpu_grads
        n_slots = self.buffers.n_gpu_act_slots
        total_chunks = len(prepared.chunks)
        num_layers = len(self.backbone)

        cur_act_slot = 0
        cur_weight_slot = 0
        # Start of backward: grad ring is oldest-layer at slot N_G-1,
        # newest-layer at slot 0. After backward we rotate.
        prefetched_grads = False
        self._next_act_slot_prefetch: tuple[int, int] = (-1, -1)

        ctx = self._layer_context()

        for layer_ind, layer in enumerate(self.backbone):
            lid = layer.layer_id
            # Wait on the layer's params being resident in this GPU slot.
            self.events.weight_inbound.wait_on(lid, self.streams.compute)

            # Fetch the weights dict for this layer at the ring slot it
            # lives in. The layout depends on layer spec (heterogeneous).
            weights = self.buffers.gpu_param_slot(
                cur_weight_slot, layer.param_spec
            )

            for chunk in prepared.chunks:
                # Wait: ring slot safe to overwrite.
                self.events.act_slot_ready.wait_on(
                    cur_act_slot, self.streams.compute
                )

                # Build the slot view at full max-tier + narrowed to num_tokens.
                computed_slot = self.buffers.gpu_act_slot(
                    cur_act_slot, layer.schema, num_tokens=chunk.meta.total_q
                )

                # Forward on the compute stream.
                with torch.cuda.stream(self.streams.compute):
                    x = self.buffers.transitions[chunk.id]
                    y = layer.forward(x, chunk.meta, weights, computed_slot, ctx)
                    self.buffers.transitions[chunk.id] = y

                lvl = plan.level_for(lid, chunk.id)
                if lvl.is_on_device:
                    # Keep on device ring; record events that let the
                    # backward prefetch loop know where to find it.
                    self.events.act_slot_ready.record_on(
                        cur_act_slot, self.streams.compute
                    )
                    self.events.inbound_act_slot_ready.record_on(
                        (lid, chunk.id), self.streams.compute
                    )
                    self.events.dev_act_slot_mapping[(lid, chunk.id)] = (
                        computed_slot
                    )
                    # Mark home_act_slot_available[...] as None
                    # (consumed sentinel), matching orig:1308.
                    self.events.home_act_slot_available.events[
                        (lid, chunk.id)
                    ] = None
                else:
                    # Offload to host on the outbound stream.
                    self.streams.outbound.wait_stream(self.streams.compute)
                    home_slot = self._host_act_slots[(lid, chunk.id)]
                    assert home_slot is not None
                    with torch.cuda.stream(self.streams.outbound):
                        send_home(home_slot, computed_slot, lvl.value)
                    # Ring slot becomes safe to overwrite once offload
                    # finishes.
                    self.events.act_slot_ready.record_on(
                        cur_act_slot, self.streams.outbound
                    )
                    self.events.home_act_slot_available.record_on(
                        (lid, chunk.id), self.streams.outbound
                    )
                    self._next_act_slot_prefetch = (layer_ind, chunk.id)

                # Advance ring — unless this is the very last pair
                # (which must stay put so backward can pick it up).
                is_last_pair = (
                    layer_ind == num_layers - 1
                    and chunk.id == total_chunks - 1
                )
                if not is_last_pair:
                    cur_act_slot = (cur_act_slot + 1) % n_slots

            # End of layer. Prefetch next layer's params, OR, if this
            # was among the last N_P layers, prefetch grad ring for
            # the tail of backward.
            if layer_ind + N_P < num_layers:
                next_layer_id = self.backbone[layer_ind + N_P].layer_id
                self.streams.inbound.wait_stream(self.streams.compute)
                with torch.cuda.stream(self.streams.inbound):
                    self.buffers.fetch_layer_params(
                        next_layer_id, cur_weight_slot, non_blocking=True
                    )
                    self.events.weight_inbound.record_on(
                        next_layer_id, self.streams.inbound
                    )
                # Mark current layer's event as consumed; the engine
                # will not wait on it again.
                self.events.weight_inbound.mark_consumed(lid)
            else:
                if not prefetched_grads:
                    # Prefetch / zero grads for the last N_G layers.
                    self.streams.inbound.wait_stream(self.streams.compute)
                    with torch.cuda.stream(self.streams.inbound):
                        for g in range(N_G):
                            grad_layer_id = self.backbone[
                                num_layers - g - 1
                            ].layer_id
                            slot_idx = N_G - g - 1
                            if self._zero_grad:
                                grad_dict = self.buffers.gpu_grad_slot(
                                    slot_idx,
                                    self.backbone[
                                        num_layers - g - 1
                                    ].param_spec,
                                )
                                for t in grad_dict.values():
                                    t.zero_()
                            else:
                                self.buffers.fetch_layer_grads(
                                    grad_layer_id, slot_idx,
                                    non_blocking=True,
                                )
                            self.events.grad_weight_inbound.record_on(
                                grad_layer_id, self.streams.inbound
                            )
                    prefetched_grads = True

            # Rotate weight ring unless we're at the last layer (which
            # will stay in place for backward's first iteration).
            if layer_ind < num_layers - 1:
                cur_weight_slot = (cur_weight_slot + 1) % N_P

        # Save the final values for use by backward.
        self._final_weight_slot = cur_weight_slot
        self._final_act_slot = cur_act_slot
        self._total_chunks = total_chunks

    # ==================================================================
    # Internal: head pass.
    # ==================================================================

    def _head_pass(
        self,
        prepared: PreparedRound,
        *,
        loss_scale: float,
        loss_fn: LossFn,
    ) -> tuple[float, int]:
        """Run head.forward_backward on every chunk.

        Returns ``(sum_of_per_token_loss, total_tokens)``. Writes
        per-token loss back into each :class:`Sequence`'s buffer.
        """
        if self._zero_grad:
            for t in self.buffers.gpu_head_grads.values():
                t.zero_()

        ctx = self._layer_context()
        round_loss = 0.0
        round_tokens = 0

        with torch.cuda.stream(self.streams.compute):
            for chunk in prepared.chunks:
                x = self.buffers.transitions[chunk.id]
                token_ctx = TokenContext(labels=chunk.label_ids)
                dx, stats = self.head.forward_backward(
                    x,
                    token_ctx,
                    chunk.meta,
                    self.buffers.gpu_head_params,
                    self.buffers.gpu_head_grads,
                    ctx,
                    loss_scale=loss_scale,
                    loss_fn=loss_fn,
                )
                self.buffers.transitions[chunk.id] = dx

                # Per-sequence writeback of per_token_loss (orig parity).
                loss_slice_cpu = stats.per_token_loss.detach().to("cpu")
                for ref in chunk.seqs:
                    s = ref.seq
                    sr0, sr1 = ref.seq_range
                    cr0, cr1 = ref.chunk_range
                    s.per_token_loss[sr0:sr1].copy_(loss_slice_cpu[cr0:cr1])
                round_loss += float(stats.per_token_loss.sum().item())
                round_tokens += stats.token_count

        return round_loss, round_tokens

    # ==================================================================
    # Internal: backward pass.
    # ==================================================================

    def _backward_pass(
        self,
        prepared: PreparedRound,
        plan: SaveLevelPlan,
        *,
        total_tokens_per_step: int | None,
    ) -> None:
        """Run backward in reverse traversal.

        Outer loop: layers (reverse). Inner loops: seq_groups (reverse),
        then chunks within group (reverse). This matches orig's
        scheduling invariant that lets the KV-context refresh hop one
        chunk "up" per iteration.

        After this method, grads for every backbone layer have been
        accumulated into host grad buffers (for N_G+ layers) or into
        the GPU grad ring (for the last N_G layers — they'll offload
        at the start of step()).
        """
        N_P = self.working_set.n_gpu_layers
        N_G = self.working_set.n_gpu_grads
        n_slots = self.buffers.n_gpu_act_slots
        num_layers = len(self.backbone)
        total_chunks = self._total_chunks

        cur_weight_slot = self._final_weight_slot
        cur_act_slot = self._final_act_slot
        cur_grad_slot = N_G - 1
        next_act_prefetch = self._next_act_slot_prefetch

        ctx = self._layer_context(total_tokens_per_step=total_tokens_per_step)

        for k_ind in range(num_layers - 1, -1, -1):
            layer = self.backbone[k_ind]
            lid = layer.layer_id

            # Wait on params + grads for this layer.
            self.events.weight_inbound.wait_on(lid, self.streams.compute)
            self.events.grad_weight_inbound.wait_on(
                lid, self.streams.compute
            )

            weights = self.buffers.gpu_param_slot(
                cur_weight_slot, layer.param_spec
            )
            grads = self.buffers.gpu_grad_slot(cur_grad_slot, layer.param_spec)

            cur_chunk_id = total_chunks - 1
            for seq_group_ind in range(len(prepared.seq_groups) - 1, -1, -1):
                group = prepared.seq_groups[seq_group_ind]
                # Ensure fwd_context refresh from the prior group is done
                # before we start consuming K/V in backward for this group.
                self.streams.compute.wait_stream(
                    self.streams.inbound_fwd_context
                )

                for chunk_in_group_ind in range(len(group) - 1, -1, -1):
                    chunk = group[chunk_in_group_ind]

                    # Wait on the chunk's activation slot being resident
                    # (either on the ring already, or just-prefetched).
                    self.events.inbound_act_slot_ready.wait_on(
                        (lid, chunk.id), self.streams.compute
                    )
                    dev_slot = self.events.dev_act_slot_mapping[
                        (lid, chunk.id)
                    ]

                    # Forward recompute (fills higher-tier fields that
                    # weren't saved) + backward.
                    with torch.cuda.stream(self.streams.compute):
                        layer.forward_recompute(
                            dev_slot, chunk.meta, weights, ctx
                        )
                        dx = self.buffers.transitions[chunk.id]
                        upstream_dx = layer.backward(
                            dx, chunk.meta, weights, grads, dev_slot, ctx
                        )
                        self.buffers.transitions[chunk.id] = upstream_dx

                    # Refresh fwd_context K/V for the PRIOR seq group (if
                    # any). The refresh runs on a dedicated stream so it
                    # overlaps with compute-stream work on the NEXT chunk.
                    self.streams.inbound_fwd_context.wait_stream(
                        self.streams.compute
                    )
                    self._update_fwd_context(
                        seq_group_ind=seq_group_ind,
                        chunk_in_group_ind=chunk_in_group_ind,
                        layer_ind=k_ind,
                        prepared=prepared,
                    )

                    # Prefetch next activation slot (if any pair is
                    # still offloaded on host and waiting to come back).
                    if next_act_prefetch[0] != -1:
                        nlay_ind, nchunk_id = next_act_prefetch
                        nlid = self.backbone[nlay_ind].layer_id
                        self._prefetch_activation(
                            layer_ind=nlay_ind,
                            chunk_id=nchunk_id,
                            prepared=prepared,
                            dest_act_slot=cur_act_slot,
                        )
                        # Compute the NEXT one to prefetch (walking in
                        # reverse order of the forward offload stream).
                        if nchunk_id > 0:
                            next_act_prefetch = (nlay_ind, nchunk_id - 1)
                        elif nlay_ind > 0:
                            next_act_prefetch = (
                                nlay_ind - 1, total_chunks - 1
                            )
                        else:
                            next_act_prefetch = (-1, -1)

                    # Advance the ring in reverse.
                    cur_act_slot = (cur_act_slot - 1) % n_slots
                    cur_chunk_id -= 1

            # End of this layer's backward pass. Offload its grads on
            # the outbound stream. MUST be inside ``with
            # torch.cuda.stream(outbound)`` — otherwise ``copy_`` runs
            # on the default stream and the wait_stream(compute) doesn't
            # synchronize the actual DMA.
            self.streams.outbound.wait_stream(self.streams.compute)
            with torch.cuda.stream(self.streams.outbound):
                self.buffers.offload_layer_grads(
                    lid, cur_grad_slot, non_blocking=True
                )

            # Prefetch previous layer's params into the weight ring
            # (reverse direction) for its upcoming backward iteration.
            if k_ind - N_P >= 0:
                prev_layer_id = self.backbone[k_ind - N_P].layer_id
                self.streams.inbound.wait_stream(self.streams.compute)
                with torch.cuda.stream(self.streams.inbound):
                    self.buffers.fetch_layer_params(
                        prev_layer_id, cur_weight_slot, non_blocking=True
                    )
                    self.events.weight_inbound.record_on(
                        prev_layer_id, self.streams.inbound
                    )
                self.events.weight_inbound.mark_consumed(lid)

            # Rotate weight ring in reverse.
            cur_weight_slot = (cur_weight_slot - 1) % N_P

            # Prefetch previous layer's grads.
            if k_ind - N_G >= 0:
                prev_grad_layer_id = self.backbone[k_ind - N_G].layer_id
                self.streams.inbound.wait_stream(self.streams.compute)
                self.streams.inbound.wait_stream(self.streams.outbound)
                with torch.cuda.stream(self.streams.inbound):
                    if self._zero_grad:
                        grad_dict = self.buffers.gpu_grad_slot(
                            cur_grad_slot,
                            self.backbone[k_ind - N_G].param_spec,
                        )
                        for t in grad_dict.values():
                            t.zero_()
                    else:
                        self.buffers.fetch_layer_grads(
                            prev_grad_layer_id,
                            cur_grad_slot,
                            non_blocking=True,
                        )
                    self.events.grad_weight_inbound.record_on(
                        prev_grad_layer_id, self.streams.inbound
                    )
                self.events.grad_weight_inbound.mark_consumed(lid)

            cur_grad_slot = (cur_grad_slot - 1) % N_G

    # ==================================================================
    # Internal: embed backward.
    # ==================================================================

    def _embed_backward(self, prepared: PreparedRound) -> None:
        if self.embed is None:
            return
        if self._zero_grad:
            for t in self.buffers.gpu_embed_grads.values():
                t.zero_()

        ctx = self._layer_context()
        with torch.cuda.stream(self.streams.compute):
            for chunk in prepared.chunks:
                dx = self.buffers.transitions[chunk.id]
                self.embed.backward(
                    dx,
                    chunk.token_ids,
                    chunk.meta,
                    self.buffers.gpu_embed_params,
                    self.buffers.gpu_embed_grads,
                    ctx,
                )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _layer_context(
        self, total_tokens_per_step: int | None = None
    ) -> LayerContext:
        """Build a fresh :class:`LayerContext` for one sub-phase."""
        return LayerContext(
            scratch=self.scratch,
            kv_cache=self.buffers.kv_fwd,
            stream=self.streams.compute,
            secondary_stream=self.streams.secondary_compute,
            total_tokens_per_step=total_tokens_per_step,
        )

    def _update_fwd_context(
        self,
        *,
        seq_group_ind: int,
        chunk_in_group_ind: int,
        layer_ind: int,
        prepared: PreparedRound,
    ) -> None:
        """Refresh the forward K/V context with activations from a
        chunk in the PRIOR seq group (or prior layer, if we're at the
        first chunk of a group).

        This is orig's subtle scheduling trick: during backward of the
        current (seq_group_ind, chunk_in_group_ind), we overwrite the
        K/V window's tail with the corresponding chunk from the
        previous seq group (same layer), so that the NEXT iteration of
        backward (seq_group_ind - 1) has the right K/V to recompute
        attention against.

        Mirrors ``orig/active_model.py:1040-1160``.
        """
        if layer_ind == 0 and seq_group_ind == 0:
            return

        # Find the source chunk whose saved K/V we'll copy into the
        # global ``kv_fwd`` window. Two paths:
        #
        # (A) Same layer, prior seq group at this ``chunk_in_group_ind``.
        #     Always preferred when available; same W_K/W_V on different
        #     tokens means K/V values differ but the source IS the right
        #     layer.
        #
        # (B) Fallback to the IMMEDIATE prior layer's last-seq-group
        #     same-chunk position. Used at seq_group=0 boundaries.
        #     For HETEROGENEOUS backbones the prior layer may be
        #     linear-attn (no ``xk``/``xv`` in its slot). That's fine:
        #     the next softmax-attn bwd doesn't run until we reach a
        #     softmax-attn layer, by which point Path A re-fires from
        #     the prior seq group. We just bail out cleanly via the
        #     ``not src_slot.has("xk")`` guard below.
        next_chunk: TrainingChunk | None = None
        next_layer_id: int = -1

        # Path A: same-layer prior-seq-group.
        if seq_group_ind > 0:
            for g_ind in range(seq_group_ind - 1, -1, -1):
                g = prepared.seq_groups[g_ind]
                if len(g) > chunk_in_group_ind:
                    next_chunk = g[chunk_in_group_ind]
                    next_layer_id = self.backbone[layer_ind].layer_id
                    break

        # Path B: immediate prior layer, last seq group, same chunk-in-group.
        if next_chunk is None and layer_ind > 0:
            for g_ind in range(len(prepared.seq_groups) - 1, -1, -1):
                g = prepared.seq_groups[g_ind]
                if len(g) > chunk_in_group_ind:
                    next_chunk = g[chunk_in_group_ind]
                    next_layer_id = self.backbone[layer_ind - 1].layer_id
                    break

        if next_chunk is None:
            return

        key = (next_layer_id, next_chunk.id)
        # Source depends on whether it's on device or home.
        meta = next_chunk.meta
        # Position in the KV window where this chunk's K/V lives.
        start_idx = (
            (meta.prior_seq_offsets_host[0] if meta.prior_seq_offsets_host else 0)
            + (meta.prior_seq_lens_host[0] if meta.prior_seq_lens_host else 0)
        )
        total_q = meta.total_q

        if key in self.events.inbound_act_slot_ready:
            # Source on device already.
            self.streams.inbound_fwd_context.wait_event(
                self.events.inbound_act_slot_ready.get(key)
            )
            src_slot: ActivationSlot = self.events.dev_act_slot_mapping[key]
            # Heterogeneous backbones: layers without softmax attention
            # (e.g. Qwen3-Next linear-attn layers) don't declare ``xk``/
            # ``xv`` activation fields. Skip the KV-window refresh for
            # those layers — they don't consume the global KV cache.
            if not src_slot.has("xk"):
                return
            with torch.cuda.stream(self.streams.inbound_fwd_context):
                self.buffers.kv_fwd.k[
                    start_idx : start_idx + total_q, :
                ].copy_(src_slot.xk)
                self.buffers.kv_fwd.v[
                    start_idx : start_idx + total_q, :
                ].copy_(src_slot.xv)
        else:
            # Source on host.
            avail = self.events.home_act_slot_available.get(key)
            if avail is not None:
                self.streams.inbound_fwd_context.wait_event(avail)
            home_slot = self._host_act_slots[key]
            if home_slot is None:
                return
            if not home_slot.has("xk"):
                return
            with torch.cuda.stream(self.streams.inbound_fwd_context):
                self.buffers.kv_fwd.k[
                    start_idx : start_idx + total_q, :
                ].copy_(home_slot.xk)
                self.buffers.kv_fwd.v[
                    start_idx : start_idx + total_q, :
                ].copy_(home_slot.xv)

    def _prefetch_activation(
        self,
        *,
        layer_ind: int,
        chunk_id: int,
        prepared: PreparedRound,
        dest_act_slot: int,
    ) -> None:
        """Prefetch an on-host activation slot into the GPU ring at
        ``dest_act_slot`` for an upcoming backward iteration.

        Mirrors ``orig/active_model.py:1465-1484``.
        """
        layer = self.backbone[layer_ind]
        lid = layer.layer_id
        chunk = prepared.chunks[chunk_id]

        self.streams.inbound.wait_stream(self.streams.compute)
        avail = self.events.home_act_slot_available.get((lid, chunk_id))
        if avail is not None:
            self.streams.inbound.wait_event(avail)

        # Build the dev slot at the SAVED level (not max_tier) so that
        # slot.has(name) returns False for fields that WEREN'T fetched
        # from host — those higher-tier fields are stale buffer views
        # from a prior use of this ring slot and forward_recompute
        # MUST overwrite them. If we returned a max-tier slot here,
        # slot.has("xq") would be True for prefetched offloaded layers
        # and recompute would silently skip — producing stale-xq bwd
        # (see [FINDING 17] in docs/NOTES.md).
        home_slot = self._host_act_slots[(lid, chunk_id)]
        assert home_slot is not None
        level = home_slot.level
        full_dev_slot = self.buffers.gpu_act_slot(
            dest_act_slot, layer.schema, num_tokens=chunk.meta.total_q
        )
        # Narrow dev_slot to the saved level for fetch_home (only the
        # offloaded fields need to be copied back from host).
        narrowed = ActivationSlot(
            schema=layer.schema,
            level=level,
            tensors={
                f.name: full_dev_slot._tensors[f.name]
                for f in layer.schema.persistent_fields_at_level(level)
                if f.offload
            },
        )
        with torch.cuda.stream(self.streams.inbound):
            fetch_home(narrowed, home_slot, level)
        # The slot we store for bwd keeps ALL _tensors views (needed
        # so slot.xq etc. still resolves via __getattr__ when
        # forward_recompute WRITES to them), but carries the SAVED
        # level so slot.has(name) correctly reports which fields are
        # already populated from host vs. need to be recomputed.
        dev_slot = ActivationSlot(
            schema=layer.schema,
            level=level,
            tensors=full_dev_slot._tensors,
        )
        self.events.dev_act_slot_mapping[(lid, chunk_id)] = dev_slot
        self.events.inbound_act_slot_ready.record_on(
            (lid, chunk_id), self.streams.inbound
        )

    # ==================================================================
    # Logging
    # ==================================================================

    def _log_round(
        self,
        round_idx: int,
        num_rounds: int,
        prepared: PreparedRound,
        plan: SaveLevelPlan,
    ) -> None:
        tiers: dict[int, int] = {}
        for lvl in plan.choices.values():
            tiers[lvl.value] = tiers.get(lvl.value, 0) + 1
        tier_str = ", ".join(
            f"level {k}: {v}"
            for k, v in sorted(tiers.items(), key=lambda kv: -kv[0])
        )
        print(
            f"[FlexTrain] round {round_idx + 1}/{num_rounds}: "
            f"{len(prepared.chunks)} chunks, {prepared.total_tokens} tokens. "
            f"save levels: {tier_str}. "
            f"est fwd={plan.estimated_fwd_time_ms:.1f}ms, "
            f"est recompute={plan.estimated_recompute_time_ms:.1f}ms "
            f"({plan.recompute_fraction:.2%})",
            flush=True,
        )
