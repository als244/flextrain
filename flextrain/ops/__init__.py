"""FlexTrain kernel package.

Triton + Python kernels, now owned by FlexTrain. Implementations live
under ``flextrain.ops._kernels.*``; this module re-exports the public
``flextrain_*`` API.

The ``matmul_dispatcher`` package (a compiled C++ extension providing
cuBLASLt bindings) is an in-tree helper under ``helpers/matmul_dispatcher``
and is built automatically by ``pip install -e .``.
"""

from __future__ import annotations

# Public kernel API (renamed from the original awsm_* prefix).
from ._kernels.adamw import flextrain_adamw_step
from ._kernels.attention import (
    FlashAttentionNotAvailableError,
    flextrain_attention_bwd,
    flextrain_attention_fwd,
)
from ._kernels.cross_entropy import (
    flextrain_cross_entropy_loss,
    flextrain_softmax_cross_entropy_loss,
)
from ._kernels.embed import flextrain_embedding_bwd
from ._kernels.gate_prep import flextrain_gate_prep_fwd
from ._kernels.gated_rmsnorm import (
    flextrain_gated_rmsnorm_bwd,
    flextrain_gated_rmsnorm_fwd,
)
from ._kernels.l2norm import (
    flextrain_l2norm_bwd_into,
    flextrain_l2norm_fwd_into,
)
from ._kernels.moe import (
    flextrain_copy_expert_counts,
    flextrain_fused_topk_softmax,
    flextrain_load_balance_bwd,
    flextrain_moe_gather,
    flextrain_moe_router_gate_bwd,
    flextrain_moe_scatter,
    flextrain_moe_scatter_routing_weights,
    flextrain_moe_sort,
    flextrain_swiglu_moe_bwd,
    flextrain_swiglu_moe_bwd_prescaled,
    flextrain_swiglu_moe_fwd,
)
from ._kernels.muon import flextrain_muon_step
from ._kernels.rmsnorm import (
    flextrain_rmsnorm_bwd,
    flextrain_rmsnorm_fwd,
    flextrain_rmsnorm_fwd_recompute,
)
from ._kernels.rope import flextrain_rope_bwd, flextrain_rope_fwd
from ._kernels.rope_partial import (
    flextrain_rope_partial_bwd,
    flextrain_rope_partial_fwd,
)
from ._kernels.sample_top_p import flextrain_sample_top_p
from ._kernels.silu_bwd import flextrain_silu_bwd
from ._kernels.softmax import flextrain_softmax
from ._kernels.swiglu import flextrain_swiglu_bwd, flextrain_swiglu_fwd
from ._kernels.gelu_tanh_gated import (
    flextrain_gelu_tanh_gated_bwd,
    flextrain_gelu_tanh_gated_fwd,
)

# cuBLASLt matmul dispatchers (external compiled extension).
from ._kernels._matmul_dispatchers import dispatcher, dispatcher_secondary

# Pinned-memory helpers.
from ._kernels._mem_register import (
    destory_tensor as destroy_pinned_tensor,
    pin_tensor,
)

__all__ = [
    "FlashAttentionNotAvailableError",
    "destroy_pinned_tensor",
    "dispatcher",
    "dispatcher_secondary",
    "flextrain_adamw_step",
    "flextrain_attention_bwd",
    "flextrain_attention_fwd",
    "flextrain_copy_expert_counts",
    "flextrain_cross_entropy_loss",
    "flextrain_embedding_bwd",
    "flextrain_gate_prep_fwd",
    "flextrain_gelu_tanh_gated_bwd",
    "flextrain_gelu_tanh_gated_fwd",
    "flextrain_gated_rmsnorm_bwd",
    "flextrain_gated_rmsnorm_fwd",
    "flextrain_l2norm_bwd_into",
    "flextrain_l2norm_fwd_into",
    "flextrain_fused_topk_softmax",
    "flextrain_load_balance_bwd",
    "flextrain_moe_gather",
    "flextrain_moe_router_gate_bwd",
    "flextrain_moe_scatter",
    "flextrain_moe_scatter_routing_weights",
    "flextrain_moe_sort",
    "flextrain_muon_step",
    "flextrain_rmsnorm_bwd",
    "flextrain_rmsnorm_fwd",
    "flextrain_rmsnorm_fwd_recompute",
    "flextrain_rope_bwd",
    "flextrain_rope_fwd",
    "flextrain_rope_partial_bwd",
    "flextrain_rope_partial_fwd",
    "flextrain_sample_top_p",
    "flextrain_silu_bwd",
    "flextrain_softmax",
    "flextrain_softmax_cross_entropy_loss",
    "flextrain_swiglu_bwd",
    "flextrain_swiglu_fwd",
    "flextrain_swiglu_moe_bwd",
    "flextrain_swiglu_moe_bwd_prescaled",
    "flextrain_swiglu_moe_fwd",
    "pin_tensor",
]
