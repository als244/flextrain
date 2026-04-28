"""Pure-arithmetic byte counters for working-set sizing.

Mirrors ``orig/awsm_transformer/utils.py`` size helpers but lives in v2 so
``flextrain/core/working_set.py`` can compute its inputs without importing
``orig``. The functions here operate on the same ``model_dims`` dict shape
that orig used (so existing callers don't have to change), and are pure
arithmetic with no GPU dependencies.

Only the byte-counting helpers needed by the working-set solver live here;
operational helpers (number-theoretic utilities like :func:`prev_high_div`
or :func:`get_divisors`) are kept beside the solver in
:mod:`flextrain.core.working_set`.
"""

from __future__ import annotations

import bisect
from typing import Mapping, Sequence

import torch


_DTYPE_NAME_TO_TORCH = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "BF16": torch.bfloat16,
    "float16":  torch.float16,  "fp16": torch.float16,  "FP16": torch.float16,
    "float32":  torch.float32,  "fp32": torch.float32,  "FP32": torch.float32,
}


def torch_dtype_from_name(dtype_str: str | torch.dtype) -> torch.dtype:
    """Map a v2/orig ``model_dims["datatypes"][...]`` name to a torch dtype."""
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    try:
        return _DTYPE_NAME_TO_TORCH[dtype_str]
    except KeyError as exc:
        raise ValueError(f"unknown dtype name: {dtype_str!r}") from exc


def _itemsize(dims: Mapping, key: str) -> int:
    return torch_dtype_from_name(dims["datatypes"][key]).itemsize


# ---------------------------------------------------------------------------
# Per-parameter-group byte counters. Mirror ``orig/awsm_transformer/utils.py``.
# ---------------------------------------------------------------------------


def embedding_size_bytes(model_dims: Mapping) -> int:
    return (
        _itemsize(model_dims, "embed")
        * model_dims["d_model"]
        * model_dims["vocab_size"]
    )


def head_size_bytes(model_dims: Mapping) -> int:
    """Output projection + RMS-norm gain. Matches orig:24-36."""
    return (
        _itemsize(model_dims, "head_proj")
        * model_dims["d_model"]
        * model_dims["vocab_size"]
        + _itemsize(model_dims, "norm") * model_dims["d_model"]
    )


def context_size_bytes(model_dims: Mapping, context_window_size: int) -> int:
    """K-cache + V-cache for one direction. Matches orig:38-50."""
    ctx_dim = model_dims["n_kv_heads"] * model_dims["head_dim"]
    return 2 * context_window_size * ctx_dim * _itemsize(model_dims, "attn_proj")


def backbone_layer_size_bytes(model_dims: Mapping) -> int:
    """One transformer block: norms + Q/K/V/O + (shared + routed) MLP +
    optional router. Mirrors orig:52-109.

    For dense models, ``num_shared_experts == 1`` and ``num_routed_experts == 0``;
    the routed-expert + router term drops to zero and we get one SwiGLU MLP.
    """
    d = model_dims["d_model"]
    n_h = model_dims["n_heads"]
    hd = model_dims["head_dim"]
    n_kv = model_dims["n_kv_heads"]
    expert_dim = model_dims["expert_dim"]
    num_shared = model_dims["num_shared_experts"]
    num_routed = model_dims["num_routed_experts"]

    attn_dim = n_h * hd
    ctx_dim = n_kv * hd

    sz_attn = _itemsize(model_dims, "attn_proj")
    sz_expert = _itemsize(model_dims, "expert_proj")
    sz_router = _itemsize(model_dims, "router")
    sz_norm = _itemsize(model_dims, "norm")

    # 2 norms (attn + ffn)
    nbytes = 2 * sz_norm * d
    # Q + O projections
    nbytes += 2 * sz_attn * d * attn_dim
    # K + V projections
    nbytes += 2 * sz_attn * d * ctx_dim
    # SwiGLU shared experts: gate + up + down (= 3 d * expert_dim each)
    nbytes += num_shared * sz_expert * 3 * d * expert_dim
    # Routed experts + router (orig:99-107 double-counts the router term;
    # we keep the same arithmetic so byte-for-byte parity holds)
    if num_routed > 0:
        nbytes += sz_router * d * num_routed
        nbytes += num_routed * sz_expert * 3 * d * expert_dim
    nbytes += sz_router * d * num_routed
    return nbytes


def full_act_slot_size_bytes(model_dims: Mapping, chunk_size: int) -> int:
    """Bytes for the FULL save level (all activations persisted) at this
    chunk size. Mirrors orig:112-143.

    Counts every persisted field exactly once: the two RMS rstds (fp32),
    the residual-stream input, K/V acts, router book-keeping (zero for
    dense), attention result + softmax_lse + Q/O, and the SwiGLU up
    intermediates for both routed + shared paths.
    """
    d = model_dims["d_model"]
    n_h = model_dims["n_heads"]
    n_kv = model_dims["n_kv_heads"]
    hd = model_dims["head_dim"]
    expert_dim = model_dims["expert_dim"]
    num_shared = model_dims["num_shared_experts"]
    num_routed = model_dims["num_routed_experts"]
    top_k = model_dims["top_k"]

    sz_act = _itemsize(model_dims, "residual")
    sz_router = _itemsize(model_dims, "router")
    fp32 = torch.float32.itemsize
    int32 = torch.int32.itemsize

    # rstds (attn + ffn norm) -- always fp32 for numerical stability
    nbytes = 2 * chunk_size * fp32
    # x_inp (residual into attention norm)
    nbytes += chunk_size * d * sz_act
    # xk, xv
    nbytes += 2 * chunk_size * n_kv * hd * sz_act
    # MoE router book-keeping (zero for dense)
    nbytes += chunk_size * num_routed * sz_router  # x_router
    nbytes += num_routed * int32                   # expert_counts
    nbytes += chunk_size * top_k * sz_router       # router_weights
    nbytes += chunk_size * top_k * int32           # chosen_experts
    nbytes += chunk_size * top_k * sz_router       # scattered_router_weights
    # Attention outputs
    nbytes += chunk_size * n_h * hd * sz_act       # attn_result
    nbytes += n_h * chunk_size * fp32              # softmax_lse
    nbytes += chunk_size * n_h * hd * sz_act       # xq
    nbytes += chunk_size * d * sz_act              # xo
    # MLP up intermediates: routed (top_k * 2 * expert_dim) + shared
    nbytes += chunk_size * top_k * 2 * expert_dim * sz_act
    nbytes += chunk_size * num_shared * 2 * expert_dim * sz_act
    return nbytes


def min_act_slot_size_bytes(model_dims: Mapping, chunk_size: int) -> int:
    """Bytes for the MIN save level (only what the engine can't recompute
    cheaply). Mirrors orig:145-170 -- the first ten fields of
    :func:`full_act_slot_size_bytes`."""
    d = model_dims["d_model"]
    n_kv = model_dims["n_kv_heads"]
    hd = model_dims["head_dim"]
    num_routed = model_dims["num_routed_experts"]
    top_k = model_dims["top_k"]

    sz_act = _itemsize(model_dims, "residual")
    sz_router = _itemsize(model_dims, "router")
    fp32 = torch.float32.itemsize
    int32 = torch.int32.itemsize

    nbytes = 2 * chunk_size * fp32                  # rstds
    nbytes += chunk_size * d * sz_act               # x_inp
    nbytes += 2 * chunk_size * n_kv * hd * sz_act   # xk, xv
    nbytes += chunk_size * num_routed * sz_router   # x_router
    nbytes += num_routed * int32                    # expert_counts
    nbytes += chunk_size * top_k * sz_router        # router_weights
    nbytes += chunk_size * top_k * int32            # chosen_experts
    nbytes += chunk_size * top_k * sz_router        # scattered_router_weights
    return nbytes


def transformer_saved_act_sizes(
    model_dims: Mapping, chunk_size: int,
) -> tuple[int, ...]:
    """Per-tier home-bytes for one (chunk, layer) -- length 4 for the
    paper's standard 4-tier schema (min, attn-only, attn+xq+xo, full).
    Mirrors ``orig/awsm_transformer/saved_activations_policy.py``
    ``get_transformer_saved_act_sizes``.

    NOTE: this is a hand-coded estimate that assumes the pure-attention
    transformer schema. For hybrid (linear+full attention) backbones,
    pass actual layer ``ActivationSchema`` objects to the working-set
    solver via ``layer_schemas=`` instead — the solver uses
    ``schema.home_size_bytes`` directly when available, which captures
    arch-specific tier-0 fields (e.g. linear-attn ``lin_z`` /
    ``lin_q_rstd`` / ...).
    """
    d = model_dims["d_model"]
    n_h = model_dims["n_heads"]
    n_kv = model_dims["n_kv_heads"]
    hd = model_dims["head_dim"]
    expert_dim = model_dims["expert_dim"]
    num_shared = model_dims["num_shared_experts"]
    top_k = model_dims["top_k"]

    sz_act = _itemsize(model_dims, "residual")
    sz_router = _itemsize(model_dims, "router")
    fp32 = torch.float32.itemsize

    level_0 = min_act_slot_size_bytes(model_dims, chunk_size)
    # Level 1: + softmax_lse + attn_result
    level_1 = level_0 + n_h * chunk_size * fp32 + chunk_size * n_h * hd * sz_act
    # Level 2: + xq + xo
    level_2 = level_1 + chunk_size * n_h * hd * sz_act + chunk_size * d * sz_act
    # Level 3: + MLP up intermediates (routed + shared)
    level_3 = level_2 + chunk_size * top_k * 2 * expert_dim * sz_act
    level_3 += chunk_size * num_shared * 2 * expert_dim * sz_act
    return (level_0, level_1, level_2, level_3)


def layer_matmul_flops_per_token(model_dims: Mapping) -> int:
    """Forward-pass matmul FLOPs per token for one layer's attention + MLP
    projections. ``2 * active_params_per_layer`` per orig:174-191.

    Active params: 4 attention projections (Q,K,V,O) + SwiGLU MLP for shared
    + routed expert paths. For dense: ``num_shared=1, top_k=0`` so the
    routed term is zero.
    """
    d = model_dims["d_model"]
    n_h = model_dims["n_heads"]
    n_kv = model_dims["n_kv_heads"]
    hd = model_dims["head_dim"]
    expert_dim = model_dims["expert_dim"]
    num_shared = model_dims["num_shared_experts"]
    top_k = model_dims["top_k"]

    attn_dim = n_h * hd
    ctx_dim = n_kv * hd
    active_params = (
        2 * d * attn_dim
        + 2 * d * ctx_dim
        + 3 * (num_shared + top_k) * d * expert_dim
    )
    return 2 * active_params


# ---------------------------------------------------------------------------
# Number-theoretic helpers used by the chunk-size search. Same behavior as
# orig but with a smaller surface (no nearest-divisor fluff).
# ---------------------------------------------------------------------------


# HCN-ish list of "nice" round-token totals. Same constants as orig:297-312.
GOOD_BATCH_SIZES: tuple[int, ...] = (
    4096, 5040, 7560, 8192, 10080, 15120, 16384, 20160, 25200, 27720, 32768,
    45360, 50400, 55440, 60480, 65520, 65536, 70560, 75600, 80640, 83160,
    85680, 90720, 95760, 100800, 105840, 110880, 115920, 120960, 126000,
    131040, 131072, 136080, 141120, 151200, 155520, 160380, 161280, 166320,
    171360, 176400, 181440, 191520, 196560, 201600, 211680, 216720, 221760,
    226800, 231840, 241920, 246960, 252000, 262080, 262144, 272160, 277200,
    282240, 287280, 302400, 317520, 322560, 327600, 332640, 342720, 352800,
    357840, 362880, 378000, 383040, 393120, 403200, 408240, 415800, 423360,
    428400, 432432, 443520, 453600, 468720, 472500, 478800, 483840, 498960,
    504000, 524288, 514080, 524160, 529200, 544320, 554400, 564480, 574560,
    584640, 589680, 604800, 612360, 622440, 635040, 645120, 655200, 665280,
    680400, 685440, 695520, 705600, 720720, 725760, 730800, 740880, 756000,
    766080, 776160, 786240, 800800, 806400, 816480, 831600, 846720, 856800,
    864864, 871200, 876960, 887040, 907200, 917280, 937440, 942480, 957600,
    972720, 982800, 997920, 1009008, 1029600, 1048320, 1048576, 1058400,
    1081080,
)
_GOOD_BATCH_SORTED = sorted(GOOD_BATCH_SIZES)


def prev_high_div(n: int) -> int:
    """Largest "nice" divisor-rich integer ``<= n``. Mirrors orig:319-322."""
    idx = bisect.bisect_right(_GOOD_BATCH_SORTED, n)
    return _GOOD_BATCH_SORTED[idx - 1] if idx > 0 else _GOOD_BATCH_SORTED[0]


def get_divisors(x: int) -> list[int]:
    """All divisors of ``x``, sorted ascending. Mirrors orig:286-291."""
    out: list[int] = []
    for i in range(1, x + 1):
        if x % i == 0:
            out.append(i)
    return out


def round_to_nearest(x: float, base: int) -> int:
    """Round ``x`` to the nearest multiple of ``base``. Mirrors orig:231-235."""
    return int(base * round(x / base))
