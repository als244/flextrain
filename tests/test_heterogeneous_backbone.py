"""Heterogeneous-backbone support.

Stress-test: a backbone that mixes layer types with:

* different ``window_size_left`` (LlamaBlock full-context vs.
  SlidingWindowLlamaBlock with W=4096) -- same schema/params, different
  kernel flag.
* different ``max_tier`` (proxy for "some layers have fewer save levels,"
  e.g. simpler layers without FFN internals to save) -- exercises the DP
  padding path.

The engine does not special-case layer type anywhere; all it sees is the
:class:`~flextrain.core.Layer` Protocol. This test confirms that's true
by building a real DP table over a mix and checking:

* ``k_global`` is the MAX across layers.
* Low-max-tier rows have -inf in padded columns.
* Per-layer schemas stay independent (no cross-contamination).
* Per-layer param_specs stay independent.

CPU-only; no GPU needed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from flextrain.core import (
    ActivationField,
    ActivationSchema,
    ChunkMeta,
    ComputeCost,
    ParamSpec,
    build_dp_tables,
)
from flextrain.core.save_level import HardwareCost
from flextrain.nn.layers import (
    LlamaBlock,
    LlamaBlockConfig,
    MistralBlock,
    MistralBlockConfig,
    Qwen3DenseBlock,
    Qwen3DenseBlockConfig,
    Qwen3DenseSWABlock,
    Qwen3DenseSWABlockConfig,
)


BASE_CFG = LlamaBlockConfig(
    d_model=256,
    n_heads=4,
    n_kv_heads=2,
    head_dim=64,
    expert_dim=512,
    rope_base=500000.0,
)


def test_mixed_llama_and_mistral_backbone() -> None:
    """Llama (full-context GQA) + Mistral (sliding-window GQA) in the same
    backbone. Same schema shapes (both pick GQA variants), different kernel
    flag; DP treats them identically memory-wise."""
    full_cfg = BASE_CFG
    mistral_cfg = MistralBlockConfig(
        d_model=BASE_CFG.d_model,
        n_heads=BASE_CFG.n_heads,
        n_kv_heads=BASE_CFG.n_kv_heads,
        head_dim=BASE_CFG.head_dim,
        expert_dim=BASE_CFG.expert_dim,
        window_size_left=128,
        rope_base=BASE_CFG.rope_base,
    )
    backbone = [
        LlamaBlock(0, full_cfg),
        MistralBlock(1, mistral_cfg),
        LlamaBlock(2, full_cfg),
    ]

    # Schemas must agree on field names + byte sizes (same compute shape)
    # but must be DIFFERENT OBJECTS (no sharing).
    schema_ids = {id(layer.schema) for layer in backbone}
    assert len(schema_ids) == len(backbone), "schemas must not be shared"
    names0 = [f.name for f in backbone[0].schema.fields]
    for layer in backbone[1:]:
        assert [f.name for f in layer.schema.fields] == names0

    # Param specs likewise independent.
    assert (
        len({id(layer.param_spec) for layer in backbone}) == len(backbone)
    )

    # Build a DP table. The SWA layer should have SAME compute cost /
    # transfer cost as its full-context neighbors (the window only changes
    # what the kernel attends to, not the kernel's memory footprint).
    chunks = [ChunkMeta.build([64], list(range(64)), [0], [0], device="cpu")]
    hw = HardwareCost(peak_tflops=900.0, pcie_bw_gbps=50.0)
    tables = build_dp_tables(backbone, chunks, BASE_CFG.dims(), hw)

    assert tables.k_global == 4  # all three layers have max_tier=3
    assert tables.T == 3  # 3 layers * 1 chunk

    # Same compute time for all three (same kernel shapes / same model dims).
    assert np.allclose(tables.compute_times, tables.compute_times[0])


def test_mixed_max_tier_backbone() -> None:
    """A layer with ``max_tier=2`` alongside two ``max_tier=3`` layers.

    Build a custom Layer-Protocol object with a truncated schema to simulate
    a block type (e.g. a lightweight one) that chooses to never have tier-3
    fields. The DP table must pad the shallow layer's row at column 3 with
    sentinel values.
    """

    full_layer = LlamaBlock(0, BASE_CFG)

    # Build a shallow-schema stub from the same LlamaBlock's first three tiers.
    shallow_fields = tuple(
        f for f in full_layer.schema.fields if f.tier <= 2
    )
    shallow_schema = ActivationSchema(
        fields=shallow_fields, max_tier=2
    )

    class ShallowStubLayer:
        """Minimum Layer Protocol implementor for DP-table testing."""

        def __init__(self, layer_id: int, schema, param_spec):
            self.layer_id = layer_id
            self.schema = schema
            self.param_spec = param_spec

        def compute_cost(self, chunk):
            # 3-element avoided tuple for max_tier=2.
            return ComputeCost(
                total_fwd_flops=1_000_000,
                avoided_recompute_flops=(0, 200_000, 500_000),
            )

        def forward(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        def forward_recompute(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

        def backward(self, *a, **k):  # pragma: no cover
            raise NotImplementedError

    shallow = ShallowStubLayer(
        layer_id=1,
        schema=shallow_schema,
        param_spec=full_layer.param_spec,  # reuse; content irrelevant here
    )
    backbone = [full_layer, shallow, LlamaBlock(2, BASE_CFG)]

    chunks = [ChunkMeta.build([64], list(range(64)), [0], [0], device="cpu")]
    hw = HardwareCost(peak_tflops=900.0, pcie_bw_gbps=50.0)
    tables = build_dp_tables(backbone, chunks, BASE_CFG.dims(), hw)

    # k_global = max across layers = 4 (LlamaBlock's max_tier=3 + 1)
    assert tables.k_global == 4

    # Shallow layer row: columns 0,1,2 finite; column 3 padded with
    # -inf value / +inf duration.
    t_shallow = tables.indexing[(1, 0)]
    assert abs(tables.values[t_shallow, 0]) < 1e17
    assert abs(tables.values[t_shallow, 1]) < 1e17
    assert abs(tables.values[t_shallow, 2]) < 1e17
    assert tables.values[t_shallow, 3] < -1e17  # sentinel
    assert tables.transfer_durations[t_shallow, 3] > 1e17

    # Full layer rows: all four columns finite.
    for lid in (0, 2):
        t = tables.indexing[(lid, 0)]
        for L in range(4):
            assert abs(tables.values[t, L]) < 1e17
            assert tables.transfer_durations[t, L] < 1e17

    # max_tier_per_task reflects each layer's own max_tier.
    assert tables.max_tier_per_task[tables.indexing[(0, 0)]] == 3
    assert tables.max_tier_per_task[tables.indexing[(1, 0)]] == 2
    assert tables.max_tier_per_task[tables.indexing[(2, 0)]] == 3


def test_qwen3_alternating_swa_backbone() -> None:
    """Qwen3-style backbone: first N layers full-context, rest SWA. This
    mirrors how Qwen3's ``max_window_layers`` config is materialized -- a
    concrete real-world case of a heterogeneous backbone.

    Qwen3 also adds per-head Q/K RMSNorm (one
    :class:`RMSNormBlock(per_head=True)` per Q/K branch), which
    contributes extra tier-0 rstd fields -- confirming the schema
    compares equal across Qwen3 layers but DIFFERS from Llama's schema.
    """
    d = 128
    n_heads = 4
    n_kv = 2
    head_dim = 32
    expert_dim = 256
    n_layers = 4
    max_window_layers = 2  # first 2 full, last 2 SWA

    full_cfg = Qwen3DenseBlockConfig(
        d_model=d,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        expert_dim=expert_dim,
    )
    swa_cfg = Qwen3DenseSWABlockConfig(
        d_model=d,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        head_dim=head_dim,
        expert_dim=expert_dim,
        window_size_left=64,
    )
    backbone = []
    for i in range(max_window_layers):
        backbone.append(Qwen3DenseBlock(i, full_cfg))
    for i in range(max_window_layers, n_layers):
        backbone.append(Qwen3DenseSWABlock(i, swa_cfg))

    # Qwen3 schemas: the ``q_norm_rstd`` + ``k_norm_rstd`` fields are
    # present (proving QK-norm composition works). Llama schema does NOT
    # have those two fields.
    qwen_fields = {f.name for f in backbone[0].schema.fields}
    assert "q_norm_rstd" in qwen_fields
    assert "k_norm_rstd" in qwen_fields

    llama_layer = LlamaBlock(
        0,
        LlamaBlockConfig(
            d_model=d,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            expert_dim=expert_dim,
        ),
    )
    llama_fields = {f.name for f in llama_layer.schema.fields}
    assert "q_norm_rstd" not in llama_fields
    # Qwen3 schema is a STRICT superset of Llama schema (both dense,
    # Qwen3 adds QK-norm rstds).
    assert llama_fields.issubset(qwen_fields)

    # Param specs: Qwen3 has two more norm weights (``w_q_norm``, ``w_k_norm``).
    qwen_params = {t.name for t in backbone[0].param_spec.tensors}
    llama_params = {t.name for t in llama_layer.param_spec.tensors}
    assert "w_q_norm" in qwen_params
    assert "w_k_norm" in qwen_params
    assert llama_params.issubset(qwen_params)

    # DP table over mixed full + SWA Qwen3 layers. Same max_tier=3 for all
    # Qwen3 variants; k_global should be 4.
    from flextrain.core.save_level import HardwareCost

    chunks = [ChunkMeta.build([64], list(range(64)), [0], [0], device="cpu")]
    hw = HardwareCost(peak_tflops=900.0, pcie_bw_gbps=50.0)
    tables = build_dp_tables(backbone, chunks, full_cfg.dims(), hw)
    assert tables.T == n_layers
    assert tables.k_global == 4
    # All four layers have the same compute cost in FLOPs because the
    # attention kernel operates on the same shapes regardless of window.
    assert np.allclose(tables.compute_times, tables.compute_times[0])


def _run_all() -> None:
    tests = [
        test_mixed_llama_and_mistral_backbone,
        test_mixed_max_tier_backbone,
        test_qwen3_alternating_swa_backbone,
    ]
    for fn in tests:
        print(f"... {fn.__name__}", flush=True)
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    print(f"\nAll {len(tests)} heterogeneous-backbone tests passed.")


if __name__ == "__main__":
    _run_all()
