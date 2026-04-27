"""Phase 1 smoke test: exercise the core abstractions on a dummy dense block.

Purpose
-------
Prove the new ``ActivationField`` / ``ActivationSchema`` / ``ActivationSlot``
abstractions produce byte-exact matches to the hand-written size arithmetic
in ``orig/awsm_transformer/dense_layer.py``, and that
``build_dp_tables`` produces rectangular, well-formed arrays.

This test does NOT exercise any real compute kernels; the ``Layer`` below is
a stub that returns canned ``ComputeCost`` values. Phase 2 brings the real
blocks.

Run with: ``python -m pytest tests/test_phase1_core.py -v``
       or ``python tests/test_phase1_core.py`` (direct script run works too).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import torch

from flextrain.core import (
    ActivationField,
    ActivationSchema,
    ActivationSlot,
    ChunkMeta,
    ComputeCost,
    Layer,
    ParamSpec,
    SaveLevel,
    SaveLevelPlan,
    TensorSpec,
    build_dp_tables,
)
from flextrain.core.activation_schema import (
    concat_fields,
    fetch_home,
    send_home,
)
from flextrain.core.save_level import HardwareCost, plan_from_solution


# ---------------------------------------------------------------------------
# Build a dense-transformer-shaped schema matching ``dense_layer.py:745-835``
# exactly. This is the reference for byte-size equivalence.
# ---------------------------------------------------------------------------


LLAMA3_8B_DIMS: dict[str, int] = {
    # from orig/model_dims.json["llama3_8B"]
    "d_model": 4096,
    "n_heads": 32,
    "n_kv_heads": 8,
    "head_dim": 128,
    "expert_dim": 14336,
}


def _attn_norm_rstd(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, 1)


def _ffn_norm_rstd(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, 1)


def _x_inp(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, d["d_model"])


def _xk(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, d["n_kv_heads"], d["head_dim"])


def _xv(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, d["n_kv_heads"], d["head_dim"])


def _attn_result(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, d["n_heads"], d["head_dim"])


def _softmax_lse(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    # (n_heads, num_tokens) -- token_axis=1
    return (d["n_heads"], n)


def _xq(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, d["n_heads"], d["head_dim"])


def _xo(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, d["d_model"])


def _x1(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, d["expert_dim"])


def _x3(n: int, d: Mapping[str, int]) -> tuple[int, ...]:
    return (n, d["expert_dim"])


def build_dense_schema() -> ActivationSchema:
    """Schema equivalent to ``TransformerLayer``'s act_slot dict in orig."""
    bf16 = torch.bfloat16
    f32 = torch.float32
    fields = (
        ActivationField("attn_norm_rstd", _attn_norm_rstd, f32, tier=0),
        ActivationField("ffn_norm_rstd", _ffn_norm_rstd, f32, tier=0),
        ActivationField("x_inp", _x_inp, bf16, tier=0),
        ActivationField("xk", _xk, bf16, tier=0),
        ActivationField("xv", _xv, bf16, tier=0),
        ActivationField("attn_result", _attn_result, bf16, tier=1),
        ActivationField("softmax_lse", _softmax_lse, f32, tier=1, token_axis=1),
        ActivationField("xq", _xq, bf16, tier=2),
        ActivationField("xo", _xo, bf16, tier=2),
        ActivationField("x1", _x1, bf16, tier=3),
        ActivationField("x3", _x3, bf16, tier=3),
    )
    return ActivationSchema(fields=fields, max_tier=3)


# ---------------------------------------------------------------------------
# Stub layer for DP table smoke test.
# ---------------------------------------------------------------------------


@dataclass
class StubLayer:
    """Minimum viable Layer: carries a schema + canned ComputeCost.

    The DP table builder only consults ``compute_cost`` and
    ``schema.offloaded_bytes_at_level``, so this is enough to exercise it.
    """

    layer_id: int
    schema: ActivationSchema
    param_spec: ParamSpec
    _cost: ComputeCost

    # Protocol-required methods we don't need for this test. Raising makes
    # accidental use loud.
    def forward(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def forward_recompute(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def backward(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def compute_cost(self, chunk: ChunkMeta) -> ComputeCost:
        return self._cost


def make_stub_chunk(num_tokens: int) -> ChunkMeta:
    return ChunkMeta.build(
        seq_lens=[num_tokens],
        seq_positions=list(range(num_tokens)),
        prior_seq_lens=[0],
        prior_seq_offsets=[0],
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_schema_byte_sizes_match_orig_arithmetic() -> None:
    """Byte-by-byte parity with ``dense_layer.py:837-916``."""
    schema = build_dense_schema()
    dims = LLAMA3_8B_DIMS
    num_tokens = 1024

    bf16_sz = torch.bfloat16.itemsize
    f32_sz = torch.float32.itemsize

    # Reproduce the hand-written formula for each tier:
    level0 = (
        num_tokens * f32_sz  # attn_norm_rstd
        + num_tokens * f32_sz  # ffn_norm_rstd
        + num_tokens * dims["d_model"] * bf16_sz  # x_inp
        + num_tokens * dims["n_kv_heads"] * dims["head_dim"] * bf16_sz  # xk
        + num_tokens * dims["n_kv_heads"] * dims["head_dim"] * bf16_sz  # xv
    )
    level1 = level0 + (
        num_tokens * dims["n_heads"] * dims["head_dim"] * bf16_sz  # attn_result
        + dims["n_heads"] * num_tokens * f32_sz  # softmax_lse
    )
    level2 = level1 + (
        num_tokens * dims["n_heads"] * dims["head_dim"] * bf16_sz  # xq
        + num_tokens * dims["d_model"] * bf16_sz  # xo
    )
    level3 = level2 + (
        num_tokens * dims["expert_dim"] * bf16_sz  # x1
        + num_tokens * dims["expert_dim"] * bf16_sz  # x3
    )

    assert schema.home_size_bytes(num_tokens, dims, 0) == level0
    assert schema.home_size_bytes(num_tokens, dims, 1) == level1
    assert schema.home_size_bytes(num_tokens, dims, 2) == level2
    assert schema.home_size_bytes(num_tokens, dims, 3) == level3

    # device_size is level3 (everything), since during fwd all fields land on device
    assert schema.device_size_bytes(num_tokens, dims) == level3


def test_schema_tier_monotonic() -> None:
    """Higher tiers must be a superset of lower tiers."""
    schema = build_dense_schema()
    names = [{f.name for f in schema.fields_at_level(L)} for L in range(4)]
    for a, b in zip(names, names[1:]):
        assert a.issubset(b)
    assert names[0] != names[3]


def test_schema_duplicate_name_rejected() -> None:
    try:
        ActivationSchema(
            fields=(
                ActivationField("x", lambda n, d: (n,), torch.bfloat16, tier=0),
                ActivationField("x", lambda n, d: (n,), torch.bfloat16, tier=1),
            ),
            max_tier=1,
        )
    except ValueError as e:
        assert "duplicate" in str(e)
    else:  # pragma: no cover
        raise AssertionError("duplicate field names should have been rejected")


def test_schema_tier_out_of_range_rejected() -> None:
    try:
        ActivationSchema(
            fields=(
                ActivationField("x", lambda n, d: (n,), torch.bfloat16, tier=5),
            ),
            max_tier=3,
        )
    except ValueError as e:
        assert "tier" in str(e)
    else:  # pragma: no cover
        raise AssertionError("out-of-range tier should have been rejected")


def test_slot_from_buffer_and_send_fetch_roundtrip() -> None:
    """Allocate a host buffer, slice into a slot, write tensors, send/fetch."""
    schema = build_dense_schema()
    dims = LLAMA3_8B_DIMS
    num_tokens = 32  # keep small so we can run on CPU without tensors blowing up
    level = 2

    # Build host and device slots backed by separate uint8 buffers.
    host_nbytes = schema.home_size_bytes(num_tokens, dims, level)
    host_buf = torch.zeros(host_nbytes, dtype=torch.uint8)
    host_slot, used = ActivationSlot.from_buffer(
        schema, level, num_tokens, dims, host_buf
    )
    assert used == host_nbytes

    # "device" slot for the test is another host uint8 buffer
    dev_nbytes = schema.device_size_bytes(num_tokens, dims)
    dev_buf = torch.zeros(dev_nbytes, dtype=torch.uint8)
    dev_slot, _ = ActivationSlot.from_buffer(
        schema,
        schema.max_tier,
        num_tokens,
        dims,
        dev_buf,
        include_nonpersistent=True,
    )

    # Write known values into every persistent field on the "device" side.
    for f in schema.persistent_fields_at_level(level):
        tensor = getattr(dev_slot, f.name)
        tensor.fill_(1.0)

    # Send -> host, zero device, fetch back, confirm parity.
    send_home(host_slot, dev_slot, level)
    for f in schema.persistent_fields_at_level(level):
        getattr(dev_slot, f.name).zero_()
        assert torch.all(getattr(host_slot, f.name) == 1.0)

    fetch_home(dev_slot, host_slot, level)
    for f in schema.persistent_fields_at_level(level):
        assert torch.all(getattr(dev_slot, f.name) == 1.0)


def test_slot_view_for_narrows_along_token_axis() -> None:
    """``softmax_lse`` has token_axis=1. ``view_for`` must narrow the right axis."""
    schema = build_dense_schema()
    dims = LLAMA3_8B_DIMS
    base_tokens = 128

    dev_buf = torch.zeros(
        schema.device_size_bytes(base_tokens, dims), dtype=torch.uint8
    )
    slot, _ = ActivationSlot.from_buffer(
        schema, 3, base_tokens, dims, dev_buf, include_nonpersistent=True
    )

    narrowed = slot.view_for(num_tokens=64, dims=dims)

    # x_inp narrows axis 0: (64, d_model)
    assert narrowed.x_inp.shape == (64, dims["d_model"])
    # softmax_lse narrows axis 1: (n_heads, 64)
    assert narrowed.softmax_lse.shape == (dims["n_heads"], 64)
    # xk unchanged axis beyond tokens
    assert narrowed.xk.shape == (64, dims["n_kv_heads"], dims["head_dim"])


def test_slot_has_and_missing_field_raises() -> None:
    schema = build_dense_schema()
    dims = LLAMA3_8B_DIMS
    num_tokens = 16
    level = 0

    host_buf = torch.zeros(
        schema.home_size_bytes(num_tokens, dims, level), dtype=torch.uint8
    )
    slot, _ = ActivationSlot.from_buffer(schema, level, num_tokens, dims, host_buf)

    assert slot.has("x_inp")
    assert not slot.has("x1")  # tier 3, not present at level 0

    try:
        _ = slot.x1
    except AttributeError as e:
        assert "x1" in str(e)
    else:  # pragma: no cover
        raise AssertionError("missing field access should raise AttributeError")


def test_dp_tables_shape_and_padding() -> None:
    """Heterogeneous backbone: one layer with max_tier=3 and one with
    max_tier=1. Confirm k_global is padded to 4 and that disallowed cells are
    -inf (value) / +inf (duration)."""
    bf16 = torch.bfloat16
    big_schema = build_dense_schema()  # max_tier=3
    small_schema = ActivationSchema(
        fields=(
            ActivationField("x_inp", _x_inp, bf16, tier=0),
            ActivationField("xk", _xk, bf16, tier=0),
            ActivationField("attn_result", _attn_result, bf16, tier=1),
        ),
        max_tier=1,
    )

    dims = LLAMA3_8B_DIMS
    param_spec = ParamSpec(tensors=())  # empty; irrelevant here

    cost_big = ComputeCost(
        total_fwd_flops=10_000,
        avoided_recompute_flops=(0, 2000, 5000, 8000),
    )
    cost_small = ComputeCost(
        total_fwd_flops=3_000,
        avoided_recompute_flops=(0, 1500),
    )

    l0 = StubLayer(0, big_schema, param_spec, cost_big)
    l1 = StubLayer(1, small_schema, param_spec, cost_small)

    chunks = [make_stub_chunk(256), make_stub_chunk(256)]
    hw = HardwareCost(peak_tflops=900.0, pcie_bw_gbps=50.0)

    tables = build_dp_tables([l0, l1, l0, l1], chunks, dims, hw)

    T = 4 * 2  # 4 layers * 2 chunks
    assert tables.T == T
    assert tables.k_global == 4
    assert tables.compute_times.shape == (T,)
    assert tables.values.shape == (T, 4)
    assert tables.transfer_durations.shape == (T, 4)

    # small_schema layers (layer_ids 1) should have -inf in cols 2 and 3
    for chunk_idx in range(2):
        t = tables.indexing[(1, chunk_idx)]
        assert tables.values[t, 0] > -1e17
        assert tables.values[t, 1] > -1e17
        assert tables.values[t, 2] < -1e17  # sentinel
        assert tables.values[t, 3] < -1e17  # sentinel
        assert tables.transfer_durations[t, 2] > 1e17
        assert tables.transfer_durations[t, 3] > 1e17

    # big_schema layers should have finite values at every column
    for chunk_idx in range(2):
        t = tables.indexing[(0, chunk_idx)]
        for L in range(4):
            assert abs(tables.values[t, L]) < 1e17
            assert tables.transfer_durations[t, L] < 1e17


def test_plan_from_solution_pins_tail_on_device() -> None:
    """The final ``n_gpu_act_slots`` tasks in the forward traversal
    order must be set to :attr:`SaveLevel.on_device` (``-1``).

    Matches ``orig/active_model.py:803-804``: the first
    ``n_home_act_slots`` pairs get DP-chosen tiers; the last
    ``n_gpu_act_slots`` pairs stay resident on the GPU activation ring
    (consumed by backward immediately, no host offload / re-fetch).
    """
    schema = build_dense_schema()
    dims = LLAMA3_8B_DIMS
    param_spec = ParamSpec(tensors=())
    cost = ComputeCost(
        total_fwd_flops=10_000,
        avoided_recompute_flops=(0, 2000, 5000, 8000),
    )
    layers = [StubLayer(i, schema, param_spec, cost) for i in range(3)]
    chunks = [make_stub_chunk(256) for _ in range(2)]
    hw = HardwareCost(peak_tflops=900.0, pcie_bw_gbps=50.0)
    tables = build_dp_tables(layers, chunks, dims, hw)

    # Pretend the solver picked all zeros for the non-tail tasks.
    choices = np.zeros(tables.T, dtype=np.int32)

    n_gpu_act_slots = 3
    plan = plan_from_solution(
        tables,
        choices,
        n_gpu_act_slots=n_gpu_act_slots,
        min_required_recompute_time_ms=0.0,
        max_optional_recompute_time_avoided_ms=100.0,
    )

    # Sort tasks in forward traversal order: the last 3 must all be on-device.
    sorted_t = sorted(tables.indexing.items(), key=lambda kv: kv[1])
    for (lid, cid), _t in sorted_t[-n_gpu_act_slots:]:
        assert plan.level_for(lid, cid).is_on_device, (
            f"tail (layer={lid}, chunk={cid}) should be on-device, "
            f"got {plan.level_for(lid, cid).value}"
        )
    # And the first (T - 3) should be a real tier >= 0.
    for (lid, cid), _t in sorted_t[: -n_gpu_act_slots]:
        assert plan.level_for(lid, cid).value >= 0, (
            f"non-tail (layer={lid}, chunk={cid}) should have tier >= 0, "
            f"got {plan.level_for(lid, cid).value}"
        )


def test_plan_on_device_fast_path() -> None:
    """``SaveLevelPlan.all_on_device`` returns every pair as ``SaveLevel(-1)``."""
    plan = SaveLevelPlan.all_on_device([0, 1, 2], [0, 1])
    assert len(plan.choices) == 6
    for lvl in plan.choices.values():
        assert lvl.is_on_device
        assert lvl.value == -1


def test_compute_cost_sum_and_monotone() -> None:
    c1 = ComputeCost(total_fwd_flops=100, avoided_recompute_flops=(0, 20, 50, 80))
    c2 = ComputeCost(total_fwd_flops=200, avoided_recompute_flops=(0, 30, 120, 180))
    total = ComputeCost.sum([c1, c2], max_tier=3)
    assert total.total_fwd_flops == 300
    assert total.avoided_recompute_flops == (0, 50, 170, 260)

    # Non-monotonic must raise
    try:
        ComputeCost(total_fwd_flops=100, avoided_recompute_flops=(0, 50, 40))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("non-monotone avoided_recompute_flops should raise")


def test_paramspec_merge_errors_on_collision() -> None:
    bf16 = torch.bfloat16
    a = ParamSpec(
        tensors=(TensorSpec.simple("w_q", lambda d: (d["d_model"], d["d_model"]), bf16),)
    )
    b = ParamSpec(
        tensors=(TensorSpec.simple("w_q", lambda d: (d["d_model"], d["d_model"]), bf16),)
    )
    try:
        ParamSpec.merge([a, b])
    except ValueError as e:
        assert "duplicate" in str(e)
    else:  # pragma: no cover
        raise AssertionError("duplicate names should have been rejected")


def test_tensorspec_per_role_dtypes() -> None:
    """Mixed-precision roles: compute bf16, master fp32, grad bf16, opt fp32."""
    shape = lambda d: (d["d_model"], d["d_model"])
    t = TensorSpec(
        name="w_q",
        shape_fn=shape,
        compute_dtype=torch.bfloat16,
        master_dtype=torch.float32,
        grad_dtype=torch.bfloat16,
        # opt_state_dtype defaults to float32
    )
    dims = {"d_model": 64}
    numel = 64 * 64

    assert t.compute_byte_size(dims) == numel * 2  # bf16
    assert t.master_byte_size(dims) == numel * 4  # fp32
    assert t.grad_byte_size(dims) == numel * 2  # bf16
    assert t.opt_state_byte_size(dims) == numel * 4  # fp32 (default)

    # simple() builds a uniform-dtype spec for block code that doesn't care
    s = TensorSpec.simple("w_k", shape, torch.bfloat16)
    assert s.compute_dtype == torch.bfloat16
    assert s.master_dtype == torch.bfloat16
    assert s.grad_dtype == torch.bfloat16
    assert s.opt_state_dtype == torch.float32  # always defaults to fp32


def test_paramspec_byte_size_by_role() -> None:
    """ParamSpec.byte_size(role) aggregates per-role sizes."""
    shape = lambda d: (d["d_model"],)
    dims = {"d_model": 1024}

    spec = ParamSpec(
        tensors=(
            TensorSpec(
                "w_a",
                shape,
                compute_dtype=torch.bfloat16,
                master_dtype=torch.float32,
            ),
            TensorSpec(
                "w_b",
                shape,
                compute_dtype=torch.bfloat16,
                master_dtype=torch.bfloat16,
            ),
        )
    )
    # compute: both bf16 -> 2 * 1024 * 2 = 4096
    assert spec.byte_size(dims, role="compute") == 4096
    # master: fp32 + bf16 -> 1024*4 + 1024*2 = 6144
    assert spec.byte_size(dims, role="master") == 6144
    # grad: both default to compute (bf16) -> 4096
    assert spec.byte_size(dims, role="grad") == 4096
    # opt_state: both default to fp32 -> 2 * 1024 * 4 = 8192
    assert spec.byte_size(dims, role="opt_state") == 8192

    try:
        spec.byte_size(dims, role="bogus")
    except ValueError as e:
        assert "unknown role" in str(e)
    else:  # pragma: no cover
        raise AssertionError("unknown role should raise ValueError")


# ---------------------------------------------------------------------------
# Script entry point (no pytest needed for a quick sanity run)
# ---------------------------------------------------------------------------


def _run_all() -> None:
    # Simple manual runner so you can just `python tests/test_phase1_core.py`.
    tests = [
        test_schema_byte_sizes_match_orig_arithmetic,
        test_schema_tier_monotonic,
        test_schema_duplicate_name_rejected,
        test_schema_tier_out_of_range_rejected,
        test_slot_from_buffer_and_send_fetch_roundtrip,
        test_slot_view_for_narrows_along_token_axis,
        test_slot_has_and_missing_field_raises,
        test_dp_tables_shape_and_padding,
        test_plan_from_solution_pins_tail_on_device,
        test_plan_on_device_fast_path,
        test_compute_cost_sum_and_monotone,
        test_paramspec_merge_errors_on_collision,
        test_tensorspec_per_role_dtypes,
        test_paramspec_byte_size_by_role,
    ]
    for fn in tests:
        print(f"... {fn.__name__}", flush=True)
        fn()
        print(f"ok  {fn.__name__}", flush=True)
    print(f"\nAll {len(tests)} Phase 1 smoke tests passed.")


if __name__ == "__main__":
    _run_all()
