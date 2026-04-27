"""Tests for :class:`flextrain.engine.buffers.BufferManager`.

Covers:
* Host ring allocation (params / grads / opt state) uses the backend.
* GPU ring views are contiguous and correctly shaped for heterogeneous
  layers.
* Param prefetch: host -> GPU DMA writes the expected values.
* Grad offload: GPU -> host DMA writes the expected values.
* Host activation buffer reserves per (layer, chunk) without
  exceeding capacity.
* Heterogeneous layer types: the ring is sized to the max across
  specs, smaller layers load fine.
* Destroy cleans up without error.

Runs on the 3090 under the ``flextrain`` conda env.
"""

from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.core.activation_schema import (  # noqa: E402
    ActivationField,
    ActivationSchema,
)
from flextrain.core.layer import ParamSpec, TensorSpec  # noqa: E402
from flextrain.core.working_set import WorkingSetConfig  # noqa: E402
from flextrain.engine.buffers import BufferManager  # noqa: E402
from flextrain.engine.host_memory import (  # noqa: E402
    LocalPinnedHostBackend,
    UnpinnedHostBackend,
)
from flextrain.optim.base import (  # noqa: E402
    OptimizerStateSpec,
    OptStateTensor,
)


DEVICE = "cuda:0"


def _dummy_dense_spec(d_model: int, expert_dim: int) -> ParamSpec:
    """Tiny dense-layer-like spec with a few tensors of different shapes.
    The expert_dim is baked into the closure so heterogeneous tests
    get layer-specific shapes."""
    # Capture expert_dim via default arg to avoid late-binding in the
    # loop that builds the layer list.
    return ParamSpec(
        tensors=(
            TensorSpec.simple(
                "w_attn_norm", lambda d: (d["d_model"],), torch.bfloat16
            ),
            TensorSpec.simple(
                "w_q", lambda d: (d["d_model"], d["d_model"]), torch.bfloat16
            ),
            TensorSpec.simple(
                "w_1",
                lambda d, ed=expert_dim: (d["d_model"], ed),
                torch.bfloat16,
            ),
        )
    )


def _dummy_schema() -> ActivationSchema:
    return ActivationSchema(
        fields=(
            ActivationField(
                "x_inp",
                lambda n, d: (n, d["d_model"]),
                torch.bfloat16,
                tier=0,
            ),
            ActivationField(
                "xq",
                lambda n, d: (n, d["d_model"]),
                torch.bfloat16,
                tier=1,
            ),
        ),
        max_tier=1,
    )


def _mk_working_set() -> WorkingSetConfig:
    return WorkingSetConfig(
        target_round_tokens=256,
        max_chunk_size=128,
        max_training_chunks=8,
        max_total_round_tokens=512,
        target_num_rounds=1,
        n_gpu_layers=2,
        n_gpu_grads=1,
        n_gpu_opt_layers=1,
        gpu_act_buffer_size=1 << 20,
        host_act_buffer_size=1 << 18,
        available_gpu_memory_bytes=1 << 30,
        available_host_memory_bytes=1 << 32,
        leeway_gpu_memory_bytes=0,
        leeway_host_memory_bytes=0,
        max_seq_len=256,
        hardware_env={},
        raw={},
    )


def _opt_spec() -> OptimizerStateSpec:
    return OptimizerStateSpec(
        tensors=(
            OptStateTensor("o_m", torch.float32),
            OptStateTensor("o_v", torch.float32),
        )
    )


def _mk_bm(
    *,
    n_layers: int = 3,
    expert_dims: list[int] | None = None,
    d_model: int = 64,
    backend=None,
) -> BufferManager:
    if not torch.cuda.is_available():  # pragma: no cover
        raise RuntimeError("buffers tests require CUDA")
    specs = [
        _dummy_dense_spec(d_model, (expert_dims or [128] * n_layers)[i])
        for i in range(n_layers)
    ]
    schemas = [_dummy_schema() for _ in range(n_layers)]
    dims = {
        "d_model": d_model,
        "expert_dim": max((expert_dims or [128])),
        "vocab_size": 256,
        "n_kv_heads": 2,
        "head_dim": 32,
    }
    return BufferManager(
        working_set=_mk_working_set(),
        dims=dims,
        layer_param_specs=specs,
        layer_schemas=schemas,
        opt_spec=_opt_spec(),
        device=DEVICE,
        host_backend=backend or UnpinnedHostBackend(),
    )


def test_host_allocation_shapes() -> None:
    bm = _mk_bm(n_layers=3)
    # 3 layers -> 3 host param dicts.
    assert len(bm.host_params) == 3
    assert len(bm.host_grads) == 3
    assert len(bm.host_opt) == 3
    # Each layer has 3 params, so 3 param tensors + 3 grad tensors + 6 opt.
    assert set(bm.host_params[0].keys()) == {"w_attn_norm", "w_q", "w_1"}
    assert set(bm.host_grads[0].keys()) == {"g_attn_norm", "g_q", "g_1"}
    assert set(bm.host_opt[0].host.keys()) == {
        "o_m_attn_norm", "o_v_attn_norm", "o_m_q", "o_v_q", "o_m_1", "o_v_1",
    }
    # Shapes respect TensorSpec.shape_fn.
    assert bm.host_params[0]["w_q"].shape == (64, 64)
    assert bm.host_params[0]["w_1"].shape == (64, 128)
    assert bm.host_params[0]["w_attn_norm"].shape == (64,)


def test_gpu_ring_sizing_and_views() -> None:
    bm = _mk_bm(n_layers=3)
    # Param ring: 2 slots * max(per-layer bytes).
    per_layer_bytes = (
        64 * 2 + 64 * 64 * 2 + 64 * 128 * 2  # bf16
    )
    assert bm.gpu_param_ring.numel() == 2 * per_layer_bytes
    # Slot view has correct keys + shapes + dtype.
    slot = bm.gpu_param_slot(0, bm.layer_param_specs[0])
    assert set(slot.keys()) == {"w_attn_norm", "w_q", "w_1"}
    assert slot["w_q"].shape == (64, 64)
    assert slot["w_q"].dtype == torch.bfloat16
    assert slot["w_q"].device.type == "cuda"


def test_param_prefetch_round_trip() -> None:
    bm = _mk_bm(n_layers=3)
    # Write known values to host.
    bm.host_params[1]["w_q"].fill_(2.5)
    bm.host_params[1]["w_attn_norm"].fill_(-3.0)
    bm.fetch_layer_params(1, 0)
    torch.cuda.synchronize()
    # GPU slot 0 should carry layer 1's values.
    gpu = bm.gpu_param_slot(0, bm.layer_param_specs[1])
    assert torch.allclose(gpu["w_q"], torch.full_like(gpu["w_q"], 2.5))
    assert torch.allclose(
        gpu["w_attn_norm"], torch.full_like(gpu["w_attn_norm"], -3.0)
    )


def test_grad_offload_round_trip() -> None:
    bm = _mk_bm(n_layers=3)
    # Fill GPU grad slot with known values, then offload.
    gpu = bm.gpu_grad_slot(0, bm.layer_param_specs[2])
    gpu["g_q"].fill_(0.125)
    gpu["g_1"].fill_(-0.5)
    # Before offload, host grad should still be zero.
    assert bm.host_grads[2]["g_q"].abs().sum().item() == 0.0
    bm.offload_layer_grads(2, 0)
    torch.cuda.synchronize()
    # After offload, host should match.
    assert torch.allclose(
        bm.host_grads[2]["g_q"], torch.full_like(bm.host_grads[2]["g_q"], 0.125)
    )
    assert torch.allclose(
        bm.host_grads[2]["g_1"], torch.full_like(bm.host_grads[2]["g_1"], -0.5)
    )


def test_heterogeneous_layer_types() -> None:
    """Layers with different expert_dim (hence different per-layer bytes)
    share the same ring. Smaller layers leave slack."""
    bm = _mk_bm(n_layers=3, expert_dims=[64, 128, 256])
    # Ring sized to max (expert_dim=256): 64*2 + 64*64*2 + 64*256*2.
    largest = 64 * 2 + 64 * 64 * 2 + 64 * 256 * 2
    assert bm.gpu_param_ring.numel() == 2 * largest
    # Each layer's slot-view has the right shape for ITS w_1.
    # Layer 0: expert_dim=64, Layer 1: 128, Layer 2: 256.
    slot0 = bm.gpu_param_slot(0, bm.layer_param_specs[0])
    slot1 = bm.gpu_param_slot(0, bm.layer_param_specs[1])
    slot2 = bm.gpu_param_slot(0, bm.layer_param_specs[2])
    assert slot0["w_1"].shape == (64, 64)
    assert slot1["w_1"].shape == (64, 128)
    assert slot2["w_1"].shape == (64, 256)


def test_host_act_slot_cursor_advances() -> None:
    bm = _mk_bm(n_layers=2)
    schema = bm.layer_schemas[0]
    bm.reset_host_act_cursor()
    slot_a, bytes_a = bm.host_act_slot(schema, num_tokens=64, level=0)
    slot_b, bytes_b = bm.host_act_slot(schema, num_tokens=64, level=0)
    # Two consecutive slots occupy disjoint storage.
    assert bytes_a == bytes_b
    # Their x_inp tensors should live at different offsets — check by
    # mutating one and seeing the other is untouched.
    slot_a.x_inp.fill_(7.0)
    assert slot_b.x_inp.abs().sum().item() == 0.0


def test_host_act_slot_exhaustion_raises() -> None:
    bm = _mk_bm(n_layers=2)
    schema = bm.layer_schemas[0]
    bm.reset_host_act_cursor()
    # Request something impossibly large.
    try:
        bm.host_act_slot(schema, num_tokens=10**6, level=1)
    except RuntimeError as e:
        assert "host act buffer exhausted" in str(e).lower()
        return
    raise AssertionError("Expected RuntimeError for exhausted host act buffer")


def test_opt_state_ring_swap() -> None:
    bm = _mk_bm(n_layers=2)
    # Write known host opt state, then stage to GPU opt ring.
    bm.host_opt[0].host["o_m_q"].fill_(4.25)
    bm.host_opt[0].host["o_v_q"].fill_(0.7)
    bm.swap_to_optimizer_state(n_gpu_opt_layers=1)
    bm.fetch_layer_opt(0, 0)
    torch.cuda.synchronize()
    gpu = bm.gpu_opt_slot(0, 0)
    assert "o_m_q" in gpu and "o_v_q" in gpu
    assert torch.allclose(gpu["o_m_q"], torch.full_like(gpu["o_m_q"], 4.25))
    assert torch.allclose(gpu["o_v_q"], torch.full_like(gpu["o_v_q"], 0.7))
    bm.restore_activation_ring()
    # After restore, gpu_opt_slot should error.
    try:
        bm.gpu_opt_slot(0, 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("gpu_opt_slot should raise after restore")


def test_destroy_is_safe() -> None:
    bm = _mk_bm(n_layers=2)
    bm.destroy()
    # Calling again shouldn't raise (LocalPinnedHostBackend dedups).


def test_local_pinned_backend() -> None:
    """End-to-end with the real LocalPinnedHostBackend (cudaHostRegister)."""
    bm = _mk_bm(n_layers=2, backend=LocalPinnedHostBackend())
    bm.host_params[0]["w_q"].fill_(1.25)
    bm.fetch_layer_params(0, 0)
    torch.cuda.synchronize()
    gpu = bm.gpu_param_slot(0, bm.layer_param_specs[0])
    assert torch.allclose(gpu["w_q"], torch.full_like(gpu["w_q"], 1.25))
    bm.destroy()


def _run_all() -> None:
    tests = [
        ("test_host_allocation_shapes", test_host_allocation_shapes),
        ("test_gpu_ring_sizing_and_views", test_gpu_ring_sizing_and_views),
        ("test_param_prefetch_round_trip", test_param_prefetch_round_trip),
        ("test_grad_offload_round_trip", test_grad_offload_round_trip),
        ("test_heterogeneous_layer_types", test_heterogeneous_layer_types),
        ("test_host_act_slot_cursor_advances",
         test_host_act_slot_cursor_advances),
        ("test_host_act_slot_exhaustion_raises",
         test_host_act_slot_exhaustion_raises),
        ("test_opt_state_ring_swap", test_opt_state_ring_swap),
        ("test_destroy_is_safe", test_destroy_is_safe),
        ("test_local_pinned_backend", test_local_pinned_backend),
    ]
    for name, fn in tests:
        print(f"  - {name}")
        fn()
    print(f"  OK ({len(tests)} tests)")


if __name__ == "__main__":
    _run_all()
