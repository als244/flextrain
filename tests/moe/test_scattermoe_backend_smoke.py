"""Smoke test for ScatterMoEExpertCompute.

Constructs a MoESwiGLUFFN with the scattermoe backend, fakes the slot
fields and chunk_extra dict, runs fwd then bwd on tiny shapes, and
confirms no exceptions plus that the output tensors are populated.

Run from the repo root with the env's libcudart on LD_LIBRARY_PATH:

  LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \\
  PYTHONPATH=. python tests/scratch/test_scattermoe_backend_smoke.py
"""
import sys
import types

import torch

sys.path.insert(0, "/home/shein/Documents/flextrain")

from flextrain.ops.moe_backend import ScatterMoEExpertCompute


def _make_fake_slot(T, K, E, F, d_model, dtype, device):
    """Build a stand-in for ActivationSlot with the fields the
    scattermoe backend reads/writes.

    Avoids the full BufferManager machinery — we only need a
    SimpleNamespace with attribute access."""
    TK = T * K
    return types.SimpleNamespace(
        # Shared block fields (declared by MoESwiGLUFFN, populated by router_topk_softmax)
        x_router=torch.empty(T, E, device=device, dtype=dtype),
        router_weights=torch.empty(T, K, device=device, dtype=dtype),
        chosen_experts=torch.empty(T, K, device=device, dtype=torch.int32),
        x_up=torch.empty(TK, 2 * F, device=device, dtype=dtype),
        # Backend-private fields
        scattermoe_sorted_expert_idxs=torch.empty(TK, device=device, dtype=torch.int32),
        scattermoe_sorted_scattered_idxs=torch.empty(TK, device=device, dtype=torch.int32),
        scattermoe_expert_offsets=torch.empty(E, device=device, dtype=torch.int32),
        index_mapping=torch.empty(T, K, device=device, dtype=torch.int32),
        # Aux dict (used to pass dprobs from bwd to caller).
        aux={},
    )


def main():
    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Tiny shapes for fast smoke.
    T, K, E, F, d_model = 64, 2, 8, 64, 128

    # Inputs.
    x = torch.randn(T, d_model, device=device, dtype=dtype)
    # Fake routing: random topk + softmax.
    logits = torch.randn(T, E, device=device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    expert_p, expert_idxs = probs.topk(K, dim=-1)
    expert_p = (expert_p / expert_p.sum(dim=-1, keepdim=True)).to(dtype)
    expert_idxs = expert_idxs.to(torch.int32)

    weights = {
        # Option-B layout: w_up (E, 2F, d), w_down (E, d, F).
        "w_up": torch.randn(E, 2 * F, d_model, device=device, dtype=dtype) / (d_model ** 0.5),
        "w_down": torch.randn(E, d_model, F, device=device, dtype=dtype) / (F ** 0.5),
    }

    backend = ScatterMoEExpertCompute()
    print("Schema fields declared by backend:")
    for f in backend.activation_fields(num_experts=E, top_k=K, expert_dim=F, d_model=d_model, compute_dtype=dtype):
        print(f"  {f.name}  shape_fn(T={T})={f.shape_fn(T, {})}  dtype={f.dtype}")

    slot = _make_fake_slot(T, K, E, F, d_model, dtype, device)
    # Populate router state (the block normally does this BEFORE
    # calling backend.fwd, via flextrain's route_topk_softmax).
    slot.router_weights.copy_(expert_p)
    slot.chosen_experts.copy_(expert_idxs)
    chunk_extra = {}

    out = torch.empty(T, d_model, device=device, dtype=dtype)
    primary_stream = torch.cuda.current_stream()

    def scratch_fn(shape, dt):
        return torch.empty(shape, dtype=dt, device=device)

    print("\n--- fwd ---")
    backend.fwd(
        x, slot.router_weights, slot.chosen_experts, weights,
        out=out, residual=None,
        slot=slot, chunk_extra=chunk_extra, layer_id=0,
        primary_stream=primary_stream,
        secondary_stream=None,
        scratch_fn=scratch_fn,
    )
    torch.cuda.synchronize()
    print(f"  out: shape={out.shape}, dtype={out.dtype}, finite={out.isfinite().all().item()}")
    print(f"  x_up populated: nonzero count={slot.x_up.count_nonzero().item()}")
    print(f"  index_mapping range: [{slot.index_mapping.min().item()}, {slot.index_mapping.max().item()}]")
    print(f"  expert_offsets: {slot.scattermoe_expert_offsets.tolist()}")
    print(f"  out_expanded saved in chunk_extra: {('scattermoe.moe.out_expanded' in chunk_extra)}")

    print("\n--- bwd ---")
    dy = torch.randn_like(out)
    grads = {
        "g_up": torch.zeros_like(weights["w_up"]),
        "g_down": torch.zeros_like(weights["w_down"]),
    }
    dx = backend.bwd(
        dy, x, weights, grads,
        slot=slot, chunk_extra=chunk_extra, layer_id=0,
        primary_stream=primary_stream,
        secondary_stream=None,
        scratch_fn=scratch_fn,
    )
    torch.cuda.synchronize()
    print(f"  dx: shape={dx.shape}, finite={dx.isfinite().all().item()}")
    print(f"  g_up: nonzero={grads['g_up'].count_nonzero().item()}, finite={grads['g_up'].isfinite().all().item()}")
    print(f"  g_down: nonzero={grads['g_down'].count_nonzero().item()}, finite={grads['g_down'].isfinite().all().item()}")
    print(f"  slot.aux['moe_dprobs']: shape={slot.aux['moe_dprobs'].shape}")
    print()
    print("OK — backend fwd+bwd ran without exceptions.")


if __name__ == "__main__":
    main()
