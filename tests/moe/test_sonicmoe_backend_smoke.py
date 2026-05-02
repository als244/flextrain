"""Smoke test for SonicMoEExpertCompute.

Constructs the backend (which runs an sm_90 check + try-imports
sonic-moe / quack-kernels), fakes the slot fields and chunk_extra dict,
runs fwd then bwd on tiny shapes, confirms no exceptions and that
output tensors are populated.

Skips gracefully on non-Hopper GPUs and when sonic-moe isn't
installed.

Run from the repo root with the env's libcudart on LD_LIBRARY_PATH:

  LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \\
  PYTHONPATH=. python tests/scratch/test_sonicmoe_backend_smoke.py
"""
import sys
import types

import torch

sys.path.insert(0, "/home/shein/Documents/flextrain")


def _make_fake_slot(T, K, E, F, d_model, dtype, device):
    """Stand-in for ActivationSlot with all fields the sonic backend
    reads/writes. Avoids the BufferManager."""
    TK = T * K
    return types.SimpleNamespace(
        # Shared block fields (router state + tier-3 pre-act buffer).
        x_router=torch.empty(T, E, device=device, dtype=dtype),
        router_weights=torch.empty(T, K, device=device, dtype=dtype),
        chosen_experts=torch.empty(T, K, device=device, dtype=torch.int32),
        x_up=torch.empty(TK, 2 * F, device=device, dtype=dtype),
        # Sonic backend's private fields.
        sonic_s_scatter_idx=torch.empty(TK, device=device, dtype=torch.int32),
        sonic_s_reverse_scatter_idx=torch.empty(TK, device=device, dtype=torch.int32),
        sonic_x_gather_idx=torch.empty(TK, device=device, dtype=torch.int32),
        sonic_expert_frequency=torch.empty(E, device=device, dtype=torch.int32),
        sonic_expert_frequency_offset=torch.empty(E + 1, device=device, dtype=torch.int32),
        sonic_num_activated_offset=torch.empty(T + 1, device=device, dtype=torch.int32),
        index_mapping=torch.empty(T, K, device=device, dtype=torch.int32),
        aux={},
    )


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return
    cap = torch.cuda.get_device_capability()
    if cap < (9, 0):
        print(f"SKIP: SonicMoE requires sm_90+ (got sm_{cap[0]}{cap[1]}). "
              "Set SONICMOE_FORCE=1 to bypass at your own risk.")
        return

    try:
        from flextrain.ops.moe_backend import SonicMoEExpertCompute
    except Exception as e:
        print(f"SKIP: import failed: {e}")
        return

    try:
        backend = SonicMoEExpertCompute()
    except RuntimeError as e:
        print(f"SKIP: SonicMoEExpertCompute() construction failed: {e}")
        return

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Tiny shapes for fast smoke.
    T, K, E, F, d_model = 64, 2, 8, 64, 128

    x = torch.randn(T, d_model, device=device, dtype=dtype)
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

    print("Schema fields declared by backend:")
    for f in backend.activation_fields(num_experts=E, top_k=K, expert_dim=F, d_model=d_model, compute_dtype=dtype):
        # token_axis None means shape is independent of T; otherwise it scales.
        n = T  # populate at this chunk's num_tokens
        print(f"  {f.name}  shape={f.shape_fn(n, {})}  dtype={f.dtype}")

    slot = _make_fake_slot(T, K, E, F, d_model, dtype, device)
    slot.router_weights.copy_(expert_p)
    slot.chosen_experts.copy_(expert_idxs)
    chunk_extra: dict = {}

    out = torch.empty(T, d_model, device=device, dtype=dtype)
    primary_stream = torch.cuda.current_stream()

    def scratch_fn(shape, dt):
        return torch.empty(shape, dtype=dt, device=device)

    print("\n--- fwd ---")
    backend.fwd(
        x, slot.router_weights, slot.chosen_experts, weights,
        out=out, residual=None,
        slot=slot, chunk_extra=chunk_extra, layer_id=0,
        primary_stream=primary_stream, secondary_stream=None,
        scratch_fn=scratch_fn,
    )
    torch.cuda.synchronize()
    print(f"  out: shape={out.shape}, dtype={out.dtype}, "
          f"finite={out.isfinite().all().item()}")
    print(f"  x_up populated: nonzero count={slot.x_up.count_nonzero().item()}")
    print(f"  index_mapping range: "
          f"[{slot.index_mapping.min().item()}, {slot.index_mapping.max().item()}]")
    print(f"  sonic_expert_frequency: {slot.sonic_expert_frequency.tolist()}")
    print(f"  sonic_expert_frequency_offset: {slot.sonic_expert_frequency_offset.tolist()}")

    print("\n--- bwd ---")
    dy = torch.randn_like(out)
    grads = {
        "g_up": torch.zeros_like(weights["w_up"]),
        "g_down": torch.zeros_like(weights["w_down"]),
    }
    dx = backend.bwd(
        dy, x, weights, grads,
        slot=slot, chunk_extra=chunk_extra, layer_id=0,
        primary_stream=primary_stream, secondary_stream=None,
        scratch_fn=scratch_fn,
    )
    torch.cuda.synchronize()
    print(f"  dx: shape={dx.shape}, finite={dx.isfinite().all().item()}")
    print(f"  g_up:   nonzero={grads['g_up'].count_nonzero().item()}, "
          f"finite={grads['g_up'].isfinite().all().item()}")
    print(f"  g_down: nonzero={grads['g_down'].count_nonzero().item()}, "
          f"finite={grads['g_down'].isfinite().all().item()}")
    print(f"  slot.aux['moe_dprobs']: shape={slot.aux['moe_dprobs'].shape}, "
          f"finite={slot.aux['moe_dprobs'].isfinite().all().item()}")
    print()
    print("OK — sonic backend fwd+bwd ran without exceptions.")


if __name__ == "__main__":
    main()
