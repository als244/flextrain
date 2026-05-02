"""Isolate the sonicmoe fwd path: call quack/sonicmoe primitives directly
on tiny tensors, compare against a hand-rolled reference. Three paths:

A. Sonicmoe-native: weight stored (E, 2F, d) contiguous, exact MoE.forward
   view chain. This MUST match the reference if quack itself is correct.

B. Flextrain materialization: weight stored (E, d, 2F) contiguous (flextrain
   layout). Apply transpose+contiguous to convert to sonicmoe layout, then
   the same view chain. This is what our backend does.

If A == reference but B != A, our materialization is wrong.
If A == B but != reference, the issue is elsewhere (routing semantics,
combine semantics, etc).

Run from the repo root:
  LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \\
  PYTHONPATH=. python tests/scratch/test_sonic_weight_handoff.py
"""
import sys
import torch
import torch.nn.functional as F_torch

sys.path.insert(0, "/home/shein/Documents/flextrain")


def naive_fwd(x, w_ft_up, w_ft_down, expert_p, expert_idxs):
    """Hand-rolled fwd with flextrain weight layout (E, in, out).
       w_ft_up: (E, d, 2F), w_ft_down: (E, F, d).
    """
    T, d = x.shape
    K = expert_p.shape[1]
    E, _, twoF = w_ft_up.shape
    F = twoF // 2

    out = torch.zeros(T, d, device=x.device, dtype=x.dtype)
    for t in range(T):
        for k in range(K):
            e = expert_idxs[t, k].item()
            pre = x[t] @ w_ft_up[e]                         # (2F,)
            value, gate = pre.chunk(2)                      # (F,), (F,)
            h = F_torch.silu(gate) * value                  # (F,)
            y = h @ w_ft_down[e]                            # (d,)
            out[t] += expert_p[t, k] * y
    return out


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA"); return
    cap = torch.cuda.get_device_capability()
    if cap < (9, 0):
        print(f"SKIP: needs sm_90+ (got sm_{cap[0]}{cap[1]})"); return

    try:
        from quack.gemm_interface import gemm, gemm_gated
        from sonicmoe.functional.triton_kernels import TC_topk_router_metadata_triton
        from sonicmoe.functional.forward import _router_forward
    except Exception as e:
        print(f"SKIP: import failed: {e}"); return

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    # Distinct dims so we can tell input vs output apart.
    T, K, E = 32, 2, 4
    d, F = 64, 32        # d != 2F
    twoF = 2 * F
    TK = T * K

    x = torch.randn(T, d, device=device, dtype=dtype)

    # ─── Build SAME logical weights in BOTH layouts ────────────────────
    # Flextrain layout (caller's storage convention)
    w_ft_up   = torch.randn(E, d, twoF, device=device, dtype=dtype) * 0.1
    w_ft_down = torch.randn(E, F, d,    device=device, dtype=dtype) * 0.1

    # Sonicmoe layout: (E, out, in) — derived by axes-1-2 transpose
    w_sn_up_storage   = w_ft_up.transpose(1, 2).contiguous()    # (E, 2F, d)
    w_sn_down_storage = w_ft_down.transpose(1, 2).contiguous()  # (E, d, F)

    # Sanity: same logical values, different storage
    assert w_ft_up[0, 5, 7].item() == w_sn_up_storage[0, 7, 5].item()
    assert w_ft_down[0, 3, 11].item() == w_sn_down_storage[0, 11, 3].item()
    print("Storage equivalence ✓")

    # ─── Routing ──────────────────────────────────────────────────────
    # Just pick K experts uniformly
    expert_idxs = torch.randint(0, E, (T, K), device=device, dtype=torch.int32)
    logits = torch.rand(T, K, device=device, dtype=torch.float32)
    expert_p = (logits / logits.sum(dim=-1, keepdim=True)).to(dtype)

    expert_freq = torch.empty(E, dtype=torch.int32, device=device)
    expert_freq_off = torch.empty(E + 1, dtype=torch.int32, device=device)
    x_gather_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)
    s_reverse_scatter_idx = torch.empty(TK, dtype=torch.int32, device=device)

    TC_topk_router_metadata_triton(
        expert_idxs, E,
        expert_freq, expert_freq_off,
        x_gather_idx, s_scatter_idx, s_reverse_scatter_idx,
    )
    print(f"expert_freq:        {expert_freq.tolist()}")
    print(f"expert_freq_off:    {expert_freq_off.tolist()}")
    print(f"x_gather_idx[:8]:   {x_gather_idx[:8].tolist()}")

    # ─── Path A: sonicmoe-native ──────────────────────────────────────
    def run_path(name, w_up_storage, w_down_storage):
        print(f"\n=== Path {name} ===")
        # Up-proj: w1 view = storage.permute(1,2,0); pass w1.permute(2,1,0)
        w1 = w_up_storage.permute(1, 2, 0)              # (2F, d, E)
        B_up = w1.permute(2, 1, 0)                       # (E, d, 2F)
        print(f"  B_up    shape={tuple(B_up.shape)}, strides={B_up.stride()}")
        preact = torch.empty(TK, twoF, device=device, dtype=dtype)
        postact = torch.empty(TK, F, device=device, dtype=dtype)
        gemm_gated(
            x, B_up, activation="swiglu",
            cu_seqlens_m=expert_freq_off, A_idx=x_gather_idx,
            preact_out=preact, postact_out=postact,
            store_preact=True, bias=None,
        )

        # Down-proj
        w2 = w_down_storage.permute(1, 2, 0)            # (d, F, E)
        B_dn = w2.permute(2, 1, 0)                       # (E, F, d)
        print(f"  B_dn    shape={tuple(B_dn.shape)}, strides={B_dn.stride()}")
        y = torch.empty(TK, d, device=device, dtype=dtype)
        gemm(postact, B_dn, out=y,
             cu_seqlens_m=expert_freq_off, bias=None)

        # Combine
        scores_flat = expert_p.view(-1).float().contiguous()
        out = torch.empty(T, d, device=device, dtype=dtype)
        _router_forward(
            y=y, o=out, topk_scores=scores_flat,
            s_reverse_scatter_idx=s_reverse_scatter_idx,
            num_activated_expert_per_token_offset=None,
            varlen_K_max=K, H=d, is_varlen_K=False,
        )
        torch.cuda.synchronize()
        return preact, postact, y, out

    pre_A, post_A, y_A, out_A = run_path("A: sonicmoe-native", w_sn_up_storage, w_sn_down_storage)
    pre_B, post_B, y_B, out_B = run_path(
        "B: flextrain materialize",
        w_ft_up.transpose(1, 2).contiguous(),
        w_ft_down.transpose(1, 2).contiguous(),
    )

    # ─── Hand-rolled reference using flextrain layout ──────────────────
    out_ref = naive_fwd(x.float(), w_ft_up.float(), w_ft_down.float(), expert_p.float(), expert_idxs).to(dtype)

    def cmp(label, a, b):
        cos = torch.nn.functional.cosine_similarity(
            a.float().flatten().unsqueeze(0),
            b.float().flatten().unsqueeze(0), dim=-1).item()
        max_abs = (a.float() - b.float()).abs().max().item()
        print(f"  {label:30s}  cos={cos:.6f}  max_abs={max_abs:.4e}")

    print("\n=== A vs B (does materialization preserve numerics?) ===")
    cmp("preact A vs B",  pre_A, pre_B)
    cmp("postact A vs B", post_A, post_B)
    cmp("y A vs B",       y_A, y_B)
    cmp("out A vs B",     out_A, out_B)

    print("\n=== A vs reference ===")
    cmp("out A vs ref", out_A, out_ref)

    print("\n=== B vs reference ===")
    cmp("out B vs ref", out_B, out_ref)

    print()
    print("Interpretation:")
    print("  A vs B mismatch → flextrain materialization is wrong")
    print("  A vs ref mismatch → sonicmoe call sequence is wrong (or routing/combine semantics)")
    print("  Both match → fwd is correct; bug is in bwd or downstream")


if __name__ == "__main__":
    main()
