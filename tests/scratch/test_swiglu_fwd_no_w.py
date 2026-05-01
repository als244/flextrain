"""Verify the optional-W swiglu_fwd path is bit-exact vs the prior
torch.ones-fallback math, and benchmark the per-call CPU dispatch
savings.
"""
import sys, time, statistics

import torch

# Allow running outside the package (CUDA-12 deps in env).
sys.path.insert(0, "/home/shein/Documents/flextrain")
from flextrain.ops._kernels.moe import flextrain_swiglu_moe_fwd

torch.cuda.set_device(0)
torch.manual_seed(0)

T, F = 512, 512
x = torch.randn(T, 2 * F, device="cuda", dtype=torch.bfloat16)
out_no_w = torch.empty(T, F, device="cuda", dtype=torch.bfloat16)
out_with_w = torch.empty(T, F, device="cuda", dtype=torch.bfloat16)
ones = torch.ones((T,), device="cuda", dtype=torch.bfloat16)

# Warm
for _ in range(50):
    flextrain_swiglu_moe_fwd(x, out=out_no_w)
    flextrain_swiglu_moe_fwd(x, w=ones, out=out_with_w)
torch.cuda.synchronize()

# Parity: with w=ones should produce the same as no-w
flextrain_swiglu_moe_fwd(x, out=out_no_w)
flextrain_swiglu_moe_fwd(x, w=ones, out=out_with_w)
torch.cuda.synchronize()
diff = (out_no_w.float() - out_with_w.float()).abs().max().item()
print(f"parity: max abs diff = {diff:.4e}  ({'OK' if diff < 1e-3 else 'FAIL'})")

# Microbench: pure CPU dispatch (sync between calls).
def bench(fn, label, *, inner=2000):
    for _ in range(500): fn()
    torch.cuda.synchronize()
    t = 0
    for _ in range(inner):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        t += time.perf_counter_ns() - t0
    print(f"  {label:50s}: {t/inner/1000:.2f} us")

print("\nPer-call CPU dispatch (sync between calls):")
bench(lambda: flextrain_swiglu_moe_fwd(x, w=ones, out=out_with_w),
      "swiglu_fwd  w=ones (skips torch.ones now)")
bench(lambda: flextrain_swiglu_moe_fwd(x, out=out_no_w),
      "swiglu_fwd  w=None  (no GPU multiply)")
