# MoE backend tests + diagnostics

Tests, smoke runs, and diagnostics for `flextrain.ops.moe_backend.*`:

- `FlextrainMoEExpertCompute` — flextrain's per-expert dispatcher loop. Supports
  LoRA per-expert callbacks; supports tier 3 saving with cheap `fwd_recompute`.
- `ScatterMoEExpertCompute` — scattermoe Triton kernels. Tier 3 only currently
  (no `fwd_recompute` impl), no LoRA.
- `SonicMoEExpertCompute` — sonicmoe + quack CUTLASS DSL kernels. Tier 3 only
  with simple `fwd_recompute` (re-runs `gemm_gated`), no LoRA. sm_90+ only.

This directory holds two kinds of files:

1. **Tests** (`test_*.py`) — assert-driven correctness checks. Run them
   directly; non-zero exit = fail.
2. **Diagnostics** (`compare_*.py`, `inspect_*.py`) — exploratory scripts
   for investigating divergence between backends. Print stats; no
   assertions. Use when a test fails or when you want to characterize
   numerical behavior.

All scripts assume the repo root is the cwd and use the
`flextrain` conda env. Several need libcudart on `LD_LIBRARY_PATH`:

```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
```

---

## Tests (assertion-based, `test_*.py`)

### `test_moe_backend_parity.py`

Numeric parity of all three MoE backends against a hand-rolled
PyTorch+autograd reference on tiny synthetic inputs (T=128, K=2, E=8).
**The canonical correctness gate** — if this fails, the backend's math
is wrong and you should NOT proceed to integration testing.

Compares `out`, `dx`, `d_expert_p`, `g_up`, `g_down` per backend at
cos ≥ 0.999 vs the reference. Sonic auto-skips on non-Hopper GPUs.

```
PYTHONPATH=. python tests/moe/test_moe_backend_parity.py
```

Expected: `PASS — flextrain and scattermoe agree with naive reference;
sonic also passed.`

### `test_sonicmoe_backend_smoke.py`

Smoke test for `SonicMoEExpertCompute`: builds a fake `ActivationSlot`,
runs fwd then bwd on tiny shapes, asserts no exceptions and that
output tensors are populated. Doesn't check numerics — see
`test_moe_backend_parity.py` for that. Skips on non-Hopper.

```
PYTHONPATH=. python tests/moe/test_sonicmoe_backend_smoke.py
```

Useful first check after touching the sonic backend's slot fields,
kernel call signatures, or routing wiring.

### `test_scattermoe_backend_smoke.py`

Same shape as above for `ScatterMoEExpertCompute`. Doesn't require Hopper.

```
PYTHONPATH=. python tests/moe/test_scattermoe_backend_smoke.py
```

### `test_sonic_weight_handoff.py`

Standalone diagnostic from the sonic-backend bring-up: bypasses the
backend wrapper entirely and calls `quack.gemm_gated` two ways on the
same logical weights — once in sonicmoe-native storage, once via the
flextrain transpose-contiguous-permute materialization. Confirms the
weight handoff is correct independently of routing / scatter / combine.

Was the bisection tool that pinned down the swiglu chunked-vs-interleaved
mismatch during initial sonic implementation. Keep around for future
weight-layout debugging — if a sonic numeric bug appears, run this
first to rule out the weight-handoff side.

---

## Diagnostics (exploratory, `compare_*.py` + `inspect_*.py`)

These all use the dump infrastructure in `flextrain/ops/_moe_dump.py`
which is env-var-gated by `FLEXTRAIN_MOE_DUMP_DIR`. When set, the MoE
fwd and bwd hooks at `flextrain/ops/full_moe.py` save tensors per layer
per phase: `<dir>/<phase>_layer<L>_<name>.pt`.

Tensors dumped:
- **fwd**: `ffn_norm_output`, `x_router`, `chosen_experts`,
  `router_weights`, `out`
- **bwd**: `dy`, `g_router` (`g_up` and `g_down` are skipped by default
  since they're huge; set `FLEXTRAIN_MOE_DUMP_INCLUDE_BIG=1` to include
  them)

The wrapper script (`compare_moe_backends_e2e.py`) handles dump-dir
plumbing for you. Use the inspectors to dig into specific tensor pairs
afterward.

### `compare_moe_backends_e2e.py`

Top-level wrapper: runs `train.py` twice (one per backend), capturing
fwd+bwd dumps per layer. Configures `FLEXTRAIN_MOE_DUMP_DIR` and
collects tensors into per-run subdirectories under
`/home/as1669/storage/flextrain/moe_dump/` (override with
`FLEXTRAIN_MOE_DUMP_ROOT=...` env var).

Edit the `CONFIG` dict at the top to change model / steps / mem
budget. Default: 1-step run on Qwen3.5-35B-A3B at 65k tokens.

Outputs: dumps in `flextrain/`, `sonicmoe/` subdirs; logs at
`flextrain.log`, `sonicmoe.log`. After both runs finish, prints an
in-memory comparison table (loads tensor pairs into RAM — slow for
large dumps; prefer the streaming `compare_moe_backend_dumps.py` for
analysis).

```
python tests/moe/compare_moe_backends_e2e.py
```

What to look for: smoke check that both runs complete and produce
matching loss curves (within 1% noise). Tensor-level analysis comes
from the inspectors below.

### `compare_moe_backend_dumps.py`

Streaming pair-wise comparator. Walks two dump directories, loads one
matched-by-filename tensor pair at a time, computes
cos / mean_diff / std_diff / max_abs / ref_scale, writes one CSV row,
optionally deletes the pair (`--rm`).

```
python tests/moe/compare_moe_backend_dumps.py \
  --dir-a /path/to/dump_a --dir-b /path/to/dump_b \
  --out /path/to/compare.csv [--rm] [--include-big]
```

Default skips `g_up`/`g_down` (huge expert-weight-sized tensors).
Failures (cos < 0.999) are flagged. The CSV has one row per
(filename, name, cos, mean_diff, std_diff, max_abs, ref_scale, status).

What to look for in the output:
- Most rows should be `OK`. Sparse `FAIL`s in deep layers are
  generally bf16 reduction-order noise.
- A clean cascade pattern (early layers OK, deep layers progressively
  worse cos) is the signature of accumulated drift, NOT a kernel bug.
- A SHAPE_MISMATCH means the schema declared different shapes for the
  field across runs — investigate the schema, not the values.

### `inspect_dump_pair.py`

Single-pair deep-dive. Loads two paired tensors and prints:
- magnitude distribution (how many positions have |val| ≥ 1e-1, 1e-2, ...)
- per-magnitude-floor cosine — restricting to top 10% / 1% / 0.1% of positions
- top-50 by magnitude with paired (flex, sonic) values

```
python tests/moe/inspect_dump_pair.py /path/a.pt /path/b.pt
```

What to look for:
- If full cos ≈ 0 but cos at high-magnitude floors ≈ 1, the runs
  agree on the actual signal and disagree only on near-zero positions
  (bf16 floor noise). Common in deep layers with heavy-tailed
  gradients.
- If cos is low even at the highest magnitude floor, a real
  position-level disagreement exists. Could still be an explainable
  reordering (see `inspect_dy_invariants.py`) rather than a kernel bug.

### `inspect_dy_invariants.py`

Position-invariant comparison of `bwd_layer*_dy.pt` tensors. Computes
three metrics that are insensitive to per-position spike-relocation:

1. **Per-token L2-norm cosine** — reduces (T, d) → (T,) by L2-norming
   each row. Invariant to feature-dim reshuffling within a row.
2. **Sorted-value cosine** — sorts all T*d magnitudes ascending.
   Fully position-invariant; asks "do the sample distributions match?"
3. **Histogram total-variation distance** — log-magnitude bucket
   comparison. Lower = better.

```
python tests/moe/inspect_dy_invariants.py \
  --dir-a /path/dump_a --dir-b /path/dump_b
```

What to look for:
- If raw `cos_full` is ~0 (per-position cosine collapsed) BUT
  `cos_norm` and `cos_sort` are high, AND TV is low, the runs are
  producing the SAME gradient distribution at the SAME per-token
  magnitude — they only differ on which feature within a row holds
  each spike. This is the expected signature of MoE topk-tiebreak
  divergence. NOT a correctness bug.
- If `cos_norm` is low too, individual tokens are getting different
  gradient magnitudes — that's a real divergence and warrants
  investigation.

### `inspect_routing_balance.py`

Per-layer expert-load and router-weight distribution metrics from a
single dump dir. Useful for characterizing how balanced routing is on
a real pretrained MoE, and for confirming two backends produce
statistically identical routing.

```
python tests/moe/inspect_routing_balance.py \
  --dir /path/dump --num-experts 256
```

Per-layer columns:
- **min/median/mean/max** tokens per expert
- **cv** = std/mean (0 = balanced, 1 = std equals mean)
- **imb** = max/mean (1 = ideal, up to E×top_k worst case)
- **entH** = entropy(p_e) / log(E) ∈ [0, 1] (1 = uniform)
- **top1% / top5%** of total slot assignments handled by the most-
  loaded 1 / 5 experts
- **unused** = fraction of dead experts (got 0 tokens)
- **rw_top1 / top2 / topk** — mean per-token softmax weight at each
  topk-rank slot
- **rw_Hnorm** — mean per-token entropy of the topk softmax,
  normalized to [0, 1]

What to look for:
- Run on both backends' dumps separately and compare line by line.
  Backends should produce statistically identical metrics (modulo
  topk tiebreak noise).
- Patterns by depth: shallow layers usually balanced, middle layers
  most specialized, deep layers may have dead experts. This is a
  property of the checkpoint, not the backend.

---

## Where dumps live

By default, `compare_moe_backends_e2e.py` writes to
`/home/as1669/storage/flextrain/moe_dump/{flextrain,sonicmoe}/` on the
machine it runs on. Set `FLEXTRAIN_MOE_DUMP_ROOT` in your environment
to override. The naming is `<phase>_layer<L>_<name>.pt` — single-step
runs only; multi-step runs would clobber earlier files (intentional —
see `_moe_dump.py` docstring).

Dump volume estimate (Qwen3.5-35B-A3B, 1 step):
- fwd `out` (T, d) bf16: ~256 MB × 40 layers = ~10 GB per backend
- everything else: ~1 GB per backend
- Total per parity comparison: ~22 GB across both backends

If you want `g_up` / `g_down` (expert-weight-sized accumulators), set
`FLEXTRAIN_MOE_DUMP_INCLUDE_BIG=1` before running. Costs ~250 GB for
35B-A3B per backend — make sure you have the disk.

---

## Why the synthetic parity test can pass while e2e per-position cosine looks bad

The synthetic parity test (`test_moe_backend_parity.py`) is the canonical
correctness gate — it compares each backend against a hand-rolled
autograd reference with **fixed random inputs**, **uniform-magnitude
random weights**, and a **single MoE block** in isolation. At e2e-shape
dimensions (T=2048, K=8, E=256, F=1024, d=2048) all three backends pass
at cos ≥ 0.9998 across `out`, `dx`, `d_expert_p`, `g_up`, `g_down`.

But when you run `compare_moe_backends_e2e.py` on a real model
(Qwen3.5-35B-A3B), the per-position cosine on `dy` (the bwd input
piped into each MoE layer) collapses to ~0.03 at shallow layers, even
though loss curves agree to within 6e-3 across 5 steps. **Same code,
seemingly contradictory signals.** Three things conspire:

### 1. Pretrained-checkpoint gradients are heavy-tailed and sparse

A pretrained 35B-A3B model on a math-instruction batch produces
gradients where most (T, d) positions have magnitude ~1e-7 and a few
thousand positions have magnitude ~1e-1 — four orders of magnitude
spread. Cosine on the flattened tensor is dominated by **where the
large-magnitude positions land**, not by the bulk distribution. If two
runs produce gradients with **identical magnitude distribution** but
the spikes happen at slightly different (token, feature) coordinates,
per-position cosine collapses to ~0 even though the runs are
"distributionally identical."

The synthetic parity test uses `randn`-uniform inputs, which give a
**flat magnitude distribution** — every position contributes roughly
equally to the cosine numerator and denominator. Spike-relocation
doesn't matter because there are no spikes. Hence the synthetic test
"sees" the actual numerical agreement that the e2e dy comparison
"loses" in the heavy tail.

### 2. Topk tiebreak sensitivity moves the spikes

Once router logits are computed, both backends pick the top-K experts
per token. For tokens where the rank-K and rank-(K+1) router logits
are within bf16 epsilon of each other, the two backends' kernel-level
reduction orders disagree on which is bigger. ~few % of tokens land
in this near-tie zone (`inspect_routing_ties.py` quantifies it).

When sonic and flextrain pick different experts for the same token,
the gradient flowing back through that token gets routed through a
**different per-expert weight chain** in each run. The token's
gradient magnitude is similar (same token, similar expert weights),
but it lands at a different physical (token, feature) coordinate in
the gather output. Over 40 layers, the cumulative position-shuffling
makes per-position cosine essentially uncorrelated.

The synthetic test routes via fixed `expert_idxs` (passed in as test
input), so there's no topk tiebreak — both backends route through
identical expert assignments and never see this divergence path.

### 3. bf16 reduction-order in heavy-tailed sums

Even when routing agrees, computing `gemm_gated`'s per-expert
matmul-and-activate involves summing across thousands of tokens'
contributions. flextrain's per-expert loop uses sequential `addmm`
into a buffer; sonic uses a single fused grouped GEMM with
warp-synchronous accumulators. These sum the same numbers in
**different orders**, and bf16 (with fp32 accumulators) is
reduction-order-sensitive. Each layer's MoE bwd amplifies any
cumulative roundoff from the layer above.

The synthetic test is one MoE block — there's no layer-cascade for
roundoff to compound through. The e2e run cascades 40 layers, so
even tiny per-layer reduction-order differences accumulate.

### Conclusion: how to interpret e2e dump comparisons

- **Loss curves agreeing within 1% over multiple steps**: strong
  evidence backends are equivalent for training.
- **`cos_norm` (per-token L2-norm cosine) high + `cos_sort`
  (sorted-value cosine) high**: gradient distributions match;
  per-position cosine collapse is just spike-relocation.
- **Synthetic parity at e2e dims passes**: kernels are mathematically
  correct.

If all three hold, per-position cosine ≈ 0 on a real-model dy is
**expected behavior**, not a bug. Use `inspect_dy_invariants.py` to
distinguish "kernels disagree" from "spikes moved."

If synthetic parity FAILS at e2e dims, that's a real bug — go fix it
before doing e2e analysis.

If synthetic parity PASSES but `cos_norm` / `cos_sort` are also low,
something subtle (e.g. wrong gradient accumulation across chunks for
multi-chunk runs — see the multi-chunk-parity todo) is amiss; dig in.

## Typical investigation flow

1. Run `test_moe_backend_parity.py` first — small synthetic, fast,
   catches kernel-level math bugs.
2. If that passes, run `test_<backend>_smoke.py` to confirm the
   backend works end-to-end on the actual `ActivationSlot` / kernel
   contract.
3. For real-model parity, run `compare_moe_backends_e2e.py` (a few
   minutes for 1 step on a 35B-A3B model) to dump tensors.
4. If loss curves agree but tensor-level diffs are large, run
   `inspect_dy_invariants.py` first — most apparent divergences are
   topk-tiebreak position swaps that don't affect the gradient
   *distribution*.
5. If invariant metrics also disagree, run `inspect_dump_pair.py` on
   a specific layer's `dy` to dig into magnitude buckets.
6. Run `inspect_routing_balance.py` on each dump separately to
   confirm or rule out routing-pattern divergence.
