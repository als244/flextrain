# Tutorial — full ladder (block → layer → model)

This tutorial walks the **full** procedure for adding a new
architecture: write a new block, wire it into a new layer, register
the layer as a new model. Use this when your target arch's
per-layer math doesn't match any existing layer family — i.e., you
can't reuse `LlamaBlock` / `Qwen3DenseBlock` / `OLMoEBlock` / etc.

If you only need an arch table + weight-unpacking hook (the common
case — your model is structurally Llama / Qwen / OLMoE under
different HF tensor names), use [`tutorial_phi3.md`](tutorial_phi3.md)
instead. That walkthrough exercises composability with existing
blocks and layers; it's shorter.

If you haven't read [`flow.md`](flow.md), do it first. The mental
model below assumes you know the per-step traversal, the slot /
chunk lifecycle, and what the engine assumes about layers.

## What we're adding: `ParArch`

A simplified parallel-residual transformer. Per layer:

```
attn_out = attn(attn_norm(x))
ffn_out  = ffn(ffn_norm(x))
x        = x + attn_out + ffn_out          # parallel residual
```

vs. Llama's serial residual (`x = x + attn(...)` then `x =
x + ffn(...)`). The FFN is a vanilla GeLU MLP, no gating:

```
ffn(h) = W_2 · GeLU(W_1 · h)               # vanilla MLP
```

vs. Llama's SwiGLU (`W_2 · (SiLU(W_1·h) ⊙ (W_3·h))`).

This shape matches **GPT-NeoX / Pythia / GPT-J / Falcon** at the
parallel-residual + vanilla-MLP level. Those real archs ALSO use
LayerNorm (not RMSNorm), have attention biases, and use partial
RoPE — three additional wrinkles we omit here so the procedural
backbone stays clear. Each is a mechanical extension of what
follows: write a `LayerNormBlock` (mirror `RMSNormBlock` with mean
centering + bias), pass `bias=True` to `GQAAttentionConfig`,
configure `apply_rope_partial_*` instead of full RoPE.

We need three new things:

1. **`MLPBlock`** — a new block under `flextrain/nn/blocks/`,
   because vanilla GeLU MLP isn't in the catalog (we have
   `SwiGLUFFN`, `MoESwiGLUFFN`, `MoESwiGLUSharedExpertFFN`, none of
   them vanilla).
2. **`ParArchBlock`** — a new layer under `flextrain/nn/layers/`,
   because parallel residual isn't in any existing layer.
3. **`ParArchForCausalLM` arch** — the `ArchSpec` + `BuildContext`
   builder + side-effect import.

## Step 1 — Write the new block (`MLPBlock`)

Create `flextrain/nn/blocks/ffn_mlp.py`. The block declares its
activation fields, parameter spec, compute cost, and `fwd` / `bwd`
methods. Mirror `flextrain/nn/blocks/ffn_dense.py` (the SwiGLUFFN
sibling) for the surrounding boilerplate.

```python
# flextrain/nn/blocks/ffn_mlp.py
"""Vanilla GeLU MLP block.

ffn(h) = W_2 · gelu(W_1 · h)

Used by GPT-2-/GPT-NeoX-/Pythia-/GPT-J-/Falcon-style architectures.
For Llama-style gated FFN, see :class:`SwiGLUFFN` in ``ffn_dense.py``.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, MutableMapping
import torch
import torch.nn.functional as F

from flextrain.core.activation_schema import ActivationField
from flextrain.core.layer import (
    ChunkMeta, ComputeCost, LayerContext, ParamSpec, TensorSpec,
)


@dataclass(frozen=True)
class MLPConfig:
    d_model: int
    expert_dim: int
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None


class MLPBlock:
    """Vanilla GeLU MLP: up-projection, GeLU activation, down-projection."""

    def __init__(self, cfg: MLPConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # 1. fields() — declare what activations this block writes to slot
    # ------------------------------------------------------------------
    def fields(self) -> tuple[ActivationField, ...]:
        cfg = self.cfg
        bf = cfg.compute_dtype
        # x_up = W_1 @ ffn_norm_output. Save at tier 3: when saved we skip
        # re-projecting in bwd; when not saved we recompute from ffn_norm_rstd.
        return (
            ActivationField(
                "x_up",
                lambda n, d: (n, cfg.expert_dim),
                bf, tier=3,
            ),
        )

    # ------------------------------------------------------------------
    # 2. param_spec() — declare what tensors this block owns
    # ------------------------------------------------------------------
    def param_spec(self) -> ParamSpec:
        cfg = self.cfg
        return ParamSpec(tensors=(
            TensorSpec(
                "w_1",                         # up projection
                lambda d: (cfg.d_model, cfg.expert_dim),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
            TensorSpec(
                "w_2",                         # down projection
                lambda d: (cfg.expert_dim, cfg.d_model),
                compute_dtype=cfg.compute_dtype,
                master_dtype=cfg.master_dtype,
                grad_dtype=cfg.grad_dtype,
            ),
        ))

    # ------------------------------------------------------------------
    # 3. fwd — compute. The layer hands us the FFN-norm output, the
    #    pre-FFN residual, the slot, and an output tensor to write into.
    # ------------------------------------------------------------------
    def fwd(
        self,
        ffn_norm_output: torch.Tensor,           # (T, d_model)
        weights: Mapping[str, torch.Tensor],
        residual: torch.Tensor,                  # (T, d_model) — added inline
        out_tensor: torch.Tensor,                # write target (T, d_model)
        slot,
        ctx: LayerContext,
    ) -> torch.Tensor:
        # x_up = ffn_norm_output @ W_1
        torch.matmul(ffn_norm_output, weights["w_1"], out=slot.x_up)

        # h = gelu(x_up). Compute in scratch — we don't save the post-gelu.
        h = ctx.scratch(slot.x_up.shape, slot.x_up.dtype)
        torch.nn.functional.gelu(slot.x_up, approximate="tanh", out=h)

        # out_tensor = residual + h @ W_2  (fused via addmm, so the
        # residual add doesn't need a second pass over GPU memory)
        torch.addmm(residual, h, weights["w_2"], out=out_tensor)
        return out_tensor

    # ------------------------------------------------------------------
    # 4. bwd — accumulate g_w1 / g_w2 into grads, return dx into ffn_norm
    # ------------------------------------------------------------------
    def bwd(
        self,
        dy_resid: torch.Tensor,                  # (T, d_model)
        ffn_norm_output: torch.Tensor,           # rematerialized by the layer
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
        *,
        skip_grads: frozenset[str] = frozenset(),
        capture_xy: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        # 1. Down-proj bwd: dy_h = dy_resid @ w_2^T. We need x_up to
        #    rebuild h = gelu(x_up) for the W_2 Wgrad operand.
        if not slot.has("x_up"):
            # Tier 3 wasn't saved — layer should have recomputed by now.
            raise RuntimeError("MLPBlock.bwd: x_up missing; recompute first")
        x_up = slot.x_up
        h = ctx.scratch(x_up.shape, x_up.dtype)
        torch.nn.functional.gelu(x_up, approximate="tanh", out=h)

        dy_h = torch.matmul(dy_resid, weights["w_2"].T)               # (T, expert)

        # 2. g_w2 += h^T @ dy_resid  (skip if LoRA fast-path requests it)
        if "g_2" in skip_grads:
            if capture_xy is not None:
                capture_xy["w_2"] = (h.clone(), dy_resid.clone())
        else:
            grads["g_2"].addmm_(h.T, dy_resid)

        # 3. GeLU bwd: dx_up = gelu_grad(x_up) * dy_h, in-place into dy_h
        gelu_grad = _gelu_tanh_grad(x_up)
        dy_h.mul_(gelu_grad)
        dx_up = dy_h                                                  # rename

        # 4. g_w1 += ffn_norm_output^T @ dx_up
        if "g_1" in skip_grads:
            if capture_xy is not None:
                capture_xy["w_1"] = (ffn_norm_output.clone(), dx_up.clone())
        else:
            grads["g_1"].addmm_(ffn_norm_output.T, dx_up)

        # 5. Return dL/d(ffn_norm_output) for the layer to feed into RMSNorm bwd.
        return torch.matmul(dx_up, weights["w_1"].T)                  # (T, d_model)

    # ------------------------------------------------------------------
    # 5. compute_cost — total fwd FLOPs + per-tier avoided-recompute FLOPs
    # ------------------------------------------------------------------
    def compute_cost(self, chunk: ChunkMeta, max_tier: int) -> ComputeCost:
        cfg = self.cfg
        avoided = [0] * (max_tier + 1)
        total = 0
        for seq_len in chunk.seq_lens_host:
            # Up projection — tier 3 avoids recompute in bwd
            up = 2 * seq_len * cfg.d_model * cfg.expert_dim
            total += up
            if max_tier >= 3:
                avoided[3] += up
            # Down projection — only needed in fwd, never recomputed
            total += 2 * seq_len * cfg.expert_dim * cfg.d_model
        return ComputeCost(
            total_fwd_flops=total,
            avoided_recompute_flops=tuple(avoided),
        )


def _gelu_tanh_grad(x: torch.Tensor) -> torch.Tensor:
    # Numerically-stable derivative of the tanh-approximation GeLU. For
    # a real impl, use a Triton kernel; vanilla autograd works for the
    # tutorial.
    sqrt_2_over_pi = (2 / torch.pi) ** 0.5
    coef = 0.044715
    inner = sqrt_2_over_pi * (x + coef * x ** 3)
    tanh_inner = torch.tanh(inner)
    sech2 = 1 - tanh_inner ** 2
    inner_grad = sqrt_2_over_pi * (1 + 3 * coef * x ** 2)
    return 0.5 * (1 + tanh_inner) + 0.5 * x * sech2 * inner_grad
```

Then export it from `flextrain/nn/blocks/__init__.py`:

```python
from .ffn_mlp import MLPBlock, MLPConfig
__all__ += ["MLPBlock", "MLPConfig"]
```

Notes on what each method does:

* `fields()` — the engine reads these to allocate one slot tensor
  per chunk per layer. Tier 3 means: when the working-set planner
  picks save level 3, `x_up` is saved and the bwd reuses it; when
  the planner picks lower, `x_up` is unset at bwd entry and the
  layer's `forward_recompute` produces it. See
  [`layer_contract.md`](layer_contract.md#activationfield--activationschema).
* `param_spec()` — the engine uses this to size the parameter,
  master, gradient, and optimizer-state buffers. Names start with
  `w_` so `g_<name>` keys are auto-derived.
* `fwd` — operates on tensors the layer slices and passes in. Don't
  call `torch.empty(...)` here; use `ctx.scratch(shape, dtype)` (the
  engine pools these).
* `bwd` — accumulates `g_1`, `g_2` into `grads` in place. The
  `skip_grads` / `capture_xy` arguments are the LoRA fast-path
  knobs. Mirror `SwiGLUFFN.bwd` if you want the full pattern.
* `compute_cost` — feeds the working-set DP solver. The shape MUST
  be: avoided-recompute is monotone non-decreasing across tiers and
  the last entry doesn't exceed `total_fwd_flops`. The numbers
  literally drive which save level the solver picks per (layer,
  chunk) — see [`../working_set.md`](../working_set.md#how-the-dp-solver-picks-save-levels)
  for what the solver minimizes and how it consumes these numbers.

For the formal block contract, see
[`block_contract.md`](block_contract.md). For why `dx_up.mul_(gelu_grad)`
is in-place, see [`flow.md`](flow.md#scratch-allocations) — saving an
allocation matters when the engine is already at GPU-budget edge.

## Step 2 — Write the new layer (`ParArchBlock`)

Create `flextrain/nn/layers/parch.py`. The layer composes existing
blocks (attention norm, GQA attention, FFN norm, our new MLPBlock)
in **parallel residual** form.

```python
# flextrain/nn/layers/parch.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, MutableMapping
import torch

from flextrain.core.activation_schema import (
    ActivationField, ActivationSchema, concat_fields,
)
from flextrain.core.layer import (
    BackwardIntermediates, ChunkMeta, ComputeCost,
    LayerContext, ParamSpec,
)
from flextrain.nn.blocks import (
    GQAAttentionBlock, GQAAttentionConfig,
    MLPBlock, MLPConfig,
    RMSNormBlock,
)


@dataclass(frozen=True)
class ParArchBlockConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    expert_dim: int
    rms_norm_eps: float = 1e-5
    rope_base: float = 10_000.0
    is_causal: bool = True
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None
    norm_grad_dtype: torch.dtype = torch.float32
    norm_master_dtype: torch.dtype = torch.float32

    def dims(self) -> dict[str, int]:
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "attn_dim": self.n_heads * self.head_dim,
            "kv_dim": self.n_kv_heads * self.head_dim,
            "expert_dim": self.expert_dim,
        }


class ParArchBlock:
    """Parallel-residual layer:
        attn_out = attn(attn_norm(x))
        ffn_out  = ffn(ffn_norm(x))
        x        = x + attn_out + ffn_out
    """

    def __init__(self, layer_id: int, cfg: ParArchBlockConfig):
        self.layer_id = layer_id
        self.cfg = cfg
        self._dims = cfg.dims()

        self.attn_norm = RMSNormBlock(
            prefix="attn_norm", eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.attn = GQAAttentionBlock(GQAAttentionConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
            rope_base=cfg.rope_base, is_causal=cfg.is_causal,
            compute_dtype=cfg.compute_dtype,
            master_dtype=cfg.master_dtype, grad_dtype=cfg.grad_dtype,
        ))
        self.ffn_norm = RMSNormBlock(
            prefix="ffn_norm", eps=cfg.rms_norm_eps,
            param_compute_dtype=cfg.compute_dtype,
            param_master_dtype=cfg.norm_master_dtype,
            param_grad_dtype=cfg.norm_grad_dtype,
        )
        self.ffn = MLPBlock(MLPConfig(           # ← our new block
            d_model=cfg.d_model, expert_dim=cfg.expert_dim,
            compute_dtype=cfg.compute_dtype,
            master_dtype=cfg.master_dtype, grad_dtype=cfg.grad_dtype,
        ))

        x_inp = ActivationField(
            "x_inp", lambda n, d: (n, d["d_model"]),
            cfg.compute_dtype, tier=0,
        )
        self.schema = ActivationSchema(
            fields=concat_fields([
                (x_inp,),
                self.attn_norm.fields(),
                self.attn.fields(),
                self.ffn_norm.fields(),
                self.ffn.fields(),
            ]),
            max_tier=3,
        )
        self.param_spec = ParamSpec.merge([
            self.attn_norm.param_spec(),
            self.attn.param_spec(),
            self.ffn_norm.param_spec(),
            self.ffn.param_spec(),
        ])

    # ------------------------------------------------------------------
    # forward — parallel residual: x + attn(norm_a(x)) + ffn(norm_f(x))
    # ------------------------------------------------------------------
    def forward(self, x, chunk, weights, slot, ctx):
        slot.x_inp.copy_(x)                                    # required: save the residual stream input

        # Branch A: attention
        attn_norm_out = ctx.scratch(x.shape, x.dtype)
        self.attn_norm.fwd(x, weights, slot.attn_norm_rstd, output=attn_norm_out)
        # GQAAttentionBlock.fwd takes (x_resid, attn_norm_output, ...) and returns
        # x_resid + attn_proj_output. For parallel residual we want JUST
        # attn(norm), not x + attn(norm), so call attn against a zero residual:
        zero = ctx.scratch(x.shape, x.dtype).zero_()
        attn_out = self.attn.fwd(zero, attn_norm_out, chunk, weights, slot, ctx)
        # attn_out is now (T, d_model) holding W_O · attn_result.

        # Branch B: FFN
        ffn_norm_out = ctx.scratch(x.shape, x.dtype)
        self.ffn_norm.fwd(x, weights, slot.ffn_norm_rstd, output=ffn_norm_out)
        # MLPBlock.fwd writes residual + W_2·gelu(W_1·norm) into out_tensor.
        # For parallel residual we want JUST ffn(norm), so pass zero residual:
        zero2 = ctx.scratch(x.shape, x.dtype).zero_()
        ffn_out = self.ffn.fwd(ffn_norm_out, weights, zero2, x, slot, ctx)
        # ffn_out aliases x (we passed x as out_tensor) and now holds W_2·gelu(...).

        # Combine: x_out = x_inp + attn_out + ffn_out (in place into x).
        ffn_out.add_(slot.x_inp).add_(attn_out)
        return ffn_out

    # ------------------------------------------------------------------
    # forward_recompute — fill in fields with tier > slot.level
    # ------------------------------------------------------------------
    def forward_recompute(self, slot, chunk, weights, ctx):
        # Recompute attn-side activations (xq/xk/xv/...) and ffn-side x_up
        # from the saved x_inp + tier-0 rstds. Mirror LlamaBlock.forward_recompute
        # for the boilerplate; the divergence is that we have TWO norm outputs
        # (attn_norm + ffn_norm) instead of attn_norm being shared.
        ...

    # ------------------------------------------------------------------
    # backward — delegating shim over the dgrad/wgrad split
    # ------------------------------------------------------------------
    def backward(self, dx, chunk, weights, grads, slot, ctx):
        upstream_dx, inter = self.backward_dgrad(dx, chunk, weights, grads, slot, ctx)
        self.backward_wgrad(inter, weights, grads, slot, ctx)
        return upstream_dx

    def backward_dgrad(self, dx, chunk, weights, grads, slot, ctx, *,
                       skip_target_names=frozenset()):
        # Parallel residual bwd: dx flows into THREE upstream gradients —
        # the residual passthrough, the attn branch, and the ffn branch —
        # and we sum them at the layer input.
        #
        # Mirror LlamaBlock.backward_dgrad for the inline-Wgrad pattern
        # (g_o, g_2, biases, RMSNorm gains accumulate in this method;
        # g_q/g_k/g_v/g_1 are deferred to backward_wgrad because they
        # need a recomputed RMSNorm output as their left operand).
        ...

    def backward_wgrad(self, inter, weights, grads, slot, ctx, *,
                       skip_target_names=frozenset()):
        ...

    # ------------------------------------------------------------------
    # compute_cost — sum block costs at the layer level
    # ------------------------------------------------------------------
    def compute_cost(self, chunk):
        max_tier = self.schema.max_tier
        return ComputeCost.sum([
            self.attn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
            self.attn.compute_cost(chunk, max_tier),
            self.ffn_norm.compute_cost(chunk.total_q, self._dims, max_tier),
            self.ffn.compute_cost(chunk, max_tier),
        ], max_tier=max_tier)
```

The `forward_recompute`, `backward_dgrad`, and `backward_wgrad`
bodies follow the same pattern as `LlamaBlock`'s — keep
`flextrain/nn/layers/llama.py` open in another window when filling
them in. The only structural difference is that the residual sums
THREE branches instead of two, so the dgrad path splits dx three
ways at the layer entry.

Why we call `attn.fwd` with a zero residual and add manually: the
existing `GQAAttentionBlock.fwd` and `MLPBlock.fwd` are written to
absorb a residual into the output via fused addmm — that's the
fastest path for serial residual. In a parallel-residual layer we
want both branches' raw outputs and then to sum them with `x_inp` at
the end. Passing `residual=zero` gives us the raw branch output;
the layer combines them. (A more aggressive impl would fuse the
parallel sum into the second `addmm`, but starting with the simple
pattern is fine — measure before optimizing.)

For the formal layer contract, see
[`layer_contract.md`](layer_contract.md). For the
`ActivationSlot` accessor patterns (`slot.x_inp.copy_(x)`,
`slot.has(name)`, `slot.attn_norm_rstd`), see
[`chunk_contract.md`](chunk_contract.md#activationslot--saved--recomputable-activations).

## Step 3 — Write the new model (arch + builder)

Now the same arch+builder+hook+wire pattern as in
[`tutorial_phi3.md`](tutorial_phi3.md), but instantiating
`ParArchBlock` instead of `LlamaBlock`. Create
`flextrain/io/arch/parch.py`:

```python
# flextrain/io/arch/parch.py
from typing import Any, Mapping
import torch

from flextrain.api import BuildContext, register_block_builder
from flextrain.nn.layers.parch import ParArchBlock, ParArchBlockConfig
from flextrain.nn.layers.lora_wrapper import LoRAWrapperLayer
from ..hf_weights import ArchSpec, Transform, WeightMapEntry, register_arch


PARCH_ARCH = ArchSpec(
    hf_arch_ids=("ParArchForCausalLM",),
    embed=(
        WeightMapEntry("w_tok_embeddings",
                       "model.embed_tokens.weight", Transform.NONE),
    ),
    head=(
        WeightMapEntry("w_final_norm",
                       "model.norm.weight", Transform.NONE),
        WeightMapEntry("w_head_proj",
                       "lm_head.weight", Transform.TRANSPOSE),
    ),
    layer=(
        WeightMapEntry("w_attn_norm",
                       "model.layers.{i}.input_layernorm.weight",
                       Transform.NONE),
        WeightMapEntry("w_ffn_norm",
                       "model.layers.{i}.post_attention_layernorm.weight",
                       Transform.NONE),
        WeightMapEntry("w_q",
                       "model.layers.{i}.self_attn.q_proj.weight",
                       Transform.TRANSPOSE),
        WeightMapEntry("w_k",
                       "model.layers.{i}.self_attn.k_proj.weight",
                       Transform.TRANSPOSE),
        WeightMapEntry("w_v",
                       "model.layers.{i}.self_attn.v_proj.weight",
                       Transform.TRANSPOSE),
        WeightMapEntry("w_o",
                       "model.layers.{i}.self_attn.o_proj.weight",
                       Transform.TRANSPOSE),
        WeightMapEntry("w_1",
                       "model.layers.{i}.mlp.up_proj.weight",
                       Transform.TRANSPOSE),
        WeightMapEntry("w_2",
                       "model.layers.{i}.mlp.down_proj.weight",
                       Transform.TRANSPOSE),
    ),
)
register_arch(PARCH_ARCH)


def hf_config_to_flextrain(hf_config: Mapping[str, Any]) -> dict[str, int]:
    n_heads = int(hf_config["num_attention_heads"])
    n_kv_heads = int(hf_config.get("num_key_value_heads", n_heads))
    head_dim = int(hf_config.get(
        "head_dim", hf_config["hidden_size"] // n_heads
    ))
    return {
        "d_model": int(hf_config["hidden_size"]),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "attn_dim": n_heads * head_dim,
        "kv_dim": n_kv_heads * head_dim,
        "expert_dim": int(hf_config["intermediate_size"]),
        "vocab_size": int(hf_config["vocab_size"]),
        "n_layers": int(hf_config["num_hidden_layers"]),
    }


def hf_config_to_hyperparams(hf_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "rms_norm_eps": float(hf_config.get("rms_norm_eps", 1e-5)),
        "rope_base": float(hf_config.get("rope_theta", 10_000.0)),
    }


def _parch_block_builder(layer_idx: int, ctx: BuildContext):
    hp = ctx.hyperparams
    cfg = ParArchBlockConfig(
        d_model=ctx.dims["d_model"],
        n_heads=ctx.dims["n_heads"],
        n_kv_heads=ctx.dims["n_kv_heads"],
        head_dim=ctx.dims["head_dim"],
        expert_dim=ctx.dims["expert_dim"],
        rms_norm_eps=hp["rms_norm_eps"],
        rope_base=hp["rope_base"],
        compute_dtype=ctx.compute_dtype,
        master_dtype=ctx.master_dtype,
        grad_dtype=ctx.grad_dtype,
        norm_grad_dtype=ctx.norm_grad_dtype,
    )
    layer = ParArchBlock(layer_id=layer_idx, cfg=cfg)
    if ctx.lora_targets:
        layer = LoRAWrapperLayer(
            layer,
            targets=ctx.lora_targets,
            rank=ctx.lora_rank,
            alpha=ctx.lora_alpha,
            adapter_compute_dtype=ctx.lora_adapter_compute_dtype,
            adapter_master_dtype=ctx.lora_adapter_master_dtype,
            adapter_grad_dtype=ctx.lora_adapter_grad_dtype,
        )
    return layer


register_block_builder(("ParArchForCausalLM",), _parch_block_builder)


def post_load_permute(am, hf_config, dims, hyperparams):
    """Q/K halved → pair-interleave for FT's RoPE kernel. Same as Llama."""
    head_dim = int(dims["head_dim"])
    n_heads = int(dims["n_heads"])
    n_kv = int(dims["n_kv_heads"])
    n_layers = int(dims["n_layers"])
    attn_dim = n_heads * head_dim
    kv_dim = n_kv * head_dim

    def _halved_to_pair(dim, head_dim):
        half = head_dim // 2
        out = torch.empty(dim, dtype=torch.int64)
        for h in range(dim // head_dim):
            base = h * head_dim
            for i in range(half):
                out[base + 2 * i] = base + i
                out[base + 2 * i + 1] = base + half + i
        return out

    q_perm = _halved_to_pair(attn_dim, head_dim)
    k_perm = _halved_to_pair(kv_dim, head_dim)
    for L in range(n_layers):
        host = am.buffers.host_params[L]
        host["w_q"].copy_(host["w_q"][:, q_perm])
        host["w_k"].copy_(host["w_k"][:, k_perm])
```

Then one line in `flextrain/io/arch/__init__.py`:

```python
from . import parch  # noqa: F401
```

`ParArchForCausalLM` snake-cases cleanly to `parch`, so no
`_ARCH_MODULE_OVERRIDES` entry is needed.

For the formal arch / builder / `from_pretrained` contract, see
[`model_contract.md`](model_contract.md).

## Step 4 — Test it

The 4-test pyramid is the same regardless of which level(s) you
wrote:

1. **Block-level math parity (most useful when you wrote a new
   block).** Build `MLPBlock` with random weights, run `MLPBlock.fwd`
   on a known input, compare against a hand-rolled reference forward
   in plain PyTorch (`torch.matmul + F.gelu + torch.matmul`).
   Same for `bwd` against `torch.autograd.backward`.
   `tests/moe/test_full_layer_parity.py` is the structural template
   (it does this for MoE; adapt to dense MLP).
2. **Step-0 logit + per-token CE diff vs HuggingFace.** Run
   `tests/test_arch_parity.py ParArch-1B`. Add an entry to its
   model list mirroring an existing arch. The harness drives the
   layer through `from_pretrained` with real HF weights. This
   exercises the layer composition AND the arch + builder + hook.
3. **LoRA-mode parity vs HF PEFT.** `tests/test_arch_lora_e2e.py
   --arch parch-1b --mode lora`. Asserts loss-curve agreement with
   PEFT under the working-set solver's chosen save tier.
4. **HF export round-trip** (only if you'll export). Register a
   pre-export hook if your weight names diverge from HF's; see
   [`../weights.md`](../weights.md) and [`../export.md`](../export.md).

## Common pitfalls (when writing all three)

* **Block bwd's `skip_grads` argument.** Even if you don't need the
  LoRA fast path right now, plumb the arg through — it's dead code
  in full-FT but needed for LoRA-targeted projections later.
  Mirror `SwiGLUFFN.bwd`.
* **Layer's residual structure.** Parallel vs. serial residual is a
  layer-level concern, not a block-level one. Blocks shouldn't
  fuse the residual unless they explicitly take it as an operand
  (like our MLPBlock taking `residual: torch.Tensor`).
* **`ctx.scratch` for branch outputs.** Parallel residual needs
  scratch for the attn-branch and ffn-branch raw outputs.
  Allocate them via `ctx.scratch(shape, dtype)`, not `torch.empty`
  — see [`flow.md`](flow.md#scratch-allocations).
* **`x_inp` MUST be saved** — `slot.x_inp.copy_(x)` in `forward`.
  Both `attn_norm.bwd` and `ffn_norm.bwd` will read it.
* **Recompute identity** — `forward_recompute` MUST produce
  byte-identical output to `forward` for the same `(x, weights)`
  pair, otherwise save-tier invariance breaks. Don't introduce
  randomness in either path.
* **Don't stash on `self`.** Recompute / backward run after
  forward, possibly after the layer has been swapped to host RAM
  and back. `self`-stashed tensors get stranded. Use `slot.aux`
  (intra-layer) or `chunk.extra` (intra-chunk).

## See also

* [`best_practices.md`](best_practices.md) — naming and class-shape
  conventions, how to pick activation tiers, the
  user-responsibility contracts (memory + compute) and the symptom
  → cause table. Read this before claiming a new block or layer
  is done.
* [`tutorial_phi3.md`](tutorial_phi3.md) — composability case.
  Adds Phi-3 by reusing `LlamaBlock` and writing only the arch +
  post-load hook. ~80% of new arches land here; read it after
  you're comfortable with the full ladder above.
* [`flow.md`](flow.md) — engine assumptions + object lifecycle.
  Re-read after writing a new block; some assumptions (recompute
  identity, no-`self`-stash) are easy to break by accident.
* [`block_contract.md`](block_contract.md) — full block convention
  + in-tree catalog with real signatures for each existing block.
* [`layer_contract.md`](layer_contract.md) — Layer Protocol +
  `ActivationField` + `ParamSpec` + `BackwardIntermediates` + I/O
  layers.
* [`chunk_contract.md`](chunk_contract.md) — `ChunkMeta`,
  `LayerContext`, `ActivationSlot`. The `chunk.extra` vs
  `slot.aux` distinction.
* [`model_contract.md`](model_contract.md) — `ArchSpec`,
  `BuildContext`, `register_block_builder`, `from_pretrained`
  dispatch, `ActiveModel` runtime API.
