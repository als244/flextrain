"""8-layer Qwen3-Next E2E loss-curve parity test.

Builds a small random-init "mini-Qwen3-Next" with 8 backbone layers
alternating Qwen3NextLinearLayer (linear-attn + MoE) and Qwen3NextFullLayer
(full-attn + MoE). Compares the loss curve produced by the FlexTrain engine
under several working-set configurations — full residency, partial offload,
aggressive offload, and forced low save level — to a naive PyTorch
reference that follows the same exact math.

Acceptance criteria
-------------------
* Every FT config must produce the same loss curve as every other FT config
  within tight bf16 tolerance (per-step max|Δ| < 0.01).
* FT vs naive PyTorch loss curve must agree within bf16 noise (per-step
  max|Δ| < 0.1 — comfortably below the 1B Llama precedent of ~0.07 over
  100 steps).

Outputs:
* ``parity_results/qwen3_next_8layer/loss_curves.csv``
* Console table.
"""
from __future__ import annotations

import csv
import dataclasses
import os
import sys
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flextrain.bench.parity import (  # noqa: E402
    DTYPE,
    _Seq,
    _flextrain_step,
    _naive_step,
    _rmsnorm,
    _rope_pair_interleave,
)
from tests.test_llama32_1b_parity import _pull_step_batches  # noqa: E402


DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Mini-Qwen3-Next configuration. ALL FT and naive runs must use these.
# ---------------------------------------------------------------------------


D_MODEL = 256
N_LAYERS = 8
LAYER_TYPES = [
    "linear_attention" if (i % 2 == 0) else "full_attention"
    for i in range(N_LAYERS)
]
N_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 32  # full-attn head_dim
LIN_NUM_V_HEADS = 8
LIN_NUM_K_HEADS = 2
LIN_HEAD_K_DIM = 32
LIN_HEAD_V_DIM = 32
LIN_CONV_KERNEL = 4
EXPERT_DIM = 512
NUM_EXPERTS = 4
TOP_K = 2
NUM_SHARED_EXPERTS = 1
SHARED_EXPERT_DIM = 128         # mini-Qwen3-Next: HF default 512 too big
VOCAB_SIZE = 32000
RMS_NORM_EPS = 1e-6
ROPE_BASE = 10_000_000.0
LOAD_BALANCE_COEF = 0.001
ROUTING_MODE = "topk_then_softmax"
PARTIAL_ROTARY_FACTOR = 0.25
ROT_DIM = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)

LR = 3e-4
N_STEPS = 10
TARGET_TOKENS_PER_STEP = 1024
INIT_SEED = 4242


# ---------------------------------------------------------------------------
# Shared-expert + partial-RoPE helpers (used by both naive blocks).
# ---------------------------------------------------------------------------


def _partial_rope(
    x: torch.Tensor, seq_positions: torch.Tensor, base: float, rot_dim: int,
) -> torch.Tensor:
    """Rotate the first ``rot_dim`` channels per head of ``x``; pass-through
    the remaining ``head_dim - rot_dim`` channels."""
    head_dim = x.shape[-1]
    if rot_dim == head_dim:
        return _rope_pair_interleave(x, seq_positions, base)
    # Split, rotate, concat.
    x_rot = x[..., :rot_dim].contiguous()
    x_pass = x[..., rot_dim:].contiguous()
    # Build a temporary "head_dim = rot_dim" tensor for the rotation helper.
    x_rot = _rope_pair_interleave(x_rot, seq_positions, base)
    return torch.cat([x_rot, x_pass], dim=-1)


def _shared_expert(
    h: torch.Tensor,                      # (T, d_model) — input to FFN
    w_shared_up: torch.Tensor,            # (S, d_model, 2 * F_s)
    w_shared_down: torch.Tensor,          # (S, F_s, d_model)
    w_shared_expert_gate: torch.Tensor,   # (d_model, S)
    F_s: int,
) -> torch.Tensor:
    """Reference for shared-expert path. Returns ``(T, d_model)`` shared
    contribution (without the routed-add and without residual)."""
    # x @ w_shared_up: (T, d) × (S, d, 2F) → (T, S, 2F)
    sh_pre = torch.einsum("td,sdf->tsf", h, w_shared_up)
    up_h = sh_pre[..., :F_s]
    gate_h = sh_pre[..., F_s:]
    sh_act = up_h * F.silu(gate_h.float()).to(h.dtype)
    sh_each = torch.einsum("tsf,sfd->tsd", sh_act, w_shared_down)        # (T, S, d)
    sh_gate_pre = h @ w_shared_expert_gate                               # (T, S)
    sh_gate = torch.sigmoid(sh_gate_pre.float()).to(h.dtype)
    return (sh_gate.unsqueeze(-1) * sh_each).sum(dim=1)                  # (T, d)


# ---------------------------------------------------------------------------
# Naive linear-attn reference (matches GatedDeltaNetBlock + outer layer math).
# ---------------------------------------------------------------------------


class NaiveQwen3NextLinearBlock(torch.nn.Module):
    """Linear-attn layer in pure PyTorch.

    Mirrors the FT ``Qwen3NextLinearLayer.forward`` semantics exactly:
    pre-norm, gated-DeltaNet linear attention block, residual, FFN-norm,
    MoE FFN, residual.
    """

    def __init__(self) -> None:
        super().__init__()
        # Norms.
        self.w_attn_norm = torch.nn.Parameter(torch.ones(D_MODEL, dtype=DTYPE))
        self.w_ffn_norm = torch.nn.Parameter(torch.ones(D_MODEL, dtype=DTYPE))
        # Linear-attn projections + tied scalars.
        key_dim = LIN_NUM_K_HEADS * LIN_HEAD_K_DIM
        value_dim = LIN_NUM_V_HEADS * LIN_HEAD_V_DIM
        proj_qkvz = 2 * key_dim + 2 * value_dim
        proj_ba = 2 * LIN_NUM_V_HEADS
        conv_dim = 2 * key_dim + value_dim
        self.w_lin_qkvz = torch.nn.Parameter(
            torch.zeros(D_MODEL, proj_qkvz, dtype=DTYPE)
        )
        self.w_lin_ba = torch.nn.Parameter(
            torch.zeros(D_MODEL, proj_ba, dtype=DTYPE)
        )
        self.w_lin_out = torch.nn.Parameter(
            torch.zeros(value_dim, D_MODEL, dtype=DTYPE)
        )
        self.w_lin_conv = torch.nn.Parameter(
            torch.zeros(conv_dim, 1, LIN_CONV_KERNEL, dtype=DTYPE)
        )
        self.w_lin_dt_bias = torch.nn.Parameter(
            torch.ones(LIN_NUM_V_HEADS, dtype=DTYPE)
        )
        self.w_lin_A_log = torch.nn.Parameter(
            torch.zeros(LIN_NUM_V_HEADS, dtype=DTYPE)
        )
        self.w_lin_norm = torch.nn.Parameter(
            torch.ones(LIN_HEAD_V_DIM, dtype=DTYPE)
        )
        # MoE FFN — routed top-K experts.
        self.w_router = torch.nn.Parameter(
            torch.zeros(D_MODEL, NUM_EXPERTS, dtype=DTYPE)
        )
        self.w_up = torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, D_MODEL, 2 * EXPERT_DIM, dtype=DTYPE)
        )
        self.w_down = torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, EXPERT_DIM, D_MODEL, dtype=DTYPE)
        )
        # Shared-expert path (always-on, sigmoid-gated). Qwen3-Next: S=1.
        self.w_shared_up = torch.nn.Parameter(
            torch.zeros(NUM_SHARED_EXPERTS, D_MODEL, 2 * SHARED_EXPERT_DIM, dtype=DTYPE)
        )
        self.w_shared_down = torch.nn.Parameter(
            torch.zeros(NUM_SHARED_EXPERTS, SHARED_EXPERT_DIM, D_MODEL, dtype=DTYPE)
        )
        self.w_shared_expert_gate = torch.nn.Parameter(
            torch.zeros(D_MODEL, NUM_SHARED_EXPERTS, dtype=DTYPE)
        )

    def _split_qkvz(self, qkvz: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        T = qkvz.shape[0]
        H = LIN_NUM_K_HEADS
        HV = LIN_NUM_V_HEADS
        hk = LIN_HEAD_K_DIM
        hv = LIN_HEAD_V_DIM
        grp = HV // H
        qkvz = qkvz.view(T, H, 2 * hk + 2 * grp * hv)
        q, k, v_grp, z_grp = torch.split(
            qkvz, [hk, hk, grp * hv, grp * hv], dim=-1,
        )
        v = v_grp.reshape(T, HV, hv)
        z = z_grp.reshape(T, HV, hv)
        return q, k, v, z

    def _split_ba(self, ba: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = ba.shape[0]
        H = LIN_NUM_K_HEADS
        HV = LIN_NUM_V_HEADS
        grp = HV // H
        ba = ba.view(T, H, 2 * grp)
        b, a = torch.split(ba, [grp, grp], dim=-1)
        return b.reshape(T, HV), a.reshape(T, HV)

    def _linear_attn(self, x: torch.Tensor) -> torch.Tensor:
        """Pure-torch + FLA reference for the GatedDeltaNetBlock fwd."""
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        T = x.shape[0]
        H = LIN_NUM_K_HEADS
        HV = LIN_NUM_V_HEADS
        hk = LIN_HEAD_K_DIM
        hv = LIN_HEAD_V_DIM
        key_dim = H * hk
        value_dim = HV * hv
        conv_dim = 2 * key_dim + value_dim

        qkvz = x @ self.w_lin_qkvz
        ba = x @ self.w_lin_ba
        q_pre, k_pre, v_pre, z = self._split_qkvz(qkvz)
        b, a = self._split_ba(ba)

        q_flat = q_pre.reshape(T, key_dim)
        k_flat = k_pre.reshape(T, key_dim)
        v_flat = v_pre.reshape(T, value_dim)
        conv_in = torch.cat([q_flat, k_flat, v_flat], dim=-1)

        K = LIN_CONV_KERNEL
        cx = conv_in.transpose(0, 1).unsqueeze(0)
        post_conv = F.conv1d(
            cx, self.w_lin_conv, bias=None,
            padding=K - 1, groups=conv_dim,
        )[..., :T]
        post_conv = F.silu(post_conv).squeeze(0).transpose(0, 1).contiguous()
        q_p, k_p, v_p = torch.split(
            post_conv, [key_dim, key_dim, value_dim], dim=-1,
        )
        q_h = q_p.reshape(T, H, hk)
        k_h = k_p.reshape(T, H, hk)
        v_h = v_p.reshape(T, HV, hv)
        if HV // H > 1:
            rep = HV // H
            q_h = q_h.repeat_interleave(rep, dim=1)
            k_h = k_h.repeat_interleave(rep, dim=1)

        a_f32 = a.float()
        A_log = self.w_lin_A_log.float()
        dt_bias = self.w_lin_dt_bias.float()
        g = -A_log.exp() * F.softplus(a_f32 + dt_bias)
        beta = b.float().sigmoid().to(x.dtype)

        o, _ = chunk_gated_delta_rule(
            q_h.unsqueeze(0), k_h.unsqueeze(0), v_h.unsqueeze(0),
            g.unsqueeze(0), beta.unsqueeze(0),
            scale=hk ** -0.5, initial_state=None,
            output_final_state=False, cu_seqlens=None,
            use_qk_l2norm_in_kernel=False,
        )
        o = o.squeeze(0)

        # Gated RMSNorm on (T, HV, hv).
        o_f = o.float()
        rms = (o_f * o_f).mean(dim=-1, keepdim=True).add_(RMS_NORM_EPS).rsqrt_()
        normed = (o_f * rms).to(x.dtype)
        o_norm = normed * self.w_lin_norm * F.silu(z.float()).to(x.dtype)
        return o_norm.reshape(T, value_dim) @ self.w_lin_out

    def _moe_ffn(self, h2: torch.Tensor) -> torch.Tensor:
        """Naive top-K + softmax MoE SwiGLU. ``topk_then_softmax`` mode.

        Mirrors :class:`tests.test_qwen3_moe_engine_parity.NaiveQwen3MoEBlock`'s
        FFN math.
        """
        router_logits = h2 @ self.w_router
        topk_vals, topk_ids = torch.topk(router_logits, k=TOP_K, dim=-1)
        topk_w = torch.softmax(topk_vals.float(), dim=-1).to(DTYPE)
        out = torch.zeros_like(h2)
        for e in range(NUM_EXPERTS):
            mask_e = (topk_ids == e)
            if not mask_e.any():
                continue
            tk_pos = mask_e.nonzero(as_tuple=False)
            t_idx = tk_pos[:, 0]
            k_idx = tk_pos[:, 1]
            h_e = h2[t_idx]
            up_e = h_e @ self.w_up[e]
            up, gate = up_e.chunk(2, dim=-1)
            act = F.silu(gate.float()).to(DTYPE) * up
            down = act @ self.w_down[e]
            scale_w = topk_w[t_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, t_idx, down * scale_w)
        return out

    def forward(self, x: torch.Tensor, seq_positions: torch.Tensor) -> torch.Tensor:
        h = _rmsnorm(x, self.w_attn_norm, RMS_NORM_EPS)
        x_after_attn = x + self._linear_attn(h)
        h2 = _rmsnorm(x_after_attn, self.w_ffn_norm, RMS_NORM_EPS)
        routed = self._moe_ffn(h2)
        shared = _shared_expert(
            h2, self.w_shared_up, self.w_shared_down,
            self.w_shared_expert_gate, SHARED_EXPERT_DIM,
        )
        return x_after_attn + routed + shared


class NaiveQwen3NextFullBlock(torch.nn.Module):
    """Full-attn layer in pure PyTorch — Qwen3-MoE-style with per-head QK-norm.

    Same as :class:`tests.test_qwen3_moe_engine_parity.NaiveQwen3MoEBlock`
    inlined, with our shared dims.
    """

    def __init__(self) -> None:
        super().__init__()
        attn_dim = N_HEADS * HEAD_DIM
        kv_dim = N_KV_HEADS * HEAD_DIM
        self.w_attn_norm = torch.nn.Parameter(torch.ones(D_MODEL, dtype=DTYPE))
        # w_q is DOUBLED: first half → query, second half → output gate
        # (Qwen3-Next / 3.5 / 3.6 attention output gate).
        self.w_q = torch.nn.Parameter(
            torch.zeros(D_MODEL, 2 * attn_dim, dtype=DTYPE)
        )
        self.w_k = torch.nn.Parameter(torch.zeros(D_MODEL, kv_dim, dtype=DTYPE))
        self.w_v = torch.nn.Parameter(torch.zeros(D_MODEL, kv_dim, dtype=DTYPE))
        self.w_o = torch.nn.Parameter(torch.zeros(attn_dim, D_MODEL, dtype=DTYPE))
        self.w_q_norm = torch.nn.Parameter(torch.ones(HEAD_DIM, dtype=DTYPE))
        self.w_k_norm = torch.nn.Parameter(torch.ones(HEAD_DIM, dtype=DTYPE))
        self.w_ffn_norm = torch.nn.Parameter(torch.ones(D_MODEL, dtype=DTYPE))
        self.w_router = torch.nn.Parameter(
            torch.zeros(D_MODEL, NUM_EXPERTS, dtype=DTYPE)
        )
        self.w_up = torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, D_MODEL, 2 * EXPERT_DIM, dtype=DTYPE)
        )
        self.w_down = torch.nn.Parameter(
            torch.zeros(NUM_EXPERTS, EXPERT_DIM, D_MODEL, dtype=DTYPE)
        )
        # Shared-expert path (Qwen3-Next: S=1, always-on, sigmoid-gated).
        self.w_shared_up = torch.nn.Parameter(
            torch.zeros(NUM_SHARED_EXPERTS, D_MODEL, 2 * SHARED_EXPERT_DIM, dtype=DTYPE)
        )
        self.w_shared_down = torch.nn.Parameter(
            torch.zeros(NUM_SHARED_EXPERTS, SHARED_EXPERT_DIM, D_MODEL, dtype=DTYPE)
        )
        self.w_shared_expert_gate = torch.nn.Parameter(
            torch.zeros(D_MODEL, NUM_SHARED_EXPERTS, dtype=DTYPE)
        )

    def _rmsnorm_head(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        rms = (x_f * x_f).mean(dim=-1, keepdim=True).add_(RMS_NORM_EPS).rsqrt_()
        return (x_f * rms).to(x.dtype) * w

    def _moe_ffn(self, h2: torch.Tensor) -> torch.Tensor:
        router_logits = h2 @ self.w_router
        topk_vals, topk_ids = torch.topk(router_logits, k=TOP_K, dim=-1)
        topk_w = torch.softmax(topk_vals.float(), dim=-1).to(DTYPE)
        out = torch.zeros_like(h2)
        for e in range(NUM_EXPERTS):
            mask_e = (topk_ids == e)
            if not mask_e.any():
                continue
            tk_pos = mask_e.nonzero(as_tuple=False)
            t_idx = tk_pos[:, 0]
            k_idx = tk_pos[:, 1]
            h_e = h2[t_idx]
            up_e = h_e @ self.w_up[e]
            up, gate = up_e.chunk(2, dim=-1)
            act = F.silu(gate.float()).to(DTYPE) * up
            down = act @ self.w_down[e]
            scale_w = topk_w[t_idx, k_idx].unsqueeze(-1)
            out.index_add_(0, t_idx, down * scale_w)
        return out

    def forward(self, x: torch.Tensor, seq_positions: torch.Tensor) -> torch.Tensor:
        attn_dim = N_HEADS * HEAD_DIM
        h = _rmsnorm(x, self.w_attn_norm, RMS_NORM_EPS)
        # Doubled Q projection → split into query and output gate.
        qproj = h @ self.w_q                                  # (T, 2*attn_dim)
        xq_flat = qproj[..., :attn_dim]                       # (T, attn_dim)
        gate_flat = qproj[..., attn_dim:]                     # (T, attn_dim)
        xq = xq_flat.view(-1, N_HEADS, HEAD_DIM)
        xk = (h @ self.w_k).view(-1, N_KV_HEADS, HEAD_DIM)
        xv = (h @ self.w_v).view(-1, N_KV_HEADS, HEAD_DIM)
        xq = self._rmsnorm_head(xq, self.w_q_norm)
        xk = self._rmsnorm_head(xk, self.w_k_norm)
        # Partial RoPE: rotate first ROT_DIM channels per head only.
        rope_q = _partial_rope(xq, seq_positions, ROPE_BASE, ROT_DIM)
        rope_k = _partial_rope(xk, seq_positions, ROPE_BASE, ROT_DIM)
        T, H, D = rope_q.shape
        H_kv = rope_k.shape[1]
        if H_kv != H:
            rep = H // H_kv
            rope_k = rope_k.repeat_interleave(rep, dim=1)
            xv = xv.repeat_interleave(rep, dim=1)
        q_ = rope_q.transpose(0, 1).float()
        k_ = rope_k.transpose(0, 1).float()
        v_ = xv.transpose(0, 1).float()
        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(q_, k_.transpose(-2, -1)) * scale
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1,
        )
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_).transpose(0, 1).to(x.dtype).contiguous()
        attn_flat = attn_out.reshape(T, -1)
        # Apply sigmoid output gate element-wise BEFORE w_o projection.
        gated = attn_flat * torch.sigmoid(gate_flat.float()).to(attn_flat.dtype)
        x_after_attn = x + gated @ self.w_o
        h2 = _rmsnorm(x_after_attn, self.w_ffn_norm, RMS_NORM_EPS)
        routed = self._moe_ffn(h2)
        shared = _shared_expert(
            h2, self.w_shared_up, self.w_shared_down,
            self.w_shared_expert_gate, SHARED_EXPERT_DIM,
        )
        return x_after_attn + routed + shared


class NaiveQwen3NextModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w_tok_embeddings = torch.nn.Parameter(
            torch.zeros(VOCAB_SIZE, D_MODEL, dtype=DTYPE)
        )
        blocks: list[torch.nn.Module] = []
        for lt in LAYER_TYPES:
            if lt == "linear_attention":
                blocks.append(NaiveQwen3NextLinearBlock())
            else:
                blocks.append(NaiveQwen3NextFullBlock())
        self.blocks = torch.nn.ModuleList(blocks)
        self.w_final_norm = torch.nn.Parameter(torch.ones(D_MODEL, dtype=DTYPE))
        self.w_head_proj = torch.nn.Parameter(
            torch.zeros(D_MODEL, VOCAB_SIZE, dtype=DTYPE)
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        seq_positions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        x = self.w_tok_embeddings[token_ids, :]
        for b in self.blocks:
            x = b(x, seq_positions)
        x = _rmsnorm(x, self.w_final_norm, RMS_NORM_EPS)
        logits = x @ self.w_head_proj
        return F.cross_entropy(logits.float(), labels, reduction="sum")


# ---------------------------------------------------------------------------
# Init helper. Same seed → identical init across naive and FT.
# ---------------------------------------------------------------------------


def _init_naive_model(seed: int) -> NaiveQwen3NextModel:
    """Initialize the naive model with deterministic, well-conditioned
    random weights. The FT engine's host buffers will be copied from the
    same naive instance so the two start bit-identical (up to bf16
    rounding when host_params is bf16, which it is)."""
    torch.manual_seed(seed)
    m = NaiveQwen3NextModel().to(DEVICE)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() >= 2:
                p.normal_(mean=0.0, std=0.02)
            elif "norm" in name and "A_log" not in name:
                # Norm weights: ~1 + tiny perturbation.
                p.copy_(torch.ones_like(p) + 0.01 * torch.randn_like(p))
            elif "A_log" in name:
                p.copy_(
                    torch.log(
                        torch.empty_like(p, dtype=torch.float32)
                        .uniform_(1.0, 16.0)
                    ).to(p.dtype)
                )
            elif "dt_bias" in name:
                p.copy_(torch.ones_like(p))
            else:
                p.zero_()
    return m


# ---------------------------------------------------------------------------
# FlexTrain engine builder.
# ---------------------------------------------------------------------------


def _build_ft_engine(
    *,
    n_gpu_layers: int,
    n_gpu_grads: int,
    n_gpu_opt_layers: int,
    force_saved_act_level: int | None,
):
    from flextrain.core.save_level import HardwareCost
    from flextrain.core.working_set import WorkingSetConfig
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.qwen3_next import (
        Qwen3NextLayerConfig, build_qwen3_next_backbone,
    )
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    cfg = Qwen3NextLayerConfig(
        d_model=D_MODEL,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        expert_dim=EXPERT_DIM,
        num_experts=NUM_EXPERTS, top_k=TOP_K,
        num_shared_experts=NUM_SHARED_EXPERTS,
        shared_expert_dim=SHARED_EXPERT_DIM,
        partial_rotary_factor=PARTIAL_ROTARY_FACTOR,
        linear_num_v_heads=LIN_NUM_V_HEADS,
        linear_num_k_heads=LIN_NUM_K_HEADS,
        linear_head_k_dim=LIN_HEAD_K_DIM,
        linear_head_v_dim=LIN_HEAD_V_DIM,
        linear_conv_kernel=LIN_CONV_KERNEL,
        rms_norm_eps=RMS_NORM_EPS,
        rope_base=ROPE_BASE,
        is_causal=True,
        load_balance_coef=LOAD_BALANCE_COEF,
        routing_mode=ROUTING_MODE,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
        norm_master_dtype=torch.float32,
    )
    backbone = build_qwen3_next_backbone(cfg, LAYER_TYPES)
    embed = TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    ))
    head = LMHead(LMHeadConfig(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE,
        rms_norm_eps=RMS_NORM_EPS, head_chunk_size=128,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    ))
    key_dim = LIN_NUM_K_HEADS * LIN_HEAD_K_DIM
    value_dim = LIN_NUM_V_HEADS * LIN_HEAD_V_DIM
    dims = dict(
        d_model=D_MODEL,
        n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        attn_dim=N_HEADS * HEAD_DIM,
        kv_dim=N_KV_HEADS * HEAD_DIM,
        expert_dim=EXPERT_DIM,
        vocab_size=VOCAB_SIZE,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        num_v_heads=LIN_NUM_V_HEADS,
        num_k_heads=LIN_NUM_K_HEADS,
        head_k_dim=LIN_HEAD_K_DIM,
        head_v_dim=LIN_HEAD_V_DIM,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_dim=2 * key_dim + value_dim,
        proj_qkvz_dim=2 * key_dim + 2 * value_dim,
        proj_ba_dim=2 * LIN_NUM_V_HEADS,
        conv_kernel_size=LIN_CONV_KERNEL,
    )
    max_seq_len = 1024
    working_set = WorkingSetConfig(
        target_round_tokens=TARGET_TOKENS_PER_STEP,
        max_chunk_size=max_seq_len,
        max_training_chunks=8,
        max_total_round_tokens=TARGET_TOKENS_PER_STEP,
        target_num_rounds=1,
        n_gpu_layers=n_gpu_layers,
        n_gpu_grads=n_gpu_grads,
        n_gpu_opt_layers=n_gpu_opt_layers,
        gpu_act_buffer_size=int(2.0 * (1 << 30)),
        host_act_buffer_size=int(2.0 * (1 << 30)),
        available_gpu_memory_bytes=int(20 * (1 << 30)),
        available_host_memory_bytes=int(40 * (1 << 30)),
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=max_seq_len, hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(
        peak_tflops=60.0, pcie_bw_gbps=20.0, practical_efficiency_factor=1.0,
    )
    opt = AdamW(
        AdamWHyperparams(
            lr=LR, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
        ),
        state_dtype=torch.bfloat16,
    )
    am = ActiveModel(
        embed=embed, backbone=backbone, head=head,
        optimizer=opt, working_set=working_set, hw_cost=hw_cost,
        dims=dims, device=DEVICE,
        force_saved_act_level=force_saved_act_level,
    )
    return am


def _copy_naive_to_ft(naive: NaiveQwen3NextModel, am) -> None:
    """Copy naive PyTorch weights into FT host buffers so the engine
    starts from the exact same init."""
    with torch.no_grad():
        am.buffers.host_embed_params["w_tok_embeddings"].copy_(
            naive.w_tok_embeddings.detach().cpu()
        )
        am.buffers.host_head_params["w_final_norm"].copy_(
            naive.w_final_norm.detach().cpu()
        )
        am.buffers.host_head_params["w_head_proj"].copy_(
            naive.w_head_proj.detach().cpu()
        )
        for i, lt in enumerate(LAYER_TYPES):
            block = naive.blocks[i]
            hp = am.buffers.host_params[i]
            if lt == "linear_attention":
                names = (
                    "w_attn_norm", "w_ffn_norm",
                    "w_lin_qkvz", "w_lin_ba", "w_lin_out", "w_lin_conv",
                    "w_lin_dt_bias", "w_lin_A_log", "w_lin_norm",
                    "w_router", "w_up", "w_down",
                    "w_shared_up", "w_shared_down", "w_shared_expert_gate",
                )
            else:
                names = (
                    "w_attn_norm", "w_ffn_norm",
                    "w_q", "w_k", "w_v", "w_o",
                    "w_q_norm", "w_k_norm",
                    "w_router", "w_up", "w_down",
                    "w_shared_up", "w_shared_down", "w_shared_expert_gate",
                )
            for name in names:
                src = getattr(block, name).detach().cpu()
                hp[name].copy_(src)
    am._refresh_gpu_residents()
    for name, dev_t in am.buffers.gpu_head_params.items():
        dev_t.copy_(am.buffers.host_head_params[name])
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Run drivers.
# ---------------------------------------------------------------------------


def _run_naive(step_batches) -> list[float]:
    print("\n=== NAIVE PyTorch reference ===")
    naive = _init_naive_model(seed=INIT_SEED)
    opt = torch.optim.AdamW(
        naive.parameters(), lr=LR,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
    )
    curve = []
    t0 = time.time()
    for step, batch in enumerate(step_batches):
        seqs = []
        for s in batch:
            ns = _Seq(s.tokens.clone())
            ns.targets = s.targets.clone()
            seqs.append(ns)
        ts = time.time()
        loss = _naive_step(naive, opt, seqs, DEVICE)
        curve.append(loss)
        print(
            f"  naive step {step:2d}: loss={loss:.4f}  "
            f"step={(time.time()-ts)*1000:.0f}ms  "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
    del naive, opt
    import gc; gc.collect()
    torch.cuda.empty_cache()
    return curve


def _run_ft(label: str, ws_kwargs: dict, step_batches) -> list[float]:
    print(f"\n=== FlexTrain ({label}) ===")
    print(
        "  config:",
        ", ".join(f"{k}={v}" for k, v in ws_kwargs.items()),
    )
    am = _build_ft_engine(**ws_kwargs)
    print(f"  built engine: {len(am.backbone)} layers")
    # Sanity: print which save level got chosen for the FORCED case.
    if ws_kwargs["force_saved_act_level"] is not None:
        print(
            f"  forcing save level = {ws_kwargs['force_saved_act_level']} "
            f"(host slots used)"
        )
    naive_init = _init_naive_model(seed=INIT_SEED)
    _copy_naive_to_ft(naive_init, am)
    del naive_init

    curve = []
    t0 = time.time()
    for step, batch in enumerate(step_batches):
        seqs = []
        for s in batch:
            ns = _Seq(s.tokens.clone())
            ns.targets = s.targets.clone()
            seqs.append(ns)
        ts = time.time()
        loss = _flextrain_step(am, seqs)
        curve.append(loss)
        print(
            f"  FT({label}) step {step:2d}: loss={loss:.4f}  "
            f"step={(time.time()-ts)*1000:.0f}ms  "
            f"elapsed={time.time()-t0:.1f}s  "
            f"max_alloc={torch.cuda.max_memory_allocated()/(1<<30):.2f}GiB",
            flush=True,
        )
    am.buffers.destroy()
    del am
    import gc; gc.collect()
    try:
        from flextrain.engine import unregister_all_process_pinned_memory
        unregister_all_process_pinned_memory()
    except Exception:
        pass
    torch.cuda.empty_cache()
    return curve


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def _build_step_batches() -> list[list[_Seq]]:
    """MathInstruct batches via the Llama-3.2-1B tokenizer; clamp ids
    into [0, VOCAB_SIZE) so the embedding lookup is in-bounds."""
    hf_tok_path = os.path.join(ROOT, "models", "Llama-3.2-1B")
    if not os.path.isdir(hf_tok_path):
        raise FileNotFoundError(
            f"Need a tokenizer dir at {hf_tok_path} for tokenization."
        )
    raw = _pull_step_batches(
        hf_tok_path, n_steps=N_STEPS,
        target_tokens_per_step=TARGET_TOKENS_PER_STEP,
        min_len=64, max_len=256,
    )
    # Clamp to vocab.
    out = []
    for batch in raw:
        seqs = []
        for s in batch:
            tok = s.tokens.clone().clamp_(0, VOCAB_SIZE - 1)
            tgt = s.targets.clone()
            mask = tgt != -100
            tgt[mask] = tgt[mask].clamp_(0, VOCAB_SIZE - 1)
            ns = _Seq(tok)
            ns.targets = tgt
            seqs.append(ns)
        out.append(seqs)
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")

    out_dir = os.path.join(ROOT, "parity_results", "qwen3_next_8layer")
    os.makedirs(out_dir, exist_ok=True)

    step_batches = _build_step_batches()
    total_tokens = sum(sum(len(s) for s in b) for b in step_batches)
    print(
        f"Prepared {len(step_batches)} steps; "
        f"total {total_tokens} tokens "
        f"(~{total_tokens / max(1, N_STEPS):.0f} tok/step)"
    )

    # Configurations:
    configs: list[tuple[str, dict]] = [
        (
            "ft-full",
            dict(
                n_gpu_layers=N_LAYERS, n_gpu_grads=N_LAYERS,
                n_gpu_opt_layers=N_LAYERS,
                force_saved_act_level=None,
            ),
        ),
        (
            "ft-off4",
            dict(
                n_gpu_layers=4, n_gpu_grads=4, n_gpu_opt_layers=4,
                force_saved_act_level=None,
            ),
        ),
        (
            "ft-off2",
            dict(
                n_gpu_layers=2, n_gpu_grads=2, n_gpu_opt_layers=2,
                force_saved_act_level=None,
            ),
        ),
        (
            "ft-lowsave",
            dict(
                n_gpu_layers=N_LAYERS, n_gpu_grads=N_LAYERS,
                n_gpu_opt_layers=N_LAYERS,
                force_saved_act_level=1,  # drops tier-2 + tier-3 fields
            ),
        ),
    ]

    # 1. Naive run.
    naive_curve = _run_naive(step_batches)

    # 2. Each FT config.
    ft_curves: dict[str, list[float]] = {}
    for label, kw in configs:
        ft_curves[label] = _run_ft(label, kw, step_batches)

    # 3. Print + write results.
    print("\n=== 8-layer Qwen3-Next E2E parity (10 steps) ===")
    header = ["step", "naive"] + list(ft_curves.keys()) + [
        "max|Δ_FT|", "|Δ_naive|"
    ]
    rows = []
    max_delta_ft_overall = 0.0
    max_delta_naive_overall = 0.0
    for i in range(N_STEPS):
        ft_vals = [ft_curves[lbl][i] for lbl in ft_curves]
        d_ft = max(ft_vals) - min(ft_vals)
        d_naive = max(abs(naive_curve[i] - v) for v in ft_vals)
        max_delta_ft_overall = max(max_delta_ft_overall, d_ft)
        max_delta_naive_overall = max(max_delta_naive_overall, d_naive)
        rows.append(
            [str(i), f"{naive_curve[i]:.4f}"]
            + [f"{v:.4f}" for v in ft_vals]
            + [f"{d_ft:.4f}", f"{d_naive:.4f}"]
        )
    # Pretty print.
    col_widths = [max(len(r[c]) for r in [header] + rows) for c in range(len(header))]
    fmt = "  ".join(f"{{:>{w}}}" for w in col_widths)
    print(fmt.format(*header))
    for r in rows:
        print(fmt.format(*r))

    # 4. CSV output.
    csv_path = os.path.join(out_dir, "loss_curves.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "naive"] + list(ft_curves.keys()))
        for i in range(N_STEPS):
            writer.writerow(
                [i, f"{naive_curve[i]:.6f}"]
                + [f"{ft_curves[lbl][i]:.6f}" for lbl in ft_curves]
            )
    print(f"\nCSV: {csv_path}")

    # 5. Summary + assertions.
    print("\nSummary:")
    base_label = "ft-full"
    base = ft_curves[base_label]
    pairwise: list[tuple[str, str, float]] = []
    for lbl, curve in ft_curves.items():
        if lbl == base_label:
            continue
        d = max(abs(a - b) for a, b in zip(base, curve))
        pairwise.append((base_label, lbl, d))
        print(f"  {base_label} vs {lbl:<11} max|Δ| = {d:.4f}")
    d_naive_full = max(abs(a - b) for a, b in zip(naive_curve, base))
    print(f"  {base_label} vs naive       max|Δ| = {d_naive_full:.4f}")

    # Acceptance:
    fail = False
    if max_delta_ft_overall > 0.01:
        print(
            f"\n!! FT cross-config divergence: max|Δ_FT|={max_delta_ft_overall:.4f}"
            f" > 0.01 — engine is NOT deterministic across working-set configs"
        )
        fail = True
    if max_delta_naive_overall > 0.10:
        print(
            f"\n!! FT vs naive divergence: max|Δ_naive|="
            f"{max_delta_naive_overall:.4f} > 0.10"
        )
        fail = True
    if fail:
        raise AssertionError(
            f"max|Δ_FT|={max_delta_ft_overall:.4f}, "
            f"max|Δ_naive|={max_delta_naive_overall:.4f}"
        )
    print(
        f"\n✓ 8-layer Qwen3-Next E2E parity PASSED  "
        f"(max|Δ_FT|={max_delta_ft_overall:.4f}, "
        f"max|Δ_naive|={max_delta_naive_overall:.4f})"
    )


if __name__ == "__main__":
    main()
