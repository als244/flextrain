"""Loss-curve parity harness: naive PyTorch vs FlexTrain.

Public entry point: :func:`run_loss_curve_parity`. It:

1. Builds a naive ``torch.nn.Module`` Llama-style reference model +
   ``torch.optim.AdamW``. Pure-PyTorch, no FlexTrain or orig kernels
   (RoPE uses the pair-interleave convention to match the Triton
   kernel).
2. Pulls N deterministic per-step batches from a FineWeb .bin shard
   (or any other source producing a list[list[Sequence]]).
3. Runs the naive baseline once, records per-step avg loss.
4. For each :class:`WorkingSetSpec`: builds a fresh FlexTrain engine
   with identical init, runs the same step sequence, records loss.
5. Compares trajectories (windowed-mean against naive; cross-config
   against each other).

Reuse
-----
The harness is intentionally configurable so you can exercise it with
different model shapes, data sources, and working-set sweeps without
editing tests/ or the engine:

    from flextrain.bench import (
        ModelShape, WorkingSetSpec, LossCurveParityConfig,
        run_loss_curve_parity,
    )

    cfg = LossCurveParityConfig(
        shape=ModelShape(d_model=512, n_layers=8),
        n_steps=200,
        target_tokens_per_step=1024,
        lr=5e-4,
        working_sets=[...],  # list of WorkingSetSpec
        shard_path="orig/fineweb/fineweb_train_000001.bin",
    )
    result = run_loss_curve_parity(cfg)
    result.print_summary()
    result.assert_all_match(windowed_atol=0.10)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import torch

from flextrain.core.save_level import HardwareCost
from flextrain.core.working_set import WorkingSetConfig


# ---------------------------------------------------------------------------
# Dtype default. bf16 matches orig + FlexTrain.
# ---------------------------------------------------------------------------


DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Model shape.
# ---------------------------------------------------------------------------


@dataclass
class ModelShape:
    """Small-but-realistic Llama-shape used for parity runs.

    Defaults give a ~20M-param model that's big enough to learn real
    structure in 100 steps but small enough to run fast. Override to
    stress other sizes.
    """

    d_model: int = 512  # = n_heads * head_dim
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 64  # flash-attn preferred
    expert_dim: int = 1024
    # GPT-2 vocab is 50257. Rounded up to 50432 (multiple of 256) for
    # matmul tile efficiency.
    vocab_size: int = 50432
    rms_norm_eps: float = 1e-5
    rope_base: float = 10000.0


# ---------------------------------------------------------------------------
# Working-set specification (typed per-config input).
# ---------------------------------------------------------------------------


@dataclass
class WorkingSetSpec:
    label: str
    n_gpu_layers: int
    n_gpu_grads: int
    n_gpu_opt_layers: int
    gpu_act_buffer_size: int
    host_act_buffer_size: int
    max_chunk_size: int
    target_round_tokens: int
    max_total_round_tokens: int
    max_training_chunks: int


# ---------------------------------------------------------------------------
# Naive pure-PyTorch reference model.
# ---------------------------------------------------------------------------


def _rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    x_fp = x.float()
    rstd = torch.rsqrt(x_fp.pow(2).mean(-1, keepdim=True) + eps)
    return (x_fp * rstd).to(x.dtype) * w


def _rope_pair_interleave(
    x: torch.Tensor, seq_positions: torch.Tensor, theta: float
) -> torch.Tensor:
    """Pair-interleave RoPE (matches the Triton kernel; NOT the HF
    halved-split convention — see docs/internal/NOTES.md [FINDING 6])."""
    T, H, D = x.shape
    half = D // 2
    pos = seq_positions.view(-1, 1, 1).float()
    exponent = 2.0 * torch.arange(0, half, device=x.device).float() / D
    inv_freq = theta ** (-exponent)
    angles = pos * inv_freq
    cos = angles.cos()
    sin = angles.sin()
    x_fp = x.float()
    even = x_fp[..., 0::2]
    odd = x_fp[..., 1::2]
    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos
    out = torch.empty_like(x_fp)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out.to(x.dtype)


class NaiveLlamaBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        expert_dim: int,
        rms_norm_eps: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base

        attn_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        self.w_attn_norm = torch.nn.Parameter(torch.ones(d_model, dtype=DTYPE))
        self.w_q = torch.nn.Parameter(torch.zeros(d_model, attn_dim, dtype=DTYPE))
        self.w_k = torch.nn.Parameter(torch.zeros(d_model, kv_dim, dtype=DTYPE))
        self.w_v = torch.nn.Parameter(torch.zeros(d_model, kv_dim, dtype=DTYPE))
        self.w_o = torch.nn.Parameter(torch.zeros(attn_dim, d_model, dtype=DTYPE))

        self.w_ffn_norm = torch.nn.Parameter(torch.ones(d_model, dtype=DTYPE))
        self.w_1 = torch.nn.Parameter(torch.zeros(d_model, expert_dim, dtype=DTYPE))
        self.w_2 = torch.nn.Parameter(torch.zeros(expert_dim, d_model, dtype=DTYPE))
        self.w_3 = torch.nn.Parameter(torch.zeros(d_model, expert_dim, dtype=DTYPE))

    def forward(
        self, x: torch.Tensor, seq_positions: torch.Tensor
    ) -> torch.Tensor:
        h = _rmsnorm(x, self.w_attn_norm, self.rms_norm_eps)
        xq = (h @ self.w_q).view(-1, self.n_heads, self.head_dim)
        xk = (h @ self.w_k).view(-1, self.n_kv_heads, self.head_dim)
        xv = (h @ self.w_v).view(-1, self.n_kv_heads, self.head_dim)
        rope_q = _rope_pair_interleave(xq, seq_positions, self.rope_base)
        rope_k = _rope_pair_interleave(xk, seq_positions, self.rope_base)

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
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_).transpose(0, 1).to(x.dtype).contiguous()
        attn_flat = attn_out.reshape(T, -1)
        x_after_attn = x + attn_flat @ self.w_o

        h2 = _rmsnorm(x_after_attn, self.w_ffn_norm, self.rms_norm_eps)
        x1 = h2 @ self.w_1
        x3 = h2 @ self.w_3
        mlp = (torch.nn.functional.silu(x1.float()).to(x1.dtype) * x3) @ self.w_2
        return x_after_attn + mlp


class NaiveQwen3Block(torch.nn.Module):
    """Qwen3-dense block: Llama + QK-norm (RMSNorm per-head on Q and K
    after projection, before RoPE). No attention bias.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        expert_dim: int,
        rms_norm_eps: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_base = rope_base

        attn_dim = n_heads * head_dim
        kv_dim = n_kv_heads * head_dim

        self.w_attn_norm = torch.nn.Parameter(torch.ones(d_model, dtype=DTYPE))
        self.w_q_norm = torch.nn.Parameter(torch.ones(head_dim, dtype=DTYPE))
        self.w_k_norm = torch.nn.Parameter(torch.ones(head_dim, dtype=DTYPE))
        self.w_q = torch.nn.Parameter(torch.zeros(d_model, attn_dim, dtype=DTYPE))
        self.w_k = torch.nn.Parameter(torch.zeros(d_model, kv_dim, dtype=DTYPE))
        self.w_v = torch.nn.Parameter(torch.zeros(d_model, kv_dim, dtype=DTYPE))
        self.w_o = torch.nn.Parameter(torch.zeros(attn_dim, d_model, dtype=DTYPE))

        self.w_ffn_norm = torch.nn.Parameter(torch.ones(d_model, dtype=DTYPE))
        self.w_1 = torch.nn.Parameter(torch.zeros(d_model, expert_dim, dtype=DTYPE))
        self.w_2 = torch.nn.Parameter(torch.zeros(expert_dim, d_model, dtype=DTYPE))
        self.w_3 = torch.nn.Parameter(torch.zeros(d_model, expert_dim, dtype=DTYPE))

    def forward(
        self, x: torch.Tensor, seq_positions: torch.Tensor
    ) -> torch.Tensor:
        h = _rmsnorm(x, self.w_attn_norm, self.rms_norm_eps)
        xq = (h @ self.w_q).view(-1, self.n_heads, self.head_dim)
        xk = (h @ self.w_k).view(-1, self.n_kv_heads, self.head_dim)
        xv = (h @ self.w_v).view(-1, self.n_kv_heads, self.head_dim)
        # Per-head QK-norm BEFORE RoPE (Qwen3-specific).
        xq = _rmsnorm(xq, self.w_q_norm, self.rms_norm_eps)
        xk = _rmsnorm(xk, self.w_k_norm, self.rms_norm_eps)
        rope_q = _rope_pair_interleave(xq, seq_positions, self.rope_base)
        rope_k = _rope_pair_interleave(xk, seq_positions, self.rope_base)

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
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(probs, v_).transpose(0, 1).to(x.dtype).contiguous()
        attn_flat = attn_out.reshape(T, -1)
        x_after_attn = x + attn_flat @ self.w_o

        h2 = _rmsnorm(x_after_attn, self.w_ffn_norm, self.rms_norm_eps)
        x1 = h2 @ self.w_1
        x3 = h2 @ self.w_3
        mlp = (torch.nn.functional.silu(x1.float()).to(x1.dtype) * x3) @ self.w_2
        return x_after_attn + mlp


class NaiveQwen3Model(torch.nn.Module):
    """Pure-PyTorch reference — Qwen3-dense family."""

    def __init__(self, shape: "ModelShape") -> None:
        super().__init__()
        self.w_tok_embeddings = torch.nn.Parameter(
            torch.zeros(shape.vocab_size, shape.d_model, dtype=DTYPE)
        )
        self.blocks = torch.nn.ModuleList(
            [
                NaiveQwen3Block(
                    shape.d_model, shape.n_heads, shape.n_kv_heads,
                    shape.head_dim, shape.expert_dim,
                    shape.rms_norm_eps, shape.rope_base,
                )
                for _ in range(shape.n_layers)
            ]
        )
        self.w_final_norm = torch.nn.Parameter(
            torch.ones(shape.d_model, dtype=DTYPE)
        )
        self.w_head_proj = torch.nn.Parameter(
            torch.zeros(shape.d_model, shape.vocab_size, dtype=DTYPE)
        )
        self.rms_norm_eps = shape.rms_norm_eps

    def forward(
        self,
        token_ids: torch.Tensor,
        seq_positions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        x = self.w_tok_embeddings[token_ids, :]
        for block in self.blocks:
            x = block(x, seq_positions)
        x = _rmsnorm(x, self.w_final_norm, self.rms_norm_eps)
        logits = x @ self.w_head_proj
        return torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="sum"
        )


class NaiveLlamaModel(torch.nn.Module):
    """Pure-PyTorch reference (embed + N × Llama block + RMSNorm + head).

    Loss: sum-reduce cross-entropy in fp32. The caller divides grads
    by ``total_tokens`` before stepping so the effective gradient is
    the per-token mean — matching FlexTrain's
    ``loss_scale_factor=1/total_tokens`` convention.
    """

    def __init__(self, shape: ModelShape) -> None:
        super().__init__()
        self.w_tok_embeddings = torch.nn.Parameter(
            torch.zeros(shape.vocab_size, shape.d_model, dtype=DTYPE)
        )
        self.blocks = torch.nn.ModuleList(
            [
                NaiveLlamaBlock(
                    shape.d_model, shape.n_heads, shape.n_kv_heads,
                    shape.head_dim, shape.expert_dim,
                    shape.rms_norm_eps, shape.rope_base,
                )
                for _ in range(shape.n_layers)
            ]
        )
        self.w_final_norm = torch.nn.Parameter(
            torch.ones(shape.d_model, dtype=DTYPE)
        )
        self.w_head_proj = torch.nn.Parameter(
            torch.zeros(shape.d_model, shape.vocab_size, dtype=DTYPE)
        )
        self.rms_norm_eps = shape.rms_norm_eps

    def forward(
        self,
        token_ids: torch.Tensor,
        seq_positions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        x = self.w_tok_embeddings[token_ids, :]
        for block in self.blocks:
            x = block(x, seq_positions)
        x = _rmsnorm(x, self.w_final_norm, self.rms_norm_eps)
        logits = x @ self.w_head_proj
        loss = torch.nn.functional.cross_entropy(
            logits.float(), labels, reduction="sum"
        )
        return loss


# ---------------------------------------------------------------------------
# Initialization: identical across naive and FlexTrain.
# ---------------------------------------------------------------------------


def _fill_weights(
    rng: torch.Generator,
    shape: tuple[int, ...],
    std: float = 0.02,
    *,
    ones: bool = False,
    device: str = "cuda:0",
) -> torch.Tensor:
    if ones:
        return torch.ones(shape, dtype=DTYPE, device=device)
    out = torch.randn(
        shape, generator=rng, dtype=torch.float32, device=device
    ) * std
    return out.to(DTYPE)


def _init_naive_model(
    model: NaiveLlamaModel, *, seed: int, device: str
) -> None:
    rng = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        model.w_tok_embeddings.copy_(
            _fill_weights(rng, tuple(model.w_tok_embeddings.shape), device=device)
        )
        for block in model.blocks:
            block.w_attn_norm.copy_(
                _fill_weights(rng, tuple(block.w_attn_norm.shape),
                              ones=True, device=device)
            )
            for name in ("w_q", "w_k", "w_v", "w_o", "w_1", "w_2", "w_3"):
                p = getattr(block, name)
                p.copy_(_fill_weights(rng, tuple(p.shape), device=device))
            block.w_ffn_norm.copy_(
                _fill_weights(rng, tuple(block.w_ffn_norm.shape),
                              ones=True, device=device)
            )
        model.w_final_norm.copy_(
            _fill_weights(rng, tuple(model.w_final_norm.shape),
                          ones=True, device=device)
        )
        model.w_head_proj.copy_(
            _fill_weights(rng, tuple(model.w_head_proj.shape), device=device)
        )


def _copy_naive_to_flextrain(naive: NaiveLlamaModel, bm) -> None:
    with torch.no_grad():
        bm.host_embed_params["w_tok_embeddings"].copy_(
            naive.w_tok_embeddings.detach()
        )
        for i, block in enumerate(naive.blocks):
            dst = bm.host_params[i]
            for name in (
                "w_attn_norm", "w_q", "w_k", "w_v", "w_o",
                "w_ffn_norm", "w_1", "w_2", "w_3",
            ):
                dst[name].copy_(getattr(block, name).detach())
        bm.host_head_params["w_final_norm"].copy_(naive.w_final_norm.detach())
        bm.host_head_params["w_head_proj"].copy_(naive.w_head_proj.detach())


# ---------------------------------------------------------------------------
# Sequence + data stream.
# ---------------------------------------------------------------------------


class _Seq:
    """Minimal duck-typed Sequence for the scheduler + optional loss_mask."""

    def __init__(
        self, tokens: torch.Tensor, *,
        loss_mask: torch.Tensor | None = None,
    ) -> None:
        self.tokens = tokens
        self.targets = torch.roll(tokens, -1)
        self.per_token_loss = torch.zeros(len(tokens), dtype=torch.float32)
        self.loss_mask = loss_mask  # None => train on all positions
        self.seq_id = 0

    def __len__(self) -> int:
        return len(self.tokens)


class FineWebDocStream:
    """Yields documents from a FineWeb .bin shard. See orig/fineweb.py
    for the format (256 int32 header + uint16 tokens, EOT = 50256)."""

    EOT = 50256

    def __init__(
        self,
        shard_path: str,
        min_len: int = 32,
        max_len: int = 512,
    ) -> None:
        import numpy as np

        if not os.path.isfile(shard_path):
            raise FileNotFoundError(
                f"FineWeb shard not found: {shard_path}. "
                f"Run orig/fineweb.py to download + tokenize."
            )
        self._arr = np.fromfile(shard_path, dtype=np.uint16, offset=256 * 4)
        self._min_len = min_len
        self._max_len = max_len
        self._cursor = 0

    def next_doc(self) -> torch.Tensor:
        import numpy as np

        arr = self._arr
        while self._cursor < len(arr):
            end = self._cursor
            while end < len(arr) and arr[end] != self.EOT:
                end += 1
            if end > self._cursor + 1:
                raw = arr[self._cursor : end]
                if len(raw) > 0 and raw[0] == self.EOT:
                    raw = raw[1:]
                if len(raw) >= self._min_len:
                    take = min(len(raw), self._max_len)
                    doc = raw[:take].astype(np.int64)
                    self._cursor = end + 1
                    return torch.from_numpy(doc.copy())
            self._cursor = end + 1
        raise StopIteration("shard exhausted")

    def reset(self) -> None:
        self._cursor = 0


def _generate_sequence_stream(
    shard_path: str,
    n_steps: int,
    target_tokens_per_step: int,
    min_len: int,
    max_len: int,
) -> list[list[_Seq]]:
    stream = FineWebDocStream(shard_path, min_len=min_len, max_len=max_len)
    step_seqs = []
    for _ in range(n_steps):
        batch, total = [], 0
        while total < target_tokens_per_step:
            try:
                tokens = stream.next_doc()
            except StopIteration:
                stream.reset()
                tokens = stream.next_doc()
            batch.append(_Seq(tokens))
            total += len(tokens)
        step_seqs.append(batch)
    return step_seqs


# ---------------------------------------------------------------------------
# Step runners.
# ---------------------------------------------------------------------------


def _naive_step(
    model,
    opt: torch.optim.Optimizer,
    seqs: list[_Seq],
    device: str,
) -> float:
    """Forward + backward + step for one gradient-accumulation batch.

    Loss masking is done by the caller via ``targets == -100`` (the
    PyTorch CE convention, natively honored by ``cross_entropy``).
    We divide grads by the count of *active* (non-ignored) targets to
    match FlexTrain's ``loss_scale_factor=1/total_tokens`` semantics.
    """
    opt.zero_grad(set_to_none=False)
    total_loss = 0.0
    total_active = 0
    for s in seqs:
        tokens = s.tokens.to(device)
        labels = s.targets.to(device)
        positions = torch.arange(len(tokens), device=device, dtype=torch.int32)
        active = int((labels != -100).sum().item())
        loss = model(tokens, positions, labels)
        loss.backward()
        total_loss += float(loss.item())
        total_active += active
    if total_active == 0:
        return 0.0
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(total_active)
    opt.step()
    return total_loss / total_active


def _flextrain_step(am, seqs: list[_Seq]) -> float:
    """Loss-scale normalized by ACTIVE targets (honoring ``targets ==
    -100`` ignore-index). Per-token loss values returned by
    ``fwd_bwd`` are zero on ignored positions
    (``flextrain.nn.loss.CrossEntropyLoss`` applies the mask); we
    average over the active count to match naive's reporting.
    """
    active = 0
    for s in seqs:
        active += int((s.targets != -100).sum().item())
    if active == 0:
        return 0.0
    stats = am.fwd_bwd(
        seqs, loss_scale_factor=1.0 / active, verbose=False,
    )
    am.step()
    # stats.total_loss is the sum of per_token_loss values; ignored
    # positions have 0 loss already. Divide by active count.
    return stats.total_loss / active


# ---------------------------------------------------------------------------
# FlexTrain engine builder.
# ---------------------------------------------------------------------------


def _build_flextrain_engine(
    shape: ModelShape, ws_spec: WorkingSetSpec, lr: float, device: str,
):
    from flextrain.engine.active_model import ActiveModel
    from flextrain.nn.embed import TokenEmbedConfig, TokenEmbedLayer
    from flextrain.nn.head import LMHead, LMHeadConfig
    from flextrain.nn.layers.llama import LlamaBlock, LlamaBlockConfig
    from flextrain.optim.adamw import AdamW, AdamWHyperparams

    cfg = LlamaBlockConfig(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim,
        rms_norm_eps=shape.rms_norm_eps, rope_base=shape.rope_base,
        is_causal=True,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    )
    backbone = [LlamaBlock(layer_id=i, cfg=cfg) for i in range(shape.n_layers)]
    embed = TokenEmbedLayer(TokenEmbedConfig(
        vocab_size=shape.vocab_size, d_model=shape.d_model,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
    ))
    head = LMHead(LMHeadConfig(
        d_model=shape.d_model, vocab_size=shape.vocab_size,
        rms_norm_eps=shape.rms_norm_eps, head_chunk_size=64,
        compute_dtype=DTYPE, master_dtype=DTYPE, grad_dtype=DTYPE,
        norm_grad_dtype=torch.float32,
    ))
    dims = dict(
        d_model=shape.d_model, n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads, head_dim=shape.head_dim,
        expert_dim=shape.expert_dim, vocab_size=shape.vocab_size,
    )
    working_set = WorkingSetConfig(
        target_round_tokens=ws_spec.target_round_tokens,
        max_chunk_size=ws_spec.max_chunk_size,
        max_training_chunks=ws_spec.max_training_chunks,
        max_total_round_tokens=ws_spec.max_total_round_tokens,
        target_num_rounds=1,
        n_gpu_layers=ws_spec.n_gpu_layers,
        n_gpu_grads=ws_spec.n_gpu_grads,
        n_gpu_opt_layers=ws_spec.n_gpu_opt_layers,
        gpu_act_buffer_size=ws_spec.gpu_act_buffer_size,
        host_act_buffer_size=ws_spec.host_act_buffer_size,
        available_gpu_memory_bytes=1 << 32,
        available_host_memory_bytes=1 << 34,
        leeway_gpu_memory_bytes=0, leeway_host_memory_bytes=0,
        max_seq_len=ws_spec.max_chunk_size * max(1, ws_spec.max_training_chunks),
        hardware_env={}, raw={},
    )
    hw_cost = HardwareCost(
        peak_tflops=10.0, pcie_bw_gbps=10.0, practical_efficiency_factor=1.0,
    )
    opt = AdamW(AdamWHyperparams(
        lr=lr, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0,
    ))
    return ActiveModel(
        embed=embed, backbone=backbone, head=head,
        optimizer=opt, working_set=working_set, hw_cost=hw_cost,
        dims=dims, device=device,
    )


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


@dataclass
class LossCurveParityConfig:
    shape: ModelShape = field(default_factory=ModelShape)
    working_sets: list[WorkingSetSpec] = field(default_factory=list)
    n_steps: int = 100
    target_tokens_per_step: int = 384
    min_seq_len: int = 64
    max_seq_len: int = 256
    lr: float = 5e-4
    init_seed: int = 4242
    shard_path: str = ""  # must be set by caller
    device: str = "cuda:0"


@dataclass
class LossCurveParityResult:
    config: LossCurveParityConfig
    naive_curve: list[float]
    ft_curves: dict[str, list[float]]  # label -> curve

    def print_summary(self) -> None:
        """Print final losses (avg of last 3 steps) for naive + every
        FlexTrain config, plus the delta against naive."""
        print("\n" + "=" * 72)
        print("  LOSS CURVE PARITY SUMMARY")
        print("=" * 72)
        print(f"  {self.config.n_steps} steps, "
              f"~{self.config.target_tokens_per_step} tokens/step, "
              f"lr={self.config.lr}, d_model={self.config.shape.d_model}, "
              f"n_layers={self.config.shape.n_layers}")
        naive_first = sum(self.naive_curve[:3]) / 3
        naive_last = sum(self.naive_curve[-3:]) / 3
        print(
            f"\n  {'config':<48} {'final loss':>12} {'Δ vs naive':>12}"
        )
        print("  " + "-" * 72)
        print(
            f"  {'naive PyTorch baseline':<48} "
            f"{naive_last:>12.4f} {'':>12}"
        )
        for label, curve in self.ft_curves.items():
            last = sum(curve[-3:]) / 3
            delta = last - naive_last
            print(
                f"  {label:<48} {last:>12.4f} {delta:>+12.4f}"
            )
        print(
            f"\n  naive first-3-avg -> last-3-avg: "
            f"{naive_first:.4f} -> {naive_last:.4f}"
        )
        print("=" * 72)

    def assert_all_match(
        self,
        *,
        window: int = 10,
        windowed_atol: float = 0.10,
        cross_config_atol: float | None = None,
    ) -> None:
        """Assert every FT config's windowed-mean curve agrees with
        naive within ``windowed_atol`` AND with every other FT config
        within ``cross_config_atol`` (default ``2 × windowed_atol``).

        Why two tolerances
        ------------------
        Two FT configs each drift from naive independently by bf16
        noise. Their cross-config delta can be up to ~2× either side's
        drift against naive (triangle inequality in the pessimistic
        limit). The "vs naive" check is the hard correctness signal;
        "cross-config" is the sanity check that scheduling decisions
        don't compound in unexpected ways.

        Raises AssertionError with a diagnostic on failure.
        """
        if cross_config_atol is None:
            cross_config_atol = 2.0 * windowed_atol

        def _windowed(curve: list[float]) -> list[float]:
            n = len(curve)
            out = []
            for i in range(n):
                lo = max(0, i - window + 1)
                out.append(sum(curve[lo : i + 1]) / (i - lo + 1))
            return out

        def _max_delta(a: list[float], b: list[float]) -> float:
            wa = _windowed(a)
            wb = _windowed(b)
            return max(abs(x - y) for x, y in zip(wa, wb))

        failures = []
        # vs naive (stricter tolerance)
        for label, curve in self.ft_curves.items():
            d = _max_delta(self.naive_curve, curve)
            ok = d < windowed_atol
            marker = "  " if ok else "!!"
            print(
                f"  [{marker}] {label} vs naive: "
                f"{window}-step windowed max |delta| = {d:.4f} "
                f"(atol {windowed_atol})"
            )
            if not ok:
                failures.append((label, "naive", d, windowed_atol))
        # cross-config (looser tolerance — 2× the vs-naive limit)
        labels = list(self.ft_curves.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = labels[i], labels[j]
                d = _max_delta(self.ft_curves[a], self.ft_curves[b])
                ok = d < cross_config_atol
                marker = "  " if ok else "!!"
                print(
                    f"  [{marker}] {a} vs {b}: "
                    f"{window}-step windowed max |delta| = {d:.4f} "
                    f"(atol {cross_config_atol})"
                )
                if not ok:
                    failures.append((a, b, d, cross_config_atol))
        if failures:
            raise AssertionError(
                f"{len(failures)} curve(s) exceeded tolerance: " + ", ".join(
                    f"{a}<->{b} (|Δ|={d:.4f} > {tol:.4f})"
                    for a, b, d, tol in failures
                )
            )


def run_loss_curve_parity(
    config: LossCurveParityConfig,
) -> LossCurveParityResult:
    """Run the naive baseline + every working-set config sequentially,
    return the :class:`LossCurveParityResult`.

    Use-site shape::

        result = run_loss_curve_parity(cfg)
        result.print_summary()
        result.assert_all_match(windowed_atol=0.1)
    """
    if not config.shard_path:
        raise ValueError("config.shard_path must be set")
    if not torch.cuda.is_available():
        raise RuntimeError("loss-curve parity requires CUDA")

    # 1. Data stream.
    step_seqs = _generate_sequence_stream(
        config.shard_path,
        n_steps=config.n_steps,
        target_tokens_per_step=config.target_tokens_per_step,
        min_len=config.min_seq_len,
        max_len=config.max_seq_len,
    )
    actual_total = sum(sum(len(s) for s in batch) for batch in step_seqs)
    print(
        f"\n  Data: {config.n_steps} steps, "
        f"~{config.target_tokens_per_step} tokens/step "
        f"(actual total={actual_total}), seq lens in "
        f"[{config.min_seq_len}, {config.max_seq_len}]"
    )

    # 2. Naive baseline.
    print("  Running naive PyTorch baseline...")
    naive = NaiveLlamaModel(config.shape).to(config.device)
    _init_naive_model(naive, seed=config.init_seed, device=config.device)
    naive_opt = torch.optim.AdamW(
        naive.parameters(), lr=config.lr, betas=(0.9, 0.95),
        eps=1e-8, weight_decay=0.0,
    )
    naive_curve = []
    for batch in step_seqs:
        seqs = [_Seq(s.tokens.clone()) for s in batch]
        naive_curve.append(_naive_step(naive, naive_opt, seqs, config.device))
    # Keep a frozen copy of the init-state naive model so every FT
    # config starts from the same weights.
    naive_init = NaiveLlamaModel(config.shape).to(config.device)
    _init_naive_model(naive_init, seed=config.init_seed, device=config.device)

    # 3. Each FlexTrain config.
    ft_curves: dict[str, list[float]] = {}
    for ws_spec in config.working_sets:
        print(f"\n  === config: {ws_spec.label} ===")
        am = _build_flextrain_engine(
            config.shape, ws_spec, lr=config.lr, device=config.device,
        )
        _copy_naive_to_flextrain(naive_init, am.buffers)
        for slot_idx in range(ws_spec.n_gpu_layers):
            am.buffers.fetch_layer_params(
                slot_idx, slot_idx, non_blocking=False
            )
        for name, dev_t in am.buffers.gpu_embed_params.items():
            dev_t.copy_(am.buffers.host_embed_params[name])
        for name, dev_t in am.buffers.gpu_head_params.items():
            dev_t.copy_(am.buffers.host_head_params[name])
        torch.cuda.synchronize()

        curve = []
        for batch in step_seqs:
            seqs = [_Seq(s.tokens.clone()) for s in batch]
            curve.append(_flextrain_step(am, seqs))
        ft_curves[ws_spec.label] = curve
        avg_first = sum(curve[:3]) / 3
        avg_last = sum(curve[-3:]) / 3
        print(
            f"    -> ft loss: avg(first 3)={avg_first:.4f}, "
            f"avg(last 3)={avg_last:.4f}"
        )
        am.buffers.destroy()
        del am
        import gc
        gc.collect()
        try:
            from flextrain.engine import unregister_all_process_pinned_memory
            unregister_all_process_pinned_memory()
        except Exception:
            pass

    return LossCurveParityResult(
        config=config, naive_curve=naive_curve, ft_curves=ft_curves,
    )
