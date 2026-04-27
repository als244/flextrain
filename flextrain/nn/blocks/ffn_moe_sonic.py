"""SonicMoE-backed MoE SwiGLU FFN block (future: H100+/B200+).

Mirrors :class:`MoESwiGLUFFN` (which uses orig's per-expert loop via
``flextrain_moe_*`` kernels) but calls into SonicMoE's fused grouped-GEMM
primitives. Requires SM_90+ (Hopper) or SM_100 (Blackwell); will fail
to import on older GPUs.

**NOT RUNNABLE on the 3090 development box** — gated behind
``torch.cuda.get_device_capability()``. Written so the module can
be imported (type-checked, param-spec consumed) but instantiation
of the kernels raises on sm_80 and below.

API mirrors :class:`MoESwiGLUFFN` so layer composition is identical:
swap one for the other at construction time based on hardware.

Design notes — see ``docs/NOTES.md`` [SonicMoE integration plan]:

* Weight layout is DIFFERENT from orig:
  - orig (our current default): ``w_up: (E, H, 2*F)``, ``w_down: (E, F, H)``.
  - sonic:                      ``w_up: (2*F, H, E)``, ``w_down: (H, F, E)``
                                 with stride order (2, 0, 1) — expert
                                 is the LAST dim, not first.
* Bypasses sonic's ``torch.autograd.Function``s entirely — they
  conflict with FlexTrain's custom fwd/bwd contract (we need to
  interleave head-loss-bwd, per-layer fwd_recompute, per-layer bwd,
  all on the engine's stream schedule, without torch autograd ever
  seeing a computation graph). Instead, call sonic's LOWER-LEVEL
  stateless primitives:
  - Forward:  ``_router_forward``, ``_topk_softmax_fwd``,
              ``TC_topk_router_metadata_triton``,
              ``_up_projection_forward``,
              ``_down_projection_forward``.
  - Backward: ``_down_projection_backward_act`` (computes dh + ds),
              ``_down_projection_backward_weight`` (g_down),
              ``_up_projection_backward_act`` (dx_expanded from dh),
              ``_up_projection_backward_weight`` (g_up),
              ``_token_broadcast_backward`` (gather dx_expanded →
              per-token dx_ffn_norm_up),
              ``_topk_softmax_bwd`` (dlogits from ds).
* Saved tensors (from reading sonic's bwd signatures):
  - **tier 0** (always saved, small): router state — ``router_logits
    (T, E) bf16``, ``topk_router_score (T, K) fp32``, ``topk_router_
    indices (T, K) int32``, ``expert_frequency_offset (E+1,) int32``,
    ``x_gather_idx (T*K,) int32``, ``s_scatter_idx (T*K,) int32``,
    ``s_reverse_scatter_idx (T*K,) int32``. All generated on-device
    during fwd via ``TC_topk_router_metadata_triton``.
  - **tier 3** (main save): ``a_prime (T*K, 2*F) bf16`` — sonic's
    pre-activation, equivalent to orig's ``x_up``. Consumed by
    ``_down_projection_backward_weight`` (weight grad of w_down).
  - NOT saved (always recomputed at bwd time, matching the
    existing dense-Llama pattern for ffn_norm_output):
    * ``x`` (the MoE input = ffn_norm_output): recomputed from
      ``slot.xo`` (tier 2, post-attn residual) + ``slot.ffn_norm_rstd``
      (tier 0) inside ``fwd_recompute_a_prime`` and stashed in
      ``slot.aux["sonic_moe_x"]`` for bwd to consume. Same discipline
      as LlamaBlock / Qwen3DenseBlock which recompute
      ``attn_norm_output`` and ``ffn_norm_output`` per-bwd-iter.
    * ``h`` (post-SwiGLU activation, (T*K, F)): sonic's
      ``_down_projection_backward_act`` needs it, and it's derived
      from ``a_prime`` by re-applying the SwiGLU activation.
      Either recompute on-the-fly in ``fwd_recompute_a_prime`` and
      stash in ``slot.aux["sonic_moe_h"]``, or reconstitute inside
      ``bwd`` at the cost of one extra element-wise pass.
  - NO pinned-host expert_counts (sonic's dispatch is device-only;
    grouped GEMM handles variable-length groups).

The fields() declaration below enumerates every saved tensor the
bwd path needs to read. If a future sonic-moe version changes
signatures, update fields() to match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

import torch

from flextrain.core.activation_schema import ActivationField
from flextrain.core.layer import (
    ChunkMeta,
    ComputeCost,
    LayerContext,
    ParamSpec,
    TensorSpec,
)


def _require_sm90():
    """SonicMoE requires SM_90+ (Hopper) or SM_100 (Blackwell)."""
    if not torch.cuda.is_available():
        raise RuntimeError("SonicMoE requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(
            f"SonicMoE requires SM_90+ (Hopper) or SM_100 (Blackwell); "
            f"got SM_{major}{minor}. Use MoESwiGLUFFN (orig-kernel path) "
            "on older GPUs (e.g. A100 SM_80, RTX 3090 SM_86)."
        )


def _sonic_import():
    """Deferred import so the module imports on non-Hopper hosts."""
    try:
        from sonicmoe.functional.forward import (  # type: ignore
            _down_projection_forward,
            _router_forward,
            _topk_softmax_fwd,
            _up_projection_forward,
        )
        from sonicmoe.functional.backward import (  # type: ignore
            _down_projection_backward_act,
            _down_projection_backward_weight,
            _token_broadcast_backward,
            _topk_softmax_bwd,
            _up_projection_backward_act,
            _up_projection_backward_weight,
        )
        from sonicmoe.functional.triton_kernels import (  # type: ignore
            TC_topk_router_metadata_triton,
        )
        from sonicmoe.enums import ActivationType  # type: ignore
        return {
            "up_fwd": _up_projection_forward,
            "down_fwd": _down_projection_forward,
            "topk_softmax_fwd": _topk_softmax_fwd,
            "router_fwd": _router_forward,
            "up_bwd_act": _up_projection_backward_act,
            "up_bwd_weight": _up_projection_backward_weight,
            "down_bwd_act": _down_projection_backward_act,
            "down_bwd_weight": _down_projection_backward_weight,
            "token_broadcast_bwd": _token_broadcast_backward,
            "topk_softmax_bwd": _topk_softmax_bwd,
            "router_metadata": TC_topk_router_metadata_triton,
            "ActivationType": ActivationType,
        }
    except ImportError as e:
        raise ImportError(
            "sonic-moe is not installed. Install via "
            "`pip install sonic-moe`. Requires SM_90+ GPU."
        ) from e


@dataclass(frozen=True)
class MoESwiGLUSonicConfig:
    """Configuration for :class:`MoESwiGLUSonicFFN`.

    Matches :class:`MoESwiGLUConfig` but with sonic-specific knobs:

    * ``norm_topk_probs``: True for Qwen3-MoE-style (re-normalize the
      top-k softmax probs so they sum to 1). False for OLMoE.
    * ``is_softmax_over_topk``: True for OLMoE/TC-style
      (``softmax(topk(logits))``). False for Qwen3-MoE-style
      (``topk(softmax(logits))``).
    * ``concat_layout``: sonic has two up-proj weight formats; the
      interleaved one (default False) is lower memory pressure.
    """

    d_model: int
    expert_dim: int
    num_experts: int
    top_k: int
    load_balance_coef: float = 0.0
    is_softmax_over_topk: bool = True
    norm_topk_probs: bool = False
    concat_layout: bool = False
    compute_dtype: torch.dtype = torch.bfloat16
    master_dtype: torch.dtype | None = None
    grad_dtype: torch.dtype | None = None


class MoESwiGLUSonicFFN:
    """SonicMoE-backed MoE SwiGLU FFN.

    Drop-in replacement for :class:`MoESwiGLUFFN` on Hopper/Blackwell.
    Do NOT instantiate on SM < 9.0 — it will fail to run the CUDA
    kernels.

    **Status: SKELETON ONLY.** The ``fwd`` / ``bwd`` bodies below are
    placeholders documenting the intended dispatch. Fill them in
    once we have H100 CI access. See
    ``docs/NOTES.md`` [SonicMoE integration plan] for the full
    design.
    """

    def __init__(self, cfg: MoESwiGLUSonicConfig) -> None:
        self.cfg = cfg
        # Don't require SM_90 at __init__ — we want to be able to
        # import/construct the block for type-checks and tests that
        # only poke at .param_spec / .fields(). Fail only when fwd runs.

    def fields(self) -> tuple[ActivationField, ...]:
        cfg = self.cfg
        bf = cfg.compute_dtype
        return (
            # Tier-0 router state (compact, always saved):
            ActivationField(
                "topk_router_score",
                lambda n, d: (n, cfg.top_k),
                torch.float32,  # sonic uses fp32 for router scores
                tier=0,
            ),
            ActivationField(
                "topk_router_indices",
                lambda n, d: (n, cfg.top_k),
                torch.int32,
                tier=0,
            ),
            ActivationField(
                "expert_frequency",
                lambda n, d: (cfg.num_experts,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "expert_frequency_offset",
                lambda n, d: (cfg.num_experts + 1,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "x_gather_idx",
                lambda n, d: (n * cfg.top_k,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "s_scatter_idx",
                lambda n, d: (n * cfg.top_k,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "s_reverse_scatter_idx",
                lambda n, d: (n * cfg.top_k,),
                torch.int32,
                tier=0,
                token_axis=None,
            ),
            ActivationField(
                "router_logits",
                lambda n, d: (n, cfg.num_experts),
                bf,
                tier=0,
            ),
            # Tier-3 saved pre-activation (sonic's ``a_prime``,
            # equivalent to orig's ``x_up``):
            ActivationField(
                "a_prime",
                lambda n, d: (n * cfg.top_k, 2 * cfg.expert_dim),
                bf,
                tier=3,
                token_axis=None,
            ),
        )

    def param_spec(self) -> ParamSpec:
        """Sonic weight layout: w_up is ``(2*F, d, E)``, w_down is
        ``(d, F, E)``, with stride order (2, 0, 1) — expert is the
        LAST dim. Different from orig (which has E first)."""
        cfg = self.cfg
        return ParamSpec(
            tensors=(
                TensorSpec(
                    "w_router",
                    lambda d: (cfg.num_experts, cfg.d_model),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "w_up",
                    # sonic: (2*F, H, E) stride (2, 0, 1)
                    lambda d: (2 * cfg.expert_dim, cfg.d_model, cfg.num_experts),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
                TensorSpec(
                    "w_down",
                    # sonic: (H, F, E) stride (2, 0, 1)
                    lambda d: (cfg.d_model, cfg.expert_dim, cfg.num_experts),
                    compute_dtype=cfg.compute_dtype,
                    master_dtype=cfg.master_dtype,
                    grad_dtype=cfg.grad_dtype,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Compute — SKELETON. Bodies to land once H100 CI is available.
    # ------------------------------------------------------------------

    def fwd(
        self,
        ffn_norm_output: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        attn_output_with_residual: torch.Tensor,
        out_tensor: torch.Tensor,
        slot,
        ctx: LayerContext,
        chunk: ChunkMeta,
        *,
        layer_id: int,
    ) -> torch.Tensor:
        """SonicMoE forward. Writes router state to tier-0 slot
        fields, a_prime to tier-3.

        Dispatches sonic's ``_router_forward`` + ``_topk_softmax_fwd``
        + ``TC_topk_router_metadata_triton`` + ``_up_projection_forward``
        + ``_down_projection_forward`` directly, saving intermediates
        on the slot so ``bwd`` can consume them. Mirrors the inlined
        logic of ``moe_TC_softmax_topk_layer`` but with slot-based
        saves instead of autograd.Function saves.
        """
        _require_sm90()
        k = _sonic_import()
        cfg = self.cfg

        # 1) Router logits.
        router_logits = slot.router_logits
        k["router_fwd"](
            ffn_norm_output, weights["w_router"], router_logits,
        )
        # 2) Top-k + softmax. Writes topk_router_score, topk_router_indices.
        k["topk_softmax_fwd"](
            router_logits,
            slot.topk_router_score,
            slot.topk_router_indices,
            cfg.num_experts, cfg.top_k,
            is_softmax_over_topk=cfg.is_softmax_over_topk,
            norm_topk_probs=cfg.norm_topk_probs,
        )
        # 3) Routing metadata (gather/scatter indices, expert frequency).
        k["router_metadata"](
            slot.topk_router_indices, cfg.num_experts,
            slot.expert_frequency,
            slot.expert_frequency_offset,
            slot.x_gather_idx,
            slot.s_scatter_idx,
            slot.s_reverse_scatter_idx,
        )
        # 4) Up projection + activation. Writes a_prime (2F scatter-
        # order) and h (F post-activation). h is internal to sonic;
        # not saved. a_prime is our tier-3 save.
        T = ffn_norm_output.shape[0]
        TK = T * cfg.top_k
        activation_type = k["ActivationType"].SWIGLU
        a, h = k["up_fwd"](
            ffn_norm_output, weights["w_up"], None,  # b1
            slot.expert_frequency_offset,
            TK, cfg.top_k,
            slot.x_gather_idx,
            slot.s_scatter_idx,
            slot.s_reverse_scatter_idx,
            None,  # expert_frequency (unused for fixed top-k)
            False,  # is_each_token_has_variable_activated_expert
            activation_type,
            False,  # is_inference_mode_enabled
            cfg.concat_layout,
            out_a=slot.a_prime,
        )
        # 5) Down projection + weighted gather. Writes final output
        # into `out_tensor` (adding the residual).
        output = k["down_fwd"](
            a, h, weights["w_down"], None,  # b2
            slot.topk_router_score,
            slot.expert_frequency_offset,
            T, cfg.top_k,
            slot.x_gather_idx,
            slot.s_scatter_idx,
            slot.s_reverse_scatter_idx,
            None, False,
            activation_type,
            residual=attn_output_with_residual,
            out=out_tensor,
        )
        return output

    def bwd(
        self,
        dy_resid: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        grads: MutableMapping[str, torch.Tensor],
        slot,
        ctx: LayerContext,
        chunk: ChunkMeta,
        *,
        layer_id: int,
    ) -> torch.Tensor:
        """SonicMoE backward. Accumulates ``g_router / g_up / g_down``
        and returns ``dx_ffn_norm_up``.

        Sequence of sonic-moe primitive calls (no torch.autograd):

        1. ``_down_projection_backward_act`` — reads (dy_resid, h, w_down,
           a_prime, topk_scores, expert_frequency_offset, x_gather_idx,
           s_scatter_idx); produces dh (pre-activation grad, (TK, 2F))
           and ds (per-slot router-weight grad, (TK,)).
        2. ``_down_projection_backward_weight`` — reads (dy_resid,
           a_prime, expert_frequency_offset, x_gather_idx); accumulates
           into g_down ((H, F, E) layout via .permute(2,0,1)).
        3. ``_up_projection_backward_act`` — reads (w_up, dh,
           expert_frequency_offset); produces dx_expanded ((TK, H))
           per-slot gradient.
        4. ``_up_projection_backward_weight`` — reads (x, dh,
           expert_frequency_offset, x_gather_idx); accumulates into
           g_up ((2F, H, E) layout via .permute(2,1,0)).
        5. ``_token_broadcast_backward`` — reads (dx_expanded,
           s_reverse_scatter_idx); produces dx_reduced (T, H) by
           summing the top-K contributions per token. (Partial — also
           contributes ``dlogits`` row for the router.)
        6. ``_topk_softmax_bwd`` — reads (router_logits, dtopk_score,
           topk_router_score, topk_router_indices); produces dlogits_full
           (T, E).
        7. Router weight grad + FFN-norm-upstream: dlogits @ w_router.T +
           g_router += x.T @ dlogits. Matches orig's pattern at lines
           680-683.
        """
        _require_sm90()
        k = _sonic_import()
        cfg = self.cfg
        is_glu = True  # SwiGLU only for now
        activation_str = "swiglu"

        # Need x (ffn_norm_output) recomputed by caller via
        # fwd_recompute_a_prime which also stashes it.
        x = slot.aux.get("sonic_moe_x", None)
        if x is None:
            raise RuntimeError(
                "MoESwiGLUSonicFFN.bwd requires fwd_recompute_a_prime "
                "to have stashed 'sonic_moe_x' in slot.aux."
            )
        # ``h`` (post-SwiGLU) isn't saved — recompute from a_prime by
        # applying SwiGLU manually. For SwiGLU: h = silu(gate) * up,
        # where a_prime = [gate, up] interleaved (or concat).
        # For now we delegate the SwiGLU to a small helper or stash
        # it at fwd_recompute_a_prime for simplicity.
        h = slot.aux.pop("sonic_moe_h", None)
        if h is None:
            raise RuntimeError(
                "sonic_moe_h scratch not found; fwd_recompute_a_prime "
                "must compute and stash h = SwiGLU(a_prime)."
            )

        T = dy_resid.shape[0]
        TK = T * cfg.top_k
        device = dy_resid.device
        dh = torch.empty(TK, 2 * cfg.expert_dim, dtype=dy_resid.dtype, device=device)
        ds = torch.zeros(TK, dtype=dy_resid.dtype, device=device)

        # 1. Down-proj bwd (act path). Writes dh (2F) and ds.
        k["down_bwd_act"](
            dy_resid, h, weights["w_down"], dh, ds, None, None,
            slot.a_prime, slot.topk_router_score,
            slot.expert_frequency_offset,
            slot.x_gather_idx, slot.s_scatter_idx,
            activation_str,
        )
        # 2. Down-proj bwd (weight path). Accumulates into g_down.
        k["down_bwd_weight"](
            dy_resid, slot.a_prime, grads["g_down"],
            slot.expert_frequency_offset, slot.x_gather_idx,
        )
        # 3. Up-proj bwd (act path). Writes dx_expanded.
        dx_expanded = torch.empty(TK, cfg.d_model, dtype=dy_resid.dtype, device=device)
        k["up_bwd_act"](
            weights["w_up"], dx_expanded, dh, None,
            slot.expert_frequency_offset, is_glu,
            concat_layout=cfg.concat_layout,
        )
        # 4. Up-proj bwd (weight path). Accumulates into g_up.
        k["up_bwd_weight"](
            x, grads["g_up"], dh,
            slot.expert_frequency_offset, slot.x_gather_idx,
            is_glu, concat_layout=cfg.concat_layout,
        )
        # 5. Token broadcast bwd: gather dx_expanded → (T, H)
        # ffn_norm_upstream contribution from the data path.
        ffn_norm_upstream = torch.zeros(T, cfg.d_model, dtype=dy_resid.dtype, device=device)
        k["token_broadcast_bwd"](
            ffn_norm_upstream, dx_expanded,
            slot.s_reverse_scatter_idx,
            None,  # num_activated_expert_per_token_offset — None for fixed top-K
            cfg.top_k, cfg.d_model,
            False,  # is_varlen_K
        )
        # 6. Top-k softmax bwd: dlogits from ds. ds was reshaped (T, K)
        # by moe_TC_softmax_topk_layer; we do it here.
        dtopk_score = ds.view(T, cfg.top_k)
        dlogits_full = torch.zeros(T, cfg.num_experts, dtype=dy_resid.dtype, device=device)
        k["topk_softmax_bwd"](
            slot.router_logits, dlogits_full, None, dtopk_score,
            slot.topk_router_score, slot.topk_router_indices,
            cfg.num_experts, cfg.top_k,
            is_softmax_over_topk=cfg.is_softmax_over_topk,
            norm_topk_probs=cfg.norm_topk_probs,
        )
        # 7. Router weight grad + accumulate downstream FFN-norm-upstream.
        # ffn_norm_upstream += dlogits_full @ w_router
        # g_router += x.T @ dlogits_full
        # Note: sonic w_router is (E, H), so use w_router directly on the right.
        ffn_norm_upstream.addmm_(dlogits_full, weights["w_router"])
        grads["g_router"].addmm_(dlogits_full.T, x)

        return ffn_norm_upstream

    def fwd_recompute_a_prime(
        self,
        ffn_norm_output: torch.Tensor,
        weights: Mapping[str, torch.Tensor],
        slot,
        chunk: ChunkMeta,
        *,
        layer_id: int,
    ) -> None:
        """Refill tier-3 ``slot.a_prime`` by re-running the up-proj,
        AND stash ``x`` (ffn_norm_output) and ``h`` (post-SwiGLU)
        in ``slot.aux`` for ``bwd`` to consume.

        Called by the enclosing layer's ``forward_recompute`` at bwd
        time. The caller (e.g. ``OLMoESonicBlock``) has already
        recomputed ``ffn_norm_output`` from ``slot.xo`` +
        ``slot.ffn_norm_rstd`` and passes it here.

        Mirrors the bwd-inputs requirement of sonic's bwd
        primitives:
        * ``_up_projection_backward_weight`` reads ``x`` →
          ``slot.aux["sonic_moe_x"]``.
        * ``_down_projection_backward_act`` reads ``h`` →
          ``slot.aux["sonic_moe_h"]``.
        * Both bwd-weight calls read ``a_prime`` → ``slot.a_prime``
          (the only tier-3 save for sonic MoE).
        """
        _require_sm90()
        k = _sonic_import()
        cfg = self.cfg
        T = ffn_norm_output.shape[0]
        TK = T * cfg.top_k

        # Stash x for up-projection-backward-weight (unchanged across
        # fwd / bwd — scatter happens inside sonic's kernel via
        # x_gather_idx, we just pass the per-token x).
        slot.aux["sonic_moe_x"] = ffn_norm_output

        # Re-run up projection → slot.a_prime (pre-activation) AND h
        # (post-activation). Both come out of _up_projection_forward.
        activation_type = k["ActivationType"].SWIGLU
        h = torch.empty(TK, cfg.expert_dim, dtype=ffn_norm_output.dtype,
                        device=ffn_norm_output.device)
        k["up_fwd"](
            ffn_norm_output, weights["w_up"], None,  # b1 None
            slot.expert_frequency_offset,
            TK, cfg.top_k,
            slot.x_gather_idx, slot.s_scatter_idx,
            slot.s_reverse_scatter_idx,
            None, False,
            activation_type,
            False,  # is_inference_mode_enabled
            cfg.concat_layout,
            out_a=slot.a_prime,
            out_h=h,
        )
        slot.aux["sonic_moe_h"] = h

    def compute_cost(self, chunk: ChunkMeta, max_tier: int) -> ComputeCost:
        """Same FLOP accounting as :class:`MoESwiGLUFFN` — the dense
        portions dominate, and sonic uses the same amount of compute."""
        cfg = self.cfg
        avoided = [0] * (max_tier + 1)
        total = 0
        top_k = cfg.top_k
        for seq_len in chunk.seq_lens_host:
            up_gate = 2 * seq_len * top_k * cfg.d_model * (2 * cfg.expert_dim)
            total += up_gate
            if max_tier >= 3:
                avoided[3] += up_gate
            down = 2 * seq_len * top_k * cfg.expert_dim * cfg.d_model
            total += down
        for seq_len in chunk.seq_lens_host:
            total += 2 * seq_len * cfg.d_model * cfg.num_experts
        return ComputeCost(
            total_fwd_flops=total,
            avoided_recompute_flops=tuple(avoided),
        )
