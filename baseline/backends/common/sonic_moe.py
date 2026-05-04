"""SonicMoE adapters for HuggingFace-style sparse MoE blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn

MoEKernelMode = Literal["hf", "auto", "sonic"]


@dataclass(frozen=True)
class SonicMoEConfig:
    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    norm_topk_prob: bool
    activation_type: str


def _load_sonic_kernel(mode: MoEKernelMode):
    if mode == "hf":
        return None
    try:
        from kernels import get_kernel
    except ImportError as exc:
        if mode == "auto":
            print("sonic_moe_backend=hf reason=kernels_package_not_installed", flush=True)
            return None
        raise RuntimeError("Install `kernels` to use --moe-kernel-backend sonic") from exc
    return get_kernel("kernels-community/sonic-moe")


def _activation_type(module: nn.Module) -> str:
    hidden_act = getattr(getattr(module, "config", None), "hidden_act", "silu")
    if hidden_act in {"silu", "swish"}:
        return "swiglu"
    if hidden_act in {"gelu", "gelu_pytorch_tanh", "gelu_new"}:
        return "geglu"
    raise ValueError(f"SonicMoE adapter only supports SwiGLU/GEGLU experts, got hidden_act={hidden_act!r}")


def _is_supported_hf_moe(module: nn.Module) -> bool:
    experts = getattr(module, "experts", None)
    if not isinstance(experts, nn.ModuleList) or len(experts) == 0:
        return False
    first = experts[0]
    return all(hasattr(first, name) for name in ("gate_proj", "up_proj", "down_proj")) and hasattr(module, "gate")


def _build_config(module: nn.Module) -> SonicMoEConfig:
    experts = module.experts
    first = experts[0]
    gate_proj = first.gate_proj
    up_proj = first.up_proj
    down_proj = first.down_proj
    if gate_proj.bias is not None or up_proj.bias is not None or down_proj.bias is not None:
        raise ValueError("SonicMoE HF adapter currently supports bias-free experts only")
    if gate_proj.out_features != up_proj.out_features:
        raise ValueError("SonicMoE HF adapter requires gate_proj and up_proj to have matching output sizes")
    num_experts = int(getattr(module, "num_experts", len(experts)))
    top_k = int(getattr(module, "top_k"))
    if num_experts != len(experts):
        raise ValueError(f"num_experts={num_experts} does not match len(experts)={len(experts)}")
    if num_experts % 8 != 0:
        raise ValueError(f"SonicMoE top-k router requires num_experts to be divisible by 8, got {num_experts}")
    if top_k > 128:
        raise ValueError(f"SonicMoE top-k router supports top_k <= 128, got {top_k}")

    return SonicMoEConfig(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=gate_proj.in_features,
        intermediate_size=gate_proj.out_features,
        norm_topk_prob=bool(getattr(module, "norm_topk_prob", False)),
        activation_type=_activation_type(module),
    )


class SonicMoEAdapter(nn.Module):
    def __init__(self, original: nn.Module, kernel_module: Any):
        super().__init__()
        self.kernel_module = kernel_module
        self.config = _build_config(original)
        self.num_experts = self.config.num_experts
        self.top_k = self.config.top_k
        self.norm_topk_prob = self.config.norm_topk_prob

        dtype = original.gate.weight.dtype
        device = original.gate.weight.device
        self.router_weight = nn.Parameter(original.gate.weight.detach().clone())
        self.w1 = nn.Parameter(
            torch.empty(
                2 * self.config.intermediate_size,
                self.config.hidden_size,
                self.config.num_experts,
                dtype=dtype,
                device=device,
            )
        )
        self.w2 = nn.Parameter(
            torch.empty(
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_experts,
                dtype=dtype,
                device=device,
            )
        )
        self.shared_expert = getattr(original, "shared_expert", None)
        self.shared_expert_gate = getattr(original, "shared_expert_gate", None)
        self._copy_expert_weights(original.experts)

    @torch.no_grad()
    def _copy_expert_weights(self, experts: nn.ModuleList) -> None:
        for idx, expert in enumerate(experts):
            self.w1[0::2, :, idx].copy_(expert.gate_proj.weight)
            self.w1[1::2, :, idx].copy_(expert.up_proj.weight)
            self.w2[:, :, idx].copy_(expert.down_proj.weight)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not hidden_states.is_cuda:
            raise RuntimeError("SonicMoE backend requires CUDA tensors")
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.config.hidden_size)
        stream_id = torch.cuda.current_stream(hidden_states.device).cuda_stream
        routed, router_logits, _ = self.kernel_module.moe_TC_softmax_topk_layer(
            flat,
            self.router_weight,
            self.w1,
            None,
            self.w2,
            None,
            self.config.top_k,
            stream_id,
            self.config.activation_type,
            not self.training,
            False,
            self.config.norm_topk_prob,
        )
        routed = routed.reshape(original_shape)
        if self.shared_expert is not None:
            shared = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared
            routed = routed + shared
        return routed, router_logits


def _replace_child(parent: nn.Module, name: str, child: nn.Module) -> None:
    if isinstance(parent, nn.ModuleList) and name.isdigit():
        parent[int(name)] = child
    else:
        setattr(parent, name, child)


def _get_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    child_name = parts[-1]
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, nn.ModuleList) and part.isdigit() else getattr(parent, part)
    return parent, child_name


def apply_sonic_moe_to_hf_model(model: nn.Module, mode: MoEKernelMode) -> int:
    kernel_module = _load_sonic_kernel(mode)
    if kernel_module is None:
        return 0

    candidates = [(name, module) for name, module in model.named_modules() if _is_supported_hf_moe(module)]
    replaced = 0
    for name, module in candidates:
        parent, child_name = _get_parent(model, name)
        adapter = SonicMoEAdapter(module, kernel_module)
        _replace_child(parent, child_name, adapter)
        replaced += 1
    print(f"sonic_moe_backend={'sonic' if replaced else 'hf'} replaced_moe_blocks={replaced}", flush=True)
    if mode == "sonic" and replaced == 0:
        raise RuntimeError("No compatible HuggingFace sparse MoE blocks were found for SonicMoE replacement")
    return replaced
