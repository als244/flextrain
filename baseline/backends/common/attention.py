"""Attention implementation selection for HuggingFace model loaders."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch
from transformers import AutoModelForCausalLM

AttentionImplementation = str


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _transformers_flag(name: str) -> bool:
    try:
        from transformers.utils import import_utils
    except ImportError:
        return False
    fn = getattr(import_utils, name, None)
    if fn is None:
        return False
    try:
        return bool(fn())
    except Exception:
        return False


def is_flash_attention_3_available() -> bool:
    return _transformers_flag("is_flash_attn_3_available")


def is_flash_attention_2_available() -> bool:
    if _transformers_flag("is_flash_attn_2_available"):
        return True
    return _has_module("flash_attn")


def is_sdpa_available() -> bool:
    return hasattr(torch.nn.functional, "scaled_dot_product_attention")


def attention_candidates(requested: str, *, allow_flash: bool = True) -> list[AttentionImplementation]:
    if requested != "auto":
        return [requested]

    candidates: list[AttentionImplementation] = []
    if allow_flash and is_flash_attention_3_available():
        candidates.append("flash_attention_3")
    if allow_flash and is_flash_attention_2_available():
        candidates.append("flash_attention_2")
    if is_sdpa_available():
        candidates.append("sdpa")
    candidates.append("eager")
    return candidates


def pick_attention_implementation(requested: str, *, allow_flash: bool = True) -> AttentionImplementation:
    return attention_candidates(requested, allow_flash=allow_flash)[0]


def load_causal_lm_with_attention(
    model_path: str,
    requested: str,
    *,
    allow_flash: bool = True,
    **model_kwargs: Any,
):
    strict = requested != "auto"
    attempts = attention_candidates(requested, allow_flash=allow_flash)
    last_exc: Exception | None = None

    for attn_implementation in attempts:
        kwargs = dict(model_kwargs)
        kwargs["attn_implementation"] = attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        except Exception as exc:
            if strict:
                raise
            last_exc = exc
            print(
                f"attention_implementation={attn_implementation} unavailable; "
                f"trying fallback ({type(exc).__name__}: {exc})",
                flush=True,
            )
            continue
        print(f"resolved_attn_implementation={attn_implementation}", flush=True)
        return model, attn_implementation

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Could not resolve attention implementation from {list(attempts)}")
