"""Global SDPA GQA dispatch without math score materialization.

Recent cuDNN SDPA builds support native GQA, including broadcast padding
masks. Prefer that path without expanding K/V when the exact eager signature
is eligible. During compilation use build and dtype facts, because constructing
SDPAParams in a traced function causes a graph break. Retain KV expansion as
the fallback for builds or signatures without cuDNN support.

Kill-switch for diagnostics only (never required for correct training):
``AI_TOOLKIT_DISABLE_SDPA_GQA_PATCH=1``.
"""

import functools
import os

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

try:
    _FLASH_AVAILABLE = bool(torch.backends.cuda.is_flash_attention_available())
except Exception:
    _FLASH_AVAILABLE = False

try:
    _CUDNN_AVAILABLE = bool(torch.backends.cuda.cudnn_sdp_enabled())
except Exception:
    _CUDNN_AVAILABLE = False


_GQA_BACKEND_MODE = "auto"


def set_gqa_backend_mode(mode: str) -> None:
    """Select GQA dispatch for a real job; benchmark modes fail closed."""
    normalized = str(mode).strip().lower()
    allowed = {"auto", "cudnn", "expanded_efficient"}
    if normalized not in allowed:
        raise ValueError(
            f"Invalid sdpa_gqa_backend {mode!r}; expected one of {sorted(allowed)}"
        )
    global _GQA_BACKEND_MODE
    _GQA_BACKEND_MODE = normalized


def get_gqa_backend_mode() -> str:
    return _GQA_BACKEND_MODE


def can_use_native_cudnn_gqa(
    query, key, value, attn_mask, dropout_p=0.0, is_causal=False
) -> bool:
    """Whether this unexpanded CUDA GQA call should be forced to cuDNN."""
    if not (
        _CUDNN_AVAILABLE
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and key.ndim >= 3
        and key.shape[-3] != query.shape[-3]
    ):
        return False
    if torch.compiler.is_compiling():
        # Avoid tracing SDPAParams (a pybind object). Only assume eligibility
        # for the exact Krea2 signature validated by the CUDA smoke.
        return (
            query.shape[-1] == 128
            and key.shape[-1] == 128
            and dropout_p == 0.0
            and not is_causal
        )
    try:
        params = torch.backends.cuda.SDPAParams(
            query, key, value, attn_mask, dropout_p, is_causal, True
        )
        return bool(torch.backends.cuda.can_use_cudnn_attention(params, False))
    except Exception:
        return False


def apply_sdpa_gqa_patch() -> None:
    """Install the wrapper on torch.nn.functional (idempotent)."""
    if os.environ.get("AI_TOOLKIT_DISABLE_SDPA_GQA_PATCH", "0") not in ("", "0"):
        return
    original = F.scaled_dot_product_attention
    if getattr(original, "_aitk_gqa_patch", False):
        return

    @functools.wraps(original)
    def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        *,
        scale=None,
        enable_gqa=False,
    ):
        needs_fast_gqa = (
            enable_gqa
            and query.is_cuda
            and key.ndim >= 3
            and key.shape[-3] != query.shape[-3]
            and (attn_mask is not None or not _FLASH_AVAILABLE)
        )
        mode = get_gqa_backend_mode()
        if needs_fast_gqa and mode == "cudnn" and not can_use_native_cudnn_gqa(
            query, key, value, attn_mask, dropout_p, is_causal
        ):
            raise RuntimeError(
                "sdpa_gqa_backend=cudnn was requested, but cuDNN rejected the "
                "unexpanded GQA signature"
            )
        if (
            needs_fast_gqa
            and mode != "expanded_efficient"
            and can_use_native_cudnn_gqa(
                query, key, value, attn_mask, dropout_p, is_causal
            )
        ):
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return original(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=True,
                )
        if needs_fast_gqa:
            groups = query.shape[-3] // key.shape[-3]
            key = key.repeat_interleave(groups, dim=-3)
            value = value.repeat_interleave(groups, dim=-3)
            enable_gqa = False
            if mode == "expanded_efficient":
                with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                    return original(
                        query,
                        key,
                        value,
                        attn_mask=attn_mask,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        scale=scale,
                        enable_gqa=False,
                    )
        return original(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    scaled_dot_product_attention._aitk_gqa_patch = True
    scaled_dot_product_attention._aitk_original = original
    F.scaled_dot_product_attention = scaled_dot_product_attention
