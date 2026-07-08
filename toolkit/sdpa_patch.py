"""Global SDPA GQA fallback: expand KV heads when enable_gqa would land on MATH.

torch's SDPA dispatcher tries backends in priority order (2.12 default:
FLASH > EFFICIENT > MATH > CUDNN). ``enable_gqa=True`` is only supported by
Flash; the memory-efficient (cutlass) backend rejects it. Windows torch
wheels ship without Flash, so ANY enable_gqa call -- and, on every platform,
any enable_gqa call WITH an attn_mask (Flash rejects arbitrary masks) --
silently falls through to the math backend, which materializes the full
(B, heads, Lq, Lk) score tensor: tens of ms and GiBs of transient VRAM at
diffusion sequence lengths (measured 40.2 ms -> 2.6 ms at L=4096 on the
RTX 4070).

Repeating each KV head across its query-head group is numerically identical
to ``enable_gqa=True`` and makes the memory-efficient backend eligible. So
wrap ``F.scaled_dot_product_attention`` once, process-wide: when a CUDA call
would fall to MATH (enable_gqa with a mask, or enable_gqa without Flash in
the build), expand KV and drop the flag. Everywhere else the wrapper is a
pure passthrough. Dynamo inlines it (the build facts are trace-time
constants), so compiled graphs are unaffected.

Kill-switch for diagnostics only (never required for correct training):
``AI_TOOLKIT_DISABLE_SDPA_GQA_PATCH=1``.
"""

import functools
import os

import torch
import torch.nn.functional as F

try:
    _FLASH_AVAILABLE = bool(torch.backends.cuda.is_flash_attention_available())
except Exception:
    _FLASH_AVAILABLE = False


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
        if (
            enable_gqa
            and query.is_cuda
            and key.ndim >= 3
            and key.shape[-3] != query.shape[-3]
            and (attn_mask is not None or not _FLASH_AVAILABLE)
        ):
            groups = query.shape[-3] // key.shape[-3]
            key = key.repeat_interleave(groups, dim=-3)
            value = value.repeat_interleave(groups, dim=-3)
            enable_gqa = False
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
