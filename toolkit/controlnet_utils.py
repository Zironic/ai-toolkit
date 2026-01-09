"""Utility helpers for strict Z-Image (VideoX) adapter handling.

These helpers prefer explicit configuration (adapter_config.controlnet_mode)
and enforce strict parity when Z-Image mode is active.
"""
from typing import Any
import inspect

# Use the more permissive name/class detection as a fallback when explicit
# `controlnet_mode` is not set on the adapter config. This preserves strict
# opt-in semantics while allowing sensible auto-detection for legacy jobs.
from .control_util import adapter_uses_zimage


def is_zimage_adapter(adapter: Any, adapter_config: Any) -> bool:
    """Return True iff Z-Image (VideoX) routing is configured or detectable.

    Detection order (prefer explicit opt-in):
    1) `adapter_config.controlnet_mode == 'zimage'` (explicit opt-in)
    2) fallback to name/class-based detection via `adapter_uses_zimage`
    """
    try:
        # Prefer an explicit opt-in signal on the adapter config
        if adapter_config is not None:
            mode = getattr(adapter_config, 'controlnet_mode', None)
            if mode is not None:
                return str(mode).lower() == 'zimage'

        # Fall back to permissive detection based on name or class hints
        return adapter_uses_zimage(adapter, adapter_config)
    except Exception:
        return False


def validate_zimage_adapter(adapter: Any, adapter_config: Any = None) -> None:
    """Validate an adapter for strict Z-Image parity.

    Raises RuntimeError with actionable messages on failure.
    Requirements enforced:
    - Adapter must accept a `control_context` kwarg on its `forward`.
    - Adapter (or adapter_config) must explicitly set `control_in_dim == 33`.
    - If the adapter appears to require Flux1-style `encoder_hidden_states` as
      a mandatory positional arg, validation fails.
    """
    # Signature check
    try:
        target_fn = getattr(adapter, 'forward', adapter)
        sig = inspect.signature(target_fn)
        params = sig.parameters
    except Exception as e:
        raise RuntimeError(f"Unable to inspect adapter signature for Z-Image validation: {e}") from e

    if 'control_context' not in params:
        raise RuntimeError("Z-Image adapters must accept a keyword argument 'control_context' in their forward signature.")

    # Check for Flux1-style required encoder_hidden_states
    if 'encoder_hidden_states' in params:
        p = params['encoder_hidden_states']
        if p.default is inspect._empty and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            raise RuntimeError("Adapter appears to require 'encoder_hidden_states' (Flux1-style). Use a VideoX-compatible adapter for Z-Image routing.")

    # control_in_dim must be explicitly 33
    cfg_dim = None
    try:
        cfg_dim = getattr(adapter, 'control_in_dim', None)
    except Exception:
        cfg_dim = None
    if cfg_dim is None and adapter_config is not None:
        try:
            cfg_dim = getattr(adapter_config, 'control_in_dim', None)
        except Exception:
            cfg_dim = None

    if cfg_dim is None:
        raise RuntimeError("Z-Image routing requires an explicit 'control_in_dim' configured on the adapter or adapter_config (expected 33). Set control_in_dim=33 to enable strict Z-Image mode.")

    if int(cfg_dim) != 33:
        raise RuntimeError(f"Z-Image routing requires control_in_dim==33, but found control_in_dim={cfg_dim}")


def ensure_zimage_mode(adapter: Any, adapter_config: Any) -> bool:
    """Ensure `adapter_config.controlnet_mode` is set to 'zimage' when detection indicates VideoX/Z-Image.

    Returns True if zimage mode is detected or was explicitly set, False otherwise.
    If `adapter_config` is None, no mutation will happen (returns False/diagnostic only).
    """
    try:
        if adapter_config is None:
            return False
        # If already explicitly set, honor it
        mode = getattr(adapter_config, 'controlnet_mode', None)
        if mode is not None:
            return str(mode).lower() == 'zimage'
        # Fallback: use permissive detection based on name/class
        if adapter_uses_zimage(adapter, adapter_config):
            try:
                adapter_config.controlnet_mode = 'zimage'
                try:
                    from .print import print_acc
                    print_acc("[CONTROLNET] Auto-set adapter_config.controlnet_mode='zimage' based on adapter/name detection")
                except Exception:
                    pass
            except Exception:
                pass
            return True
        return False
    except Exception:
        return False


__all__ = ["is_zimage_adapter", "validate_zimage_adapter", "ensure_zimage_mode"]