"""Small helper utilities for routing ControlNet adapters."""
from typing import Optional
import os
try:
    import yaml
except Exception:
    yaml = None
from .print import print_acc


def adapter_uses_zimage(adapter, adapter_config) -> bool:
    """Return True if the adapter should be routed via the zimage (VideoX) path.

    Detection order (prefer explicit opt-in):
    1) adapter_config.controlnet_mode in ('zimage','video_x') -> True
    2) adapter_config.name_or_path or adapter.name_or_path contains known zimage substrings
    3) adapter class name contains zimage / videox hints
    """
    try:
        mode = getattr(adapter_config, 'controlnet_mode', None)
        if mode is not None and str(mode).lower() in ('zimage', 'video_x'):
            return True
    except Exception:
        pass

    name_candidates = []
    try:
        if getattr(adapter_config, 'name_or_path', None) is not None:
            name_candidates.append(str(adapter_config.name_or_path))
    except Exception:
        pass
    try:
        if adapter is not None and getattr(adapter, 'name_or_path', None) is not None:
            name_candidates.append(str(adapter.name_or_path))
    except Exception:
        pass

    for n in name_candidates:
        nlow = n.lower()
        for pat in ('zimage', 'z_image', 'z-image', 'videox', 'video_x', 'pipeline_z_image', 'zimage_control'):
            if pat in nlow:
                return True

    try:
        cls = adapter.__class__.__name__.lower() if adapter is not None else ''
        if 'zimage' in cls or 'videox' in cls or 'z_image' in cls:
            return True
    except Exception:
        pass

    return False


def infer_expected_in_ch(adapter):
    """Infer the expected input channel count for a control adapter.

    Strategy (prefer explicit signals):
    1. `adapter.control_in_dim` attribute
    2. `conv_in.weight.shape[1]` if a named `conv_in` exists anywhere on the adapter or its inner wrapper
    3. Search for a module named 'conv_in' in named_modules()
    4. Prefer common video/control sizes (4, 3, 1) if present among convs
    5. Fallback: most common conv in_channels

    This helper is defensive about wrappers (VideoXControlnetWrapper) and will attempt
    to unwrap common wrapper attribute names like `inner` or `module` when necessary.
    """
    def _maybe_unwrap(obj):
        # Unwrap common wrapper patterns to reveal the inner controlnet
        seen = set()
        cur = obj
        while True:
            if cur is None or id(cur) in seen:
                return cur
            seen.add(id(cur))
            # common wrapper fields
            for attr in ('inner', 'module', 'model', 'base_model', 'controlnet'):
                try:
                    candidate = getattr(cur, attr, None)
                except Exception:
                    candidate = None
                if candidate is not None and candidate is not cur:
                    cur = candidate
                    break
            else:
                return cur

    expected_in_ch = None
    try:
        a = _maybe_unwrap(adapter)
        # Only honor an explicit `control_in_dim` attribute on the adapter.
        expected_in_ch = getattr(a, 'control_in_dim', None)
        if expected_in_ch is not None:
            return int(expected_in_ch)
    except Exception:
        pass

    # Heuristic inspection removed: do not infer expected channels from the
    # adapter's internals. Only honor an explicit `control_in_dim` attribute.
    return None


def ensure_control_in_dim(adapter, strict: bool = False, fallback: Optional[int] = None) -> bool:
    """Ensure the adapter exposes a numeric `control_in_dim` attribute if it can be derived.

    Attempts (in order):
    - adapter.config['control_in_dim'] or config.control_in_dim

    If `strict` is True, raise a RuntimeError when a value cannot be determined.
    If `fallback` is provided it MUST be 33; non-33 fallbacks are rejected to
    avoid accidental assignments to non-VideoX defaults. When `fallback=33` is
    provided and no value is discoverable, `33` will be assigned to
    `adapter.control_in_dim`.

    Returns True when attribute was set, False otherwise. This helper provides
    deterministic semantics for loader code paths.
    """
    try:
        if adapter is None:
            if strict:
                raise RuntimeError("Cannot determine control_in_dim: adapter is None")
            # If adapter is None and a fallback is provided, nothing to set
            return False
        cfg = getattr(adapter, 'config', None)
        ctl = None
        if cfg is not None:
            # Only honor explicit `control_in_dim` in the config. Avoid using
            # a generic `in_channels` value as it can lead to mistaken inference
            # (this was the source of accidental `control_in_dim=4` in jobs).
            if isinstance(cfg, dict):
                ctl = cfg.get('control_in_dim', None)
            else:
                ctl = getattr(cfg, 'control_in_dim', None)
        # Do NOT infer from adapter.in_channels: this avoids accidental overrides
        # that were previously the root cause of training misconfigurations.
        if ctl is None:
            if fallback is not None:
                # Enforce that only fallback==33 is allowed
                try:
                    if int(fallback) != 33:
                        raise RuntimeError("Non-33 fallback for control_in_dim is not permitted. Use explicit config or fallback=33.")
                except Exception:
                    raise
                ctl_val = int(fallback)
                setattr(adapter, 'control_in_dim', ctl_val)
                try:
                    adapter_repr = None if adapter is None else f"{type(adapter).__name__}@{hex(id(adapter))}"
                    print_acc(f"[CONTROLNET-LOAD] ensure_control_in_dim: defaulted control_in_dim={ctl_val} adapter={adapter_repr}")
                except Exception:
                    pass
                return True
            if strict:
                raise RuntimeError(
                    "Unable to determine 'control_in_dim' for adapter. "
                    "Set adapter.control_in_dim explicitly or include 'control_in_dim' in the adapter config."
                )
            return False
        # coerce to int when possible
        try:
            ctl_val = int(ctl)
        except Exception:
            ctl_val = ctl
        setattr(adapter, 'control_in_dim', ctl_val)
        try:
            adapter_repr = None if adapter is None else f"{type(adapter).__name__}@{hex(id(adapter))}"
            print_acc(f"[CONTROLNET-LOAD] ensure_control_in_dim: set control_in_dim={ctl_val} adapter={adapter_repr}")
        except Exception:
            pass
        return True
    except Exception:
        # Re-raise for strict mode to give deterministic failure semantics
        if strict:
            raise
        return False


def enforce_zimage_control_in_dim(adapter, expected: int = 33, force: bool = True) -> bool:
    """Ensure adapter uses VideoX/Z-Image control_in_dim.

    Strict policy: this helper will only set `control_in_dim` to 33 (VideoX default).
    Calls that pass a different `expected` value will be coerced to 33 and a warning
    will be emitted. This prevents accidental assignments to non-33 values (source
    of many training mismatches).
    """
    try:
        if adapter is None:
            return False
        # Only apply to adapters that look like Z-Image/VideoX adapters
        if not adapter_uses_zimage(adapter, None):
            return False

        # Enforce 33 unconditionally when forcing.
        if expected != 33:
            try:
                from .print import print_acc
                print_acc(f"[CONTROLNET-LOAD] enforce_zimage_control_in_dim called with expected={expected}; coercing to 33 to maintain strict VideoX parity")
            except Exception:
                pass
            expected = 33

        cur = getattr(adapter, 'control_in_dim', None)
        if cur == expected:
            return False
        if force:
            try:
                setattr(adapter, 'control_in_dim', int(expected))
                setattr(adapter, '_control_in_dim_forced', True)
                try:
                    from .print import print_acc
                    name = getattr(adapter, 'name_or_path', None)
                    print_acc(f"[CONTROLNET-LOAD] forced control_in_dim={expected} for adapter={name} (was={cur})")
                except Exception:
                    pass
                # Additionally, emit a helpful note if the adapter contains conv modules
                try:
                    named = getattr(adapter, 'named_modules', None)
                    if callable(named):
                        for n, mod in named():
                            # try to detect conv modules and check their in_channels
                            if hasattr(mod, 'in_channels'):
                                ic = getattr(mod, 'in_channels')
                                if ic is not None and int(ic) != int(expected):
                                    try:
                                        from .print import print_acc
                                        print_acc(f"[CONTROLNET-LOAD] warning: adapter module {n} has in_channels={ic} which conflicts with forced control_in_dim={expected}")
                                    except Exception:
                                        pass
                                    break
                except Exception:
                    pass
                return True
            except Exception:
                return False
        return False
    except Exception:
        return False


def set_adapter_name_if_missing(adapter, name: str) -> bool:
    """Set `adapter.name_or_path` to `name` if it's missing.

    Returns True when the name was set, False if it already existed or could not be set.
    """
    try:
        if adapter is None:
            return False
        cur = getattr(adapter, 'name_or_path', None)
        if cur is not None:
            return False
        setattr(adapter, 'name_or_path', str(name))
        setattr(adapter, '_name_forced', True)
        try:
            from .print import print_acc
            adapter_repr = f"{type(adapter).__name__}@{hex(id(adapter))}"
            print_acc(f"[CONTROLNET-LOAD] forced name_or_path={name} for adapter={adapter_repr}")
        except Exception:
            pass
        return True
    except Exception:
        return False