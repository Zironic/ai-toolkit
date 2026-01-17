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


# ---------------------------------------------------------------------------
# Deterministic, load-time helper for constructing and validating adapters
# ---------------------------------------------------------------------------

def _load_adapter_from_spec(adapter_spec, adapter_config=None, train_config=None):
    """Load an adapter object from a specification.

    adapter_spec may be:
    - an already-instantiated adapter object (returned unchanged)
    - a string path or repo id to pass to the relevant `from_pretrained` loader

    This function is intentionally defensive: it attempts to load common adapter
    types (ControlNetModel, T2IAdapter) when available and otherwise returns
    the original object or raises a helpful RuntimeError if loading fails.
    """
    # If already an object, return as-is
    if adapter_spec is None:
        return None
    # already an object
    if not isinstance(adapter_spec, (str, bytes)):
        return adapter_spec

    # adapter_spec is a path / id; attempt to load
    # Defer heavy imports to runtime so module import is cheap when unused
    last_err = None
    try:
        from diffusers import ControlNetModel
        try:
            return ControlNetModel.from_pretrained(adapter_spec)
        except Exception as e:
            last_err = e
    except Exception as e:
        last_err = e

    try:
        from diffusers import T2IAdapter
        try:
            return T2IAdapter.from_pretrained(adapter_spec)
        except Exception as e:
            last_err = e
    except Exception as e:
        last_err = e

    # If we didn't find a loader, raise a clear error
    raise RuntimeError(
        f"Failed to load adapter from '{adapter_spec}'. Ensure the path is correct and that the required Diffusers adapter type is available. Underlying error: {last_err}"
    )


def prepare_controlnet_adapter(sd, adapter_spec_or_obj, adapter_config=None, train_config=None, *, strict: bool = True, require_zimage_model: bool = True):
    """Deterministically construct and validate a ControlNet/T2I adapter for training.

    Responsibilities:
    - Load the adapter if a path is provided
    - Set `name_or_path` if missing when a repo/path is supplied via adapter_config
    - Enforce or set `control_in_dim` for VideoX/Z-Image adapters (33)
    - Validate that the provided StableDiffusion `sd` instance supports required
      model-side hooks for Z-Image routing when `require_zimage_model` is True

    Returns the adapter instance on success.

    Raises RuntimeError with clear remediation steps when strict and validation fails.
    """
    # Load adapter if needed
    try:
        adapter = _load_adapter_from_spec(adapter_spec_or_obj, adapter_config, train_config)
    except Exception as e:
        if strict:
            raise RuntimeError(f"[CONTROLNET-LOAD] adapter load failed: {e}") from e
        else:
            return None

    # If adapter_config supplies a name, set it when missing
    try:
        name = None
        if adapter_config is not None:
            name = getattr(adapter_config, 'name_or_path', None) or getattr(adapter_config, 'adapter_name', None)
        # If we loaded from a string, try to attach the path as name
        if name is None and isinstance(adapter_spec_or_obj, str):
            name = adapter_spec_or_obj
        if name is not None:
            try:
                set_adapter_name_if_missing(adapter, name)
            except Exception:
                pass
    except Exception:
        pass

    # If this looks like a Z-Image/VideoX adapter, enforce control_in_dim==33
    try:
        if adapter_uses_zimage(adapter, adapter_config):
            # Enforce strictly; force will set to 33 when missing
            ok = enforce_zimage_control_in_dim(adapter, expected=33, force=True)
            if not ok and strict:
                raise RuntimeError("[CONTROLNET-LOAD] Failed to enforce VideoX control_in_dim=33 on adapter; ensure adapter exposes 'control_in_dim' or use a known VideoX adapter.")
    except Exception as e:
        if strict:
            raise RuntimeError(f"[CONTROLNET-LOAD] Z-Image control_in_dim enforcement failed: {e}") from e

    # If strict, validate sd compatibility for Z-Image routing
    try:
        if adapter_uses_zimage(adapter, adapter_config) and require_zimage_model:
            missing = []
            if sd is None:
                missing.append("StableDiffusion instance (sd) was not provided for validation")
            else:
                if not hasattr(sd, 'get_noise_prediction') or not callable(getattr(sd, 'get_noise_prediction')):
                    missing.append("model-side hook `get_noise_prediction` is missing on the loaded SD instance")
                if not hasattr(sd, 'encode_control_images') or not callable(getattr(sd, 'encode_control_images')):
                    missing.append("SD instance lacks `encode_control_images` required for Z-Image routing")

            if missing:
                msg = (
                    "Z-Image adapter requires a Z-Image aware SD model. "
                    "The following issues were detected: " + "; ".join(missing) + ". "
                    "Remediation: load an SD variant that supports VideoX/Z-Image routing (model transformer with control_in_dim=33 and `get_noise_prediction`) or set `train_config.require_zimage_model=False` to allow trainer-side fallback."
                )
                raise RuntimeError(f"[CONTROLNET-LOAD] {msg}")
    except Exception as e:
        if strict:
            raise
        else:
            # Non-strict: attach a note on the adapter and continue
            try:
                setattr(adapter, '_controlnet_load_note', str(e))
            except Exception:
                pass

    # Final check: ensure adapter has numeric control_in_dim set
    try:
        if not hasattr(adapter, 'control_in_dim') or getattr(adapter, 'control_in_dim') is None:
            # For non-zimage adapters strict may tolerate missing value, but for zimage it should be present
            if adapter_uses_zimage(adapter, adapter_config):
                if strict:
                    raise RuntimeError("[CONTROLNET-LOAD] adapter.control_in_dim is missing after enforcement. This is required for Z-Image adapters.")
                else:
                    # set a conservative default only if requested
                    enforce_zimage_control_in_dim(adapter, expected=33, force=True)
    except Exception:
        raise

    # Attach a reference to the SD instance for adapter-level diagnostics/timing when possible
    try:
        if sd is not None:
            setattr(adapter, '_owner_sd', sd)
    except Exception:
        pass

    # Return the prepared adapter
    return adapter
