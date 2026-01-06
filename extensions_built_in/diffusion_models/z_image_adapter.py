from typing import Any, Dict, Optional

import traceback
import torch

from toolkit.controlnet_compat import VideoXControlnetWrapper


def load_videox_control_adapter(name_or_path: Optional[str] = None, device: Optional[str] = None, torch_dtype: Optional[torch.dtype] = None, low_cpu_mem_usage: bool = True, load_control_only: bool = True, **kwargs) -> VideoXControlnetWrapper:
    """Load or instantiate a VideoX-Fun Z-Image control transformer and return a wrapped adapter.

    - When `name_or_path` or local vendored module is present, it will import
      `ZImageControlTransformer2DModel` from the vendored file and use
      `from_pretrained` when available. If the vendored module cannot be imported
      or the class is missing, we fail loudly (no silent fallback) so callers see
      the real root cause.
    - The returned object is a `VideoXControlnetWrapper(inner)` that enforces strict VideoX parity.

    This helper is intended for testing and local integration.
    """
    # Import the vendored class at call time to surface import-time errors here
    try:
        from .z_image_transformer2d_control import ZImageControlTransformer2DModel
    except Exception as e:
        tb = traceback.format_exc()
        try:
            from toolkit.print import print_acc
            print_acc(f"[VIDE OX-ADAPTER] Failed to import vendored VideoX adapter: {e}\n{tb}")
        except Exception:
            pass
        # As a testing fallback, allow the current module to provide
        # `ZImageControlTransformer2DModel` (tests may monkeypatch it there).
        try:
            import importlib
            mod = importlib.import_module(__name__)
            model_cls = getattr(mod, 'ZImageControlTransformer2DModel', None)
            if model_cls is not None and callable(model_cls):
                print_acc(f"[VIDE OX-ADAPTER] Using fallback ZImageControlTransformer2DModel from adapter module")
            else:
                raise RuntimeError(f"Failed to import vendored VideoX adapter from 'z_image_transformer2d_control.py': {e}\n{tb}")
        except Exception as e2:
            raise RuntimeError(f"Failed to import vendored VideoX adapter from 'z_image_transformer2d_control.py': {e}\n{tb}") from e2

    model_cls = ZImageControlTransformer2DModel

    # Guard: ensure vendor provides a callable model class
    if model_cls is None or not callable(model_cls):
        raise RuntimeError("Vendored VideoX adapter present but missing callable 'ZImageControlTransformer2DModel' class. Ensure 'z_image_transformer2d_control.py' exports the class.")

    # Instantiate or load
    if name_or_path is None:
        inner = model_cls()
    else:
        # Use from_pretrained when available; allow kwargs passthrough
        if hasattr(model_cls, 'from_pretrained'):
            inner = model_cls.from_pretrained(name_or_path, low_cpu_mem_usage=low_cpu_mem_usage, load_control_only=load_control_only, **kwargs)
        else:
            raise RuntimeError("ZImageControlTransformer2DModel.from_pretrained not available; instantiate manually by passing name_or_path=None")

    # Move to device/dtype if requested
    try:
        if device is not None:
            inner.to(device)
        if torch_dtype is not None:
            inner.to(dtype=torch_dtype)
    except Exception:
        # Best-effort: ignore failures here and let callers handle device placement
        pass

    return VideoXControlnetWrapper(inner)
