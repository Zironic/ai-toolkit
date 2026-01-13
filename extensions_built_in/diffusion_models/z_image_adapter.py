from typing import Any, Dict, Optional, Union

import traceback
import torch

from toolkit.controlnet_compat import VideoXControlnetWrapper


def load_videox_control_adapter(
    name_or_path: Optional[str] = None, 
    device: Optional[str] = None, 
    torch_dtype: Optional[torch.dtype] = None, 
    low_cpu_mem_usage: bool = True, 
    load_control_only: bool = True,
    quantize: Optional[Union[str, bool]] = None,
    quantize_device: Optional[str] = None,
    base_transformer_path: Optional[str] = None,
    **kwargs
) -> VideoXControlnetWrapper:
    """Load or instantiate a VideoX-Fun Z-Image control transformer and return a wrapped adapter.

    - When `name_or_path` or local vendored module is present, it will import
      `ZImageControlTransformer2DModel` from the vendored file and use
      `from_pretrained` when available. If the vendored module cannot be imported
      or the class is missing, we fail loudly (no silent fallback) so callers see
      the real root cause.
    - The returned object is a `VideoXControlnetWrapper(inner)` that enforces strict VideoX parity.
    
    Args:
        name_or_path: Path to pretrained weights
        device: Target device for final model placement (e.g. 'cpu', 'cuda:0')
        torch_dtype: Target dtype (e.g. torch.bfloat16)
        low_cpu_mem_usage: Use accelerate's init_empty_weights for memory efficiency
        load_control_only: Only load control-related weights
        quantize: Quantization type (e.g. 'qfloat8', 'int8', 'fp8') or True to use 'qfloat8'
        quantize_device: Device to use for quantization (float8 needs CUDA). Model stays on this device after quantization.
        base_transformer_path: Path to base transformer weights. Required for control-only
            checkpoints (<8GB) which only contain delta weights. The base weights are 
            loaded first, then control weights override them.
    
    NOTE: low_cpu_mem_usage=True by default. This uses accelerate's init_empty_weights
    to create meta tensors, then loads weights directly without double-allocation.
    This is the memory-efficient path for large models (~20GB).

    This helper is intended for testing and local integration.
    """
    # Import the control transformer from our local extensions
    # (extends diffusers' ZImageTransformer2DModel)
    try:
        from .z_image_transformer2d_control import ZImageControlTransformer2DModel
    except Exception as e:
        tb = traceback.format_exc()
        raise RuntimeError(
            f"Failed to import ZImageControlTransformer2DModel from local z_image_transformer2d_control.py: {e}\n{tb}"
        ) from e

    model_cls = ZImageControlTransformer2DModel

    # Guard: ensure we have a callable model class
    if model_cls is None or not callable(model_cls):
        raise RuntimeError("Z-Image control transformer missing or not callable. Ensure 'z_image_transformer2d_control.py' exports ZImageControlTransformer2DModel.")

    # Determine target device for materialization
    # If we're going to quantize, load directly to GPU to avoid CPU->GPU copy
    if quantize:
        materialize_device = quantize_device or 'cuda'
    else:
        materialize_device = device or 'cpu'

    # Instantiate or load
    if name_or_path is None:
        inner = model_cls()
    else:
        # Use from_pretrained when available; allow kwargs passthrough
        # Pass torch_dtype so weights are loaded in the right dtype (avoids conversion later)
        # Pass materialize_device so weights load directly to target device
        # Pass base_transformer_path for control-only checkpoints
        if hasattr(model_cls, 'from_pretrained'):
            inner = model_cls.from_pretrained(
                name_or_path, 
                low_cpu_mem_usage=low_cpu_mem_usage, 
                load_control_only=load_control_only,
                torch_dtype=torch_dtype,  # Load in target dtype directly
                materialize_device=materialize_device,  # Load directly to target device
                base_transformer_path=base_transformer_path,  # For control-only checkpoints
                **kwargs
            )
        else:
            raise RuntimeError("ZImageControlTransformer2DModel.from_pretrained not available; instantiate manually by passing name_or_path=None")

    # Store the path for potential later use
    if name_or_path is not None:
        inner.name_or_path = name_or_path

    # Quantize if requested - this significantly reduces memory footprint
    # Model is already on quantize_device from from_pretrained
    if quantize:
        try:
            from toolkit.util.quantize import quantize as do_quantize, get_qtype
            from toolkit.print import print_acc
            
            # Normalize quantize value
            qtype_str = quantize if isinstance(quantize, str) else 'qfloat8'
            if qtype_str == 'fp8':
                qtype_str = 'float8'
            elif qtype_str == 'qfloat8':
                qtype_str = 'float8'
            
            print_acc(f"[CONTROLNET] Quantizing controlnet with {qtype_str} (model already on {materialize_device})...")
            
            # Apply quantization - this reduces memory from ~20GB to ~10GB for float8
            weights_qtype = get_qtype(qtype_str)
            do_quantize(inner, weights=weights_qtype)
            
            print_acc(f"[CONTROLNET] Quantization complete")
        except Exception as e:
            from toolkit.print import print_acc
            print_acc(f"[CONTROLNET] Quantization failed: {e}, continuing without quantization")
        
        # After quantization, model is already on quantize_device (usually GPU)
        # Don't move again - let caller handle final placement
        return VideoXControlnetWrapper(inner)

    # Move to device only (dtype already handled in from_pretrained)
    # Model comes back on CPU from from_pretrained, move to target device
    if device is not None:
        try:
            inner = inner.to(device)
        except NotImplementedError as e:
            # Meta tensors - this shouldn't happen with our fixed from_pretrained
            raise RuntimeError(f"Model still has meta tensors after from_pretrained - loading failed: {e}") from e

    return VideoXControlnetWrapper(inner)
