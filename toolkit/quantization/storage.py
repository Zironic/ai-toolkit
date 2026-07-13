"""Opaque ordered tensor-storage declarations for Linear execution."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorStorageBinding:
    name: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class LayerStorageBinding:
    tensors: tuple[TensorStorageBinding, ...]
    execution_key: tuple
    weight_leaf_count: int
    weight_template: torch.Tensor


def _flatten_named(value, prefix):
    try:
        names, _context = value.__tensor_flatten__()
    except Exception:
        return [TensorStorageBinding(prefix, value)]
    out = []
    for name in names:
        child = getattr(value, name, None)
        if child is not None:
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            out.extend(_flatten_named(child, child_prefix))
    return out


def linear_storage_binding(weight, bias=None) -> LayerStorageBinding:
    """Describe physical storage without exposing its meaning to the mover."""
    weight_value = weight.data if isinstance(weight, torch.nn.Parameter) else weight
    bias_value = bias.data if isinstance(bias, torch.nn.Parameter) else bias
    from .fp8_linear import declare_fp8_linear, fp8_execution_key

    fp8 = declare_fp8_linear(weight_value)
    if fp8 is None:
        weight_tensors = _flatten_named(weight_value, "weight")
        execution_key = (
            type(weight_value).__module__,
            type(weight_value).__qualname__,
            tuple(
                (item.name, tuple(item.tensor.shape), str(item.tensor.dtype))
                for item in weight_tensors
            ),
        )
        weight_template = weight_value
    else:
        weight_tensors = [
            TensorStorageBinding("qdata", fp8.qdata),
            TensorStorageBinding("scale", fp8.scale),
        ]
        execution_key = fp8_execution_key(fp8.spec)
        weight_template = weight_value
    bias_tensors = [] if bias_value is None else _flatten_named(bias_value, "bias")
    tensors = tuple(weight_tensors + bias_tensors)
    return LayerStorageBinding(
        tensors=tensors,
        execution_key=execution_key,
        weight_leaf_count=len(weight_tensors),
        weight_template=weight_template,
    )


def temporary_materialization_bytes(value, dtype=torch.bfloat16) -> int:
    """Return scratch bytes needed when wrapped storage must be materialized."""
    value = value.data if isinstance(value, torch.nn.Parameter) else value
    binding = linear_storage_binding(value)
    if binding.weight_leaf_count <= 1:
        return 0
    return int(value.numel() * torch.empty((), dtype=dtype).element_size())
