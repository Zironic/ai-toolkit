"""Arena integration for the neutral row-wise FP8 execution policy."""

from __future__ import annotations

import torch

from toolkit.quantization.fp8_linear import (
    bind_parameter_operation,
    set_fp8_grad_input_enabled,
)


LINEAR_MODULES = {"Linear", "LoRACompatibleLinear", "QLinear"}


def _container(child):
    container, attribute = child, "forward"
    owner_ref = getattr(child, "ara_lora_ref", None)
    owner = owner_ref() if callable(owner_ref) else None
    if owner is None:
        candidate = getattr(getattr(child, "forward", None), "__self__", None)
        if candidate is not None and candidate is not child:
            owner = candidate
    if owner is not None and hasattr(owner, "org_forward"):
        container, attribute = owner, "org_forward"
    return container, attribute


def enable(model, *, include_ids=None, training: bool):
    """Install bound FP8 operations without owning their execution policy."""
    restores = []
    include_ids = None if include_ids is None else set(include_ids)
    for child in model.modules():
        if child.__class__.__name__ not in LINEAR_MODULES:
            continue
        if include_ids is not None and id(child) not in include_ids:
            continue
        weight = getattr(child, "weight", None)
        if not isinstance(weight, torch.nn.Parameter) or weight.requires_grad:
            continue
        bias = getattr(child, "bias", None)
        operation, tensors = bind_parameter_operation(
            weight,
            bias,
            device=weight.device,
        )
        if operation.format_key != "rowwise_fp8" or not operation.native:
            continue
        container, attribute = _container(child)
        original = getattr(container, attribute)

        def installed(
            x,
            *args,
            _tensors=tensors,
            _operation=operation,
            _original=original,
            **kwargs,
        ):
            if args or kwargs:
                return _original(x, *args, **kwargs)
            if training:
                return _operation.forward_train(x, _tensors)
            return _operation.forward_sample(x, _tensors)

        setattr(container, attribute, installed)
        restores.append((container, attribute, original, installed))
    return restores


def disable(restores) -> None:
    for container, attribute, original, installed in reversed(restores):
        if getattr(container, attribute, None) is installed:
            setattr(container, attribute, original)
