"""Neutral functional adapter execution primitives.

This module intentionally knows nothing about LoRA networks, Krea2, or memory
management. Runtimes provide the explicit Linear callback and adapters wrap it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


_USE_BASE = object()


@dataclass(frozen=True, eq=False)
class FunctionalLinear:
    """A Linear operation backed by explicit, already-selected tensors."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    scale: torch.Tensor | None
    call_fn: Callable
    materialize_fn: Callable

    def __call__(
        self,
        x,
        *args,
        weight=_USE_BASE,
        bias=_USE_BASE,
        scale=_USE_BASE,
        **kwargs,
    ):
        if args or kwargs:
            raise TypeError("FunctionalLinear accepts only x and explicit tensor overrides")
        return self.call_fn(
            x,
            self.weight if weight is _USE_BASE else weight,
            self.bias if bias is _USE_BASE else bias,
            self.scale if scale is _USE_BASE else scale,
        )

    def materialized_weight(self, dtype=None) -> torch.Tensor:
        return self.materialize_fn(self.weight, self.scale, dtype)


def functional_base_weight(inner, *, dtype=None):
    materialize = getattr(inner, "materialized_weight", None)
    if materialize is None:
        return None
    return materialize(dtype=dtype)


def functional_base_bias(inner):
    return getattr(inner, "bias", None) if isinstance(inner, FunctionalLinear) else None


def run_functional_adapter(adapter, inner, x, *args, **kwargs):
    method = getattr(adapter, "functional_forward", None)
    if method is None:
        raise TypeError(f"{type(adapter).__name__} does not implement functional_forward")
    return method(inner, x, *args, **kwargs)
