"""Compatibility surface for in-graph weight streaming primitives.

Static host layout lives in arena_offload.layout and fetch/checkpoint lifetime
lives in arena_offload.transfer. Existing imports remain available here while
model integrations migrate to the arena facade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from toolkit.memory_management.manager_modules import (
    _fp8_linear_compiled,
    _fp8_linear_training,
)



from toolkit.memory_management.arena_offload.layout import (
    LEAF_ALIGN,
    ArenaBorrowError,
    BlockLeafPlan,
    BlockPack,
    BlockPlanResult,
    IngraphPackError,
    LeafSpec,
    LinearSpec,
    LinearView,
    PackBuildResult,
    _aligned_offsets,
    _empty_host_flat,
    _flatten_leaves,
    _fp8_rowwise_qualifies,
    _rebuild_from_leaves,
    assemble_leaf_args,
    block_linear_views,
    block_tensor_views,
    build_block_leaf_plans,
    build_or_borrow_block_packs,
    is_streamed_module,
    leaf_view,
    make_block_view_maker,
    pack_block_host,
    pack_block_host_from_flat,
    release_pack,
    resident_linear_tensors,
)

def functional_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
    if weight.dtype != x.dtype and weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
        weight = weight.to(dtype=x.dtype)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(dtype=x.dtype)
    return F.linear(x, weight, bias)


def materialized_weight(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
) -> torch.Tensor:
    if scale is None:
        return weight
    view_shape = [weight.shape[0]] + [1] * (weight.ndim - 1)
    return weight.to(torch.bfloat16) * scale.reshape(view_shape).to(torch.bfloat16)


def streamed_linear_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor | None,
    *,
    fp8_qualifies: bool,
    training: bool = False,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: "float | torch.Tensor | None" = None,
):
    """Pure traced Linear math from tensor views only."""
    if scale is not None and fp8_qualifies:
        fp8_linear = _fp8_linear_training if training else _fp8_linear_compiled
        base = fp8_linear(x, weight.t(), scale.reshape(-1), bias)
    else:
        base = functional_linear(x, materialized_weight(weight, scale), bias)
    if lora_a is not None:
        lora_out = (x.to(lora_a.dtype) @ lora_a.t() @ lora_b.t()) * lora_scale
        base = base + lora_out.to(base.dtype)
    return base


@dataclass(frozen=True)
class LoraEntry:
    """Trainable LoRA leaves for one streamed Linear.

    NOT part of the host pack: A/B are small trainable fp32 Parameters that
    must stay ordinary graph inputs (GPU-resident, grad-carrying). ``scale``
    folds alpha/rank and the network multiplier -- both must be trace-time
    scalars (non-scalar multipliers fail closed as lora_untraceable at
    enable time, before any entry is built)."""

    a: torch.Tensor  # lora_down weight, (rank, in_features)
    b: torch.Tensor  # lora_up weight, (out_features, rank)
    # float (folded at enable time) or a scalar tensor (live network
    # multiplier as an ordinary graph input -- tracks with-network toggling
    # without recompiles).
    scale: "float | torch.Tensor"


@dataclass(frozen=True)
class TrainLeaf:
    """LinearView plus optional LoRA entry for the training leaves path.

    Lets the block forward keep its single `streamed_linear(x, leaf)` call
    shape for both modes: a bare LinearView selects the no-grad sampling
    path, a TrainLeaf the grad-safe training path (dispatch is on dataclass
    type -- a trace-time constant)."""

    view: LinearView
    lora: LoraEntry | None = None


def streamed_linear(
    x: torch.Tensor,
    view: LinearView | "TrainLeaf",
    *,
    training: bool = False,
    lora: LoraEntry | None = None,
):
    """Pure traced Linear math from pack views.

    ``training`` and ``lora`` presence are trace-time constants selected at
    enable time (not data-dependent branches). The training fp8 path uses the
    grad-safe autograd.Function (no weight grad, grad-input via scale
    folding); the frozen base views never require grad, so nothing here saves
    a weight for backward."""
    if isinstance(view, TrainLeaf):
        lora = view.lora
        training = True
        view = view.view
    if view.spec.kind == "fp8_rowwise" and view.spec.fp8_qualifies:
        if view.scale is None:
            raise RuntimeError(f"missing scale for {view.spec.name}")
        fp8_linear = _fp8_linear_training if training else _fp8_linear_compiled
        base = fp8_linear(x, view.weight.t(), view.scale.reshape(-1), view.bias)
    else:
        base = functional_linear(x, view.materialized_weight(), view.bias)
    if lora is not None:
        # Same math as the compile-fast LoRA path: adapter computed in its
        # own dtype (fp32), scaled, cast back to the base dtype.
        lora_out = (x.to(lora.a.dtype) @ lora.a.t() @ lora.b.t()) * lora.scale
        base = base + lora_out.to(base.dtype)
    return base


class CompileRegionError(RuntimeError):
    def __init__(self, reasons: Iterable[str]):
        self.reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
        super().__init__("in-graph compile region is not clean: " + ",".join(self.reasons))


def compile_region_reasons(module: torch.nn.Module) -> list[str]:
    reasons = []
    for child in module.modules():
        if hasattr(child, "_layer_memory_manager"):
            reasons.append("legacy_layer_manager_present")
        has_hooks = bool(getattr(child, "_forward_pre_hooks", None)) or bool(
            getattr(child, "_forward_hooks", None)
        ) or bool(getattr(child, "_forward_hooks_with_kwargs", None))
        if has_hooks:
            reasons.append("hook_present")
        if "forward" in getattr(child, "__dict__", {}):
            reasons.append("forward_hijack_present")
    return list(dict.fromkeys(reasons))


def assert_compile_region_clean(module: torch.nn.Module) -> None:
    reasons = compile_region_reasons(module)
    if reasons:
        raise CompileRegionError(reasons)


from toolkit.memory_management.arena_offload.transfer import (
    _LIFETIME_STATS,
    _STATS,
    checkpoint_recompute_context,
    compiled_checkpoint_context,
    configure_fetch_runtime,
    drain_fetch_runtime,
    fetch_free,
    fetch_free_after,
    fetch_performance_metrics,
    fetch_report,
    fetch_start,
    fetch_start_after,
    fetch_start_gated,
    fetch_start_multi,
    fetch_start_multi_after,
    fetch_start_multi_gated,
    fetch_stats,
    fetch_wait,
    free_on_backward,
    in_recompute,
    lifetime_fetch_stats,
    raise_dynamo_recompile_limit,
    reset_fetch_stats,
    set_h2d_timing_blocking,
)
