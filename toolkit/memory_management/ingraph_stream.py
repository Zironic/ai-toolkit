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



from toolkit.memory_management.arena_offload.layout import (
    LEAF_ALIGN,
    ArenaBorrowError,
    BlockLeafPlan,
    BlockPack,
    BlockPlanResult,
    IngraphPackError,
    LayerStorageView,
    LeafSpec,
    LinearSpec,
    PackBuildResult,
    _aligned_offsets,
    _empty_host_flat,
    _flatten_leaves,
    _rebuild_from_leaves,
    assemble_leaf_args,
    block_storage_views,
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


@dataclass(frozen=True)
class LinearView:
    """Legacy execution-bound view layered over an opaque storage view.

    The immutable arena itself publishes only ``LayerStorageView``. This
    compatibility type remains here for the older in-graph streaming surface,
    whose callers already bind operations outside storage movement.
    """

    storage: LayerStorageView
    operation: object

    @property
    def spec(self):
        return self.storage.spec

    @property
    def tensors(self):
        return self.storage.tensors

    @property
    def weight(self):
        return self.operation.functional_components(self.tensors)[0]

    @property
    def bias(self):
        return self.operation.functional_components(self.tensors)[1]

    @property
    def scale(self):
        return self.operation.functional_components(self.tensors)[2]

    def materialized_weight(self) -> torch.Tensor:
        return self.operation.materialize(self.tensors)

    def __iter__(self):
        yield self.materialized_weight()
        yield self.bias


def block_linear_views(flat: torch.Tensor, pack: BlockPack, operations):
    """Bind caller-owned operations to storage-only arena views."""
    storage_views = block_storage_views(flat, pack)
    out = {}
    for index, spec in enumerate(pack.linears):
        operation = (
            operations[spec.name]
            if isinstance(operations, dict)
            else operations[index]
        )
        out[spec.name] = LinearView(
            storage=storage_views[spec.name],
            operation=operation,
        )
    return out


def streamed_linear_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scale: torch.Tensor | None,
    *,
    operation,
    training: bool = False,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    lora_scale: "float | torch.Tensor | None" = None,
):
    """Pure traced Linear math from tensor views only."""
    tensors = operation.explicit_tensors(weight, bias, scale)
    forward = operation.forward_train if training else operation.forward_sample
    base = forward(x, tensors)
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
    forward = view.operation.forward_train if training else view.operation.forward_sample
    base = forward(x, view.tensors)
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
    fetch_start_multi,
    fetch_start_multi_after,
    fetch_stats,
    fetch_wait,
    free_on_backward,
    in_recompute,
    lifetime_fetch_stats,
    raise_dynamo_recompile_limit,
    reset_fetch_stats,
    set_h2d_timing_blocking,
)
