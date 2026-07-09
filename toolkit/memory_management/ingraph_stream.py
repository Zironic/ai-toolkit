"""In-graph weight streaming primitives.

This module contains the host-side flat block representation and the custom
ops used by the compile-visible streaming path. It is intentionally independent
from the legacy per-Linear forward hijack path so models can opt in block by
block.
"""

from __future__ import annotations

import collections
import contextlib
import itertools
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from toolkit.memory_management import pin_manager

from toolkit.memory_management.manager_modules import (
    _fp8_linear_compiled,
    _fp8_linear_training,
)



LEAF_ALIGN = 256


@dataclass(frozen=True)
class LeafSpec:
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]
    role: str


@dataclass(frozen=True)
class LinearSpec:
    name: str
    weight: LeafSpec
    bias: LeafSpec | None
    weight_requires_grad: bool
    bias_requires_grad: bool
    kind: str = "float"
    weight_scale: LeafSpec | None = None
    fp8_qualifies: bool = False


@dataclass
class BlockPack:
    block_key: str
    host_flat: torch.Tensor
    linears: tuple[LinearSpec, ...]
    required_pin_bytes: int
    pinned: bool
    fp8_flags: tuple[bool, ...] = ()
    view_maker: object | None = None
    # Ownership of ``host_flat``'s pin grant. ``pin_handle`` is the PinHandle
    # returned by pin_manager.pin_alloc for this pack's OWN flat allocation
    # (None when the pack didn't allocate -- e.g. it borrows another owner's
    # storage). ``owns_flat`` gates release_pack: a borrowed pack (arena-backed,
    # borrowed_from_arena=True) must never release someone else's handle.
    pin_handle: object | None = None
    owns_flat: bool = True
    borrowed_from_arena: bool = False


@dataclass(frozen=True)
class LinearView:
    spec: LinearSpec
    weight: torch.Tensor
    bias: torch.Tensor | None
    scale: torch.Tensor | None = None

    def materialized_weight(self) -> torch.Tensor:
        if self.spec.kind != "fp8_rowwise":
            return self.weight
        if self.scale is None:
            raise RuntimeError(f"missing scale for {self.spec.name}")
        view_shape = [self.weight.shape[0]] + [1] * (self.weight.ndim - 1)
        return self.weight.to(torch.bfloat16) * self.scale.reshape(view_shape).to(torch.bfloat16)

    def __iter__(self):
        yield self.materialized_weight()
        yield self.bias


def _flatten_leaves(t):
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        return [t]
    out = []
    for name in names:
        inner = getattr(t, name, None)
        if inner is not None:
            out.extend(_flatten_leaves(inner))
    return out


def _rebuild_from_leaves(src, leaves_iter):
    try:
        names, ctx = src.__tensor_flatten__()
    except Exception:
        return next(leaves_iter)
    moved = {}
    for name in names:
        inner = getattr(src, name, None)
        moved[name] = None if inner is None else _rebuild_from_leaves(inner, leaves_iter)
    return type(src).__tensor_unflatten__(moved, ctx, src.size(), src.stride())


def _aligned_offsets(leaves: Iterable[torch.Tensor], align: int = LEAF_ALIGN):
    offsets = []
    total = 0
    for leaf in leaves:
        total = (total + align - 1) // align * align
        offsets.append(total)
        total += leaf.numel() * leaf.element_size()
    return offsets, total


def _fp8_rowwise_qualifies(qdata: torch.Tensor, scale: torch.Tensor) -> bool:
    if not hasattr(torch, "_scaled_mm"):
        return False
    return not (
        qdata.dtype != torch.float8_e4m3fn
        or qdata.ndim != 2
        or scale.numel() != qdata.shape[0]
        or qdata.shape[0] % 16
        or qdata.shape[1] % 16
    )


def _empty_host_flat(
    nbytes: int,
    *,
    pin: bool = True,
    kind: str = "ingraph_pack",
    pin_mechanism: str = "alloc",
) -> tuple[torch.Tensor, bool, object | None]:
    if not pin:
        return torch.empty(nbytes, dtype=torch.uint8), False, None
    if pin_mechanism == "register":
        # Exact DXGI cost: cudaHostRegister on an exact-size tensor, no
        # caching-allocator power-of-two bucket rounding. The pinned weight
        # arena's flats are large and long-lived, where the rounding
        # overhead compounds to gigabytes (observed live: 8.86 GiB of
        # pin_alloc flats committed 12.70 GiB of DXGI usage).
        handle = pin_manager.pin_register(nbytes, kind, required=False)
    else:
        handle = pin_manager.pin_alloc(
            nbytes,
            kind,
            required=False,
            mode="sampling",
        )
    return handle.tensor, bool(handle.pinned), handle


def release_pack(pack: "BlockPack | None") -> None:
    """Release a pack's own pin grant.

    A borrowed pack (``owns_flat=False``, e.g. arena-backed) must never
    release someone else's handle -- the owner (the arena) is responsible for
    its own flat's lifetime."""
    if pack is None or not pack.owns_flat:
        return
    pin_manager.release(pack.pin_handle)
    pack.pin_handle = None


def pack_block_host(
    block_key: str,
    linears,
    *,
    repoint: bool = True,
    pin: bool = True,
    kind: str = "ingraph_pack",
    pin_mechanism: str = "alloc",
) -> BlockPack:
    """Pack a block's Linear weights/biases into one aligned host byte buffer.

    ``linears`` is an iterable of ``(name, module)`` or
    ``(name, weight, bias)`` entries. When ``repoint`` is true the modules'
    Parameters are replaced by views into the flat host buffer. ``kind`` is
    the pin_manager ledger kind for this pack's own allocation (callers that
    build a persistent weight arena pass ``kind="weights"`` so the bytes are
    accounted under the same tier as ordinary offload pins).
    """
    normalized = []
    leaves = []
    for entry in linears:
        if len(entry) == 2:
            name, module = entry
            weight = module.weight
            bias = getattr(module, "bias", None)
        else:
            name, weight, bias = entry
            module = None
        w_leaves = _flatten_leaves(weight.data if isinstance(weight, torch.nn.Parameter) else weight)
        b_data = bias.data if isinstance(bias, torch.nn.Parameter) else bias
        b_leaves = _flatten_leaves(b_data) if b_data is not None else []
        normalized.append((name, module, weight, bias, w_leaves, b_leaves))
        leaves.extend(w_leaves)
        leaves.extend(b_leaves)

    offsets, total = _aligned_offsets(leaves)
    host, pinned, pin_handle = _empty_host_flat(
        total, pin=pin, kind=kind, pin_mechanism=pin_mechanism
    )
    try:
        for leaf, offset in zip(leaves, offsets):
            nbytes = leaf.numel() * leaf.element_size()
            host[offset:offset + nbytes].view(leaf.dtype).reshape(leaf.shape).copy_(leaf)

        cursor = 0
        specs = []
        for name, module, weight, bias, w_leaves, b_leaves in normalized:
            rebuilt = []
            for leaf in itertools.chain(w_leaves, b_leaves):
                offset = offsets[cursor]
                nbytes = leaf.numel() * leaf.element_size()
                rebuilt.append(host[offset:offset + nbytes].view(leaf.dtype).reshape(leaf.shape))
                cursor += 1
            leaf_kind = "float"
            weight_scale_spec = None
            fp8_qualifies = False
            if len(w_leaves) == 1:
                weight_role = "float_weight"
            elif (
                len(w_leaves) == 2
                and w_leaves[0].dtype == torch.float8_e4m3fn
                and w_leaves[1].is_floating_point()
            ):
                leaf_kind = "fp8_rowwise"
                weight_role = "qdata"
                scale_leaf = w_leaves[1]
                fp8_qualifies = _fp8_rowwise_qualifies(w_leaves[0], scale_leaf)
                scale_offset = offsets[cursor - len(w_leaves) - len(b_leaves) + 1]
                weight_scale_spec = LeafSpec(
                    offset=scale_offset,
                    nbytes=scale_leaf.numel() * scale_leaf.element_size(),
                    dtype=scale_leaf.dtype,
                    shape=tuple(scale_leaf.shape),
                    role="scale",
                )
            else:
                raise ValueError("unsupported_quant_wrapper")
            w_spec = LeafSpec(
                offset=offsets[cursor - len(w_leaves) - len(b_leaves)],
                nbytes=w_leaves[0].numel() * w_leaves[0].element_size(),
                dtype=w_leaves[0].dtype,
                shape=tuple(w_leaves[0].shape),
                role=weight_role,
            )
            b_spec = None
            if b_leaves:
                b_leaf = b_leaves[0]
                b_offset = offsets[cursor - len(b_leaves)]
                b_spec = LeafSpec(
                    offset=b_offset,
                    nbytes=b_leaf.numel() * b_leaf.element_size(),
                    dtype=b_leaf.dtype,
                    shape=tuple(b_leaf.shape),
                    role="bias",
                )
            if repoint and module is not None:
                if leaf_kind == "float":
                    w_view = rebuilt[0]
                else:
                    w_view = _rebuild_from_leaves(
                        weight.data if isinstance(weight, torch.nn.Parameter) else weight,
                        iter(rebuilt[:len(w_leaves)]),
                    )
                module.weight = torch.nn.Parameter(
                    w_view,
                    requires_grad=getattr(weight, "requires_grad", False),
                )
                if bias is not None and b_spec is not None:
                    b_view = rebuilt[len(w_leaves)]
                    module.bias = torch.nn.Parameter(
                        b_view,
                        requires_grad=getattr(bias, "requires_grad", False),
                    )
            specs.append(
                LinearSpec(
                    name=name,
                    weight=w_spec,
                    bias=b_spec,
                    weight_requires_grad=getattr(weight, "requires_grad", False),
                    bias_requires_grad=getattr(bias, "requires_grad", False) if bias is not None else False,
                    kind=leaf_kind,
                    weight_scale=weight_scale_spec,
                    fp8_qualifies=fp8_qualifies,
                )
            )
    except Exception:
        pin_manager.release(pin_handle)
        raise
    linears_tuple = tuple(specs)
    pack = BlockPack(
        block_key=block_key,
        host_flat=host,
        linears=linears_tuple,
        required_pin_bytes=int(total),
        pinned=bool(pinned),
        fp8_flags=tuple(spec.fp8_qualifies for spec in linears_tuple),
        pin_handle=pin_handle,
        owns_flat=True,
        borrowed_from_arena=False,
    )
    pack.view_maker = make_block_view_maker(pack)
    return pack


class ArenaBorrowError(ValueError):
    """A block's live params don't actually live in the flat they were
    expected to borrow from -- caller must fall back to an owned pack."""


def pack_block_host_from_flat(block_key: str, linears, flat: torch.Tensor) -> "BlockPack":
    """Build a BORROWED BlockPack over an already-pinned flat someone else
    owns (the pinned-arena, ticket 534ea49's Slice 4): no allocation, no copy,
    no new pin grant. Offsets are derived from each leaf's REAL data_ptr
    relative to ``flat`` -- never trusted from a caller-supplied layout --
    so a leaf that turns out not to live in ``flat`` (stale/rebuilt block,
    a sibling Linear whose Parameter was replaced) fails closed with
    ArenaBorrowError instead of silently packing mixed storage.

    ``linears`` entries may use whatever naming convention the caller likes
    (e.g. ingraph's short per-block names, distinct from the arena's own
    full module-path names) -- only real storage identity is checked.
    """
    flat_storage = flat.untyped_storage()
    flat_ptr = flat.data_ptr()
    flat_end = flat_ptr + flat.numel() * flat.element_size()

    def _offset_of(leaf: torch.Tensor) -> int:
        if leaf.untyped_storage().data_ptr() != flat_storage.data_ptr():
            raise ArenaBorrowError(f"arena_layout_mismatch:{block_key}:not_in_flat")
        offset = leaf.data_ptr() - flat_ptr
        nbytes = leaf.numel() * leaf.element_size()
        if offset < 0 or offset + nbytes > flat_end - flat_ptr:
            raise ArenaBorrowError(f"arena_layout_mismatch:{block_key}:out_of_range")
        return offset

    specs = []
    for entry in linears:
        if len(entry) == 2:
            name, module = entry
            weight = module.weight
            bias = getattr(module, "bias", None)
        else:
            name, weight, bias = entry
        w_data = weight.data if isinstance(weight, torch.nn.Parameter) else weight
        b_data = bias.data if isinstance(bias, torch.nn.Parameter) else bias
        w_leaves = _flatten_leaves(w_data)
        b_leaves = _flatten_leaves(b_data) if b_data is not None else []

        kind = "float"
        weight_scale_spec = None
        fp8_qualifies = False
        if len(w_leaves) == 1:
            weight_role = "float_weight"
        elif (
            len(w_leaves) == 2
            and w_leaves[0].dtype == torch.float8_e4m3fn
            and w_leaves[1].is_floating_point()
        ):
            kind = "fp8_rowwise"
            weight_role = "qdata"
            scale_leaf = w_leaves[1]
            fp8_qualifies = _fp8_rowwise_qualifies(w_leaves[0], scale_leaf)
            weight_scale_spec = LeafSpec(
                offset=_offset_of(scale_leaf),
                nbytes=scale_leaf.numel() * scale_leaf.element_size(),
                dtype=scale_leaf.dtype,
                shape=tuple(scale_leaf.shape),
                role="scale",
            )
        else:
            raise ArenaBorrowError(f"arena_layout_mismatch:{block_key}:unsupported_quant_wrapper")
        w_leaf = w_leaves[0]
        w_spec = LeafSpec(
            offset=_offset_of(w_leaf),
            nbytes=w_leaf.numel() * w_leaf.element_size(),
            dtype=w_leaf.dtype,
            shape=tuple(w_leaf.shape),
            role=weight_role,
        )
        b_spec = None
        if b_leaves:
            b_leaf = b_leaves[0]
            b_spec = LeafSpec(
                offset=_offset_of(b_leaf),
                nbytes=b_leaf.numel() * b_leaf.element_size(),
                dtype=b_leaf.dtype,
                shape=tuple(b_leaf.shape),
                role="bias",
            )
        specs.append(
            LinearSpec(
                name=name,
                weight=w_spec,
                bias=b_spec,
                weight_requires_grad=getattr(weight, "requires_grad", False),
                bias_requires_grad=getattr(bias, "requires_grad", False) if bias is not None else False,
                kind=kind,
                weight_scale=weight_scale_spec,
                fp8_qualifies=fp8_qualifies,
            )
        )
    linears_tuple = tuple(specs)
    pack = BlockPack(
        block_key=block_key,
        host_flat=flat,
        linears=linears_tuple,
        required_pin_bytes=int(flat.numel() * flat.element_size()),
        # cudaHostRegister'd arena flats report is_pinned()==False (torch only
        # tracks its own caching-allocator pins); consult the registration
        # table too or every register-mechanism borrow falsely reads pageable.
        pinned=bool(pin_manager.is_host_pinned(flat)),
        fp8_flags=tuple(spec.fp8_qualifies for spec in linears_tuple),
        pin_handle=None,
        owns_flat=False,
        borrowed_from_arena=True,
    )
    pack.view_maker = make_block_view_maker(pack)
    return pack


class IngraphPackError(RuntimeError):
    """A block pack could not be built or borrowed. ``reasons`` carries the
    stable fail-closed tokens callers surface as ``_ingraph_unavailable_reasons``
    (``non_pinned_pack``, ``unsupported_quant_wrapper``, ``wrapper_pack_missing``,
    ``arena_borrow_required``)."""

    def __init__(self, reasons, message: str = ""):
        self.reasons = tuple(dict.fromkeys(reasons))
        super().__init__(message or ",".join(self.reasons))


@dataclass
class PackBuildResult:
    # Keyed by the model's STABLE block_key string, never a positional index --
    # the caller maps its own indices back locally.
    packs: "dict[str, BlockPack]"
    borrowed: int
    owned: int
    pageable: int  # always 0 on success (a pageable pack raises non_pinned_pack)
    reasons: tuple = ()


def build_or_borrow_block_packs(
    arena,
    entries_by_block: dict,
    *,
    repoint: bool = False,
    pin_mechanism: str = "register",
    allow_owned_fallback: bool = True,
) -> PackBuildResult:
    """Borrow each block's pack from the pinned arena, else build an owned one.

    The single place the in-graph pack policy lives, shared by every model's
    ``enable_ingraph_sampling`` / ``enable_ingraph_training`` glue (see the
    "in-graph arena protocol" in ``pinned_arena``). Nothing here knows about any
    particular model: ``entries_by_block`` maps a stable ``block_key`` to that
    block's ``(name, module)`` linear entries, and ``arena`` is duck-typed (any
    object exposing ``try_borrow_pack(block_key, entries)``), so this module
    never imports ``pinned_arena`` -- which imports it.

    Policy, centralized so callers cannot re-implement it inconsistently:

    * Borrow when the arena holds a current, pinned flat for the block: zero
      alloc, zero copy, no second pin of the same bytes.
    * Otherwise build an owned pack, but only if ``allow_owned_fallback``.
      Under strict pinned-arena validation the caller passes False so a silent
      fall back to owned packs cannot make a run "pass" without proving a
      single borrow.
    * Every streamed pack must be pinned; a pageable one fails the whole set
      closed (``non_pinned_pack``) -- strict in-graph is all-or-nothing.
    * On any failure, release ONLY packs we own. ``release_pack`` no-ops on a
      borrowed pack (``owns_flat=False``), so the arena's flats are never freed
      out from under it.
    """
    packs: "dict[str, BlockPack]" = {}
    borrowed = 0
    owned = 0
    try:
        for block_key, raw_entries in entries_by_block.items():
            entries = list(raw_entries)
            pack = arena.try_borrow_pack(block_key, entries) if arena is not None else None
            if pack is not None:
                borrowed += 1
            else:
                if not allow_owned_fallback:
                    raise IngraphPackError(
                        ("arena_borrow_required",),
                        f"arena_borrow_required: block {block_key!r} is not "
                        "borrowable from the pinned arena",
                    )
                try:
                    pack = pack_block_host(
                        block_key,
                        entries,
                        repoint=repoint,
                        pin_mechanism=pin_mechanism,
                    )
                except ValueError as error:
                    message = str(error)
                    reason = (
                        "wrapper_pack_missing"
                        if "wrapper packing" in message
                        else "unsupported_quant_wrapper"
                    )
                    raise IngraphPackError(
                        (reason,), f"{reason} ({message})"
                    ) from error
                owned += 1
            packs[block_key] = pack
        if any(not pack.pinned for pack in packs.values()):
            raise IngraphPackError(("non_pinned_pack",))
    except BaseException:
        for pack in packs.values():
            release_pack(pack)
        raise
    return PackBuildResult(packs=packs, borrowed=borrowed, owned=owned, pageable=0)


def _flat_view(
    flat: torch.Tensor,
    offset: int,
    nbytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return flat[offset:offset + nbytes].view(dtype).reshape(shape)


def _flat_clone_view(
    flat: torch.Tensor,
    offset: int,
    nbytes: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return flat[offset:offset + nbytes].clone().view(dtype).reshape(shape)

def leaf_view(flat: torch.Tensor, spec: LeafSpec) -> torch.Tensor:
    return _flat_view(flat, spec.offset, spec.nbytes, spec.dtype, spec.shape)


def block_linear_views(flat: torch.Tensor, pack: BlockPack) -> dict[str, LinearView]:
    out = {}
    for spec in pack.linears:
        weight = leaf_view(flat, spec.weight)
        scale = None
        if spec.kind == "fp8_rowwise":
            if spec.weight_scale is None:
                raise RuntimeError(f"missing scale for {spec.name}")
            scale = leaf_view(flat, spec.weight_scale)
        bias = leaf_view(flat, spec.bias) if spec.bias is not None else None
        out[spec.name] = LinearView(spec=spec, weight=weight, bias=bias, scale=scale)
    return out


def make_block_view_maker(pack: BlockPack):
    """Return a flat-buffer view maker that yields only tensor tuples."""
    entries = []
    for spec in pack.linears:
        scale_spec = spec.weight_scale
        entries.append(
            (
                (
                    spec.weight.offset,
                    spec.weight.nbytes,
                    spec.weight.dtype,
                    spec.weight.shape,
                ),
                None if spec.bias is None else (
                    spec.bias.offset,
                    spec.bias.nbytes,
                    spec.bias.dtype,
                    spec.bias.shape,
                ),
                None if scale_spec is None else (
                    scale_spec.offset,
                    scale_spec.nbytes,
                    scale_spec.dtype,
                    scale_spec.shape,
                ),
            )
        )
    entries = tuple(entries)

    def view_maker(flat: torch.Tensor, _entries=entries):
        out = []
        for weight, bias, scale in _entries:
            w = _flat_view(flat, weight[0], weight[1], weight[2], weight[3])
            b = None if bias is None else _flat_clone_view(flat, bias[0], bias[1], bias[2], bias[3])
            s = None if scale is None else _flat_clone_view(flat, scale[0], scale[1], scale[2], scale[3])
            out.append((w, b, s))
        return tuple(out)

    return view_maker


def block_tensor_views(flat: torch.Tensor, pack: BlockPack) -> tuple:
    maker = pack.view_maker
    if maker is None:
        maker = make_block_view_maker(pack)
        pack.view_maker = maker
    return maker(flat)


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


@dataclass
class _Ticket:
    tid: int
    device_buffer: torch.Tensor
    ready_event: torch.cuda.Event | None
    free_event: torch.cuda.Event | None = None
    h2d_start: torch.cuda.Event | None = None
    h2d_end: torch.cuda.Event | None = None
    nbytes: int = 0


_STATE_LOCK = threading.Lock()
_TICKETS: dict[int, _Ticket] = {}
_LIVE: collections.deque[int] = collections.deque()
_NEXT_ID = 0
_DEPTH = 2
_TRANSFER_STREAMS: dict[torch.device, torch.cuda.Stream] = {}
_STATS = {
    "fetches": 0,
    "bytes": 0,
    "h2d_ms": 0.0,
    "wait_ms": 0.0,
    "depth_waits": 0,
}


def configure_fetch_runtime(*, depth: int = 2) -> None:
    global _DEPTH, _NEXT_ID
    _DEPTH = max(1, int(depth))
    with _STATE_LOCK:
        _TICKETS.clear()
        _LIVE.clear()
        _NEXT_ID = 0


def reset_fetch_stats() -> None:
    for key in _STATS:
        _STATS[key] = 0


def fetch_stats(reset: bool = False) -> dict:
    stats = dict(_STATS)
    if reset:
        reset_fetch_stats()
    return stats


def fetch_report(reset: bool = False) -> str | None:
    stats = fetch_stats(reset=reset)
    if not stats["fetches"]:
        return None
    gb = stats["bytes"] / 1024 ** 3
    return (
        f"[InGraphStream] fetches={int(stats['fetches'])} "
        f"bytes={gb:.2f} GiB h2d_ms={stats['h2d_ms']:.3f} "
        f"wait_ms={stats['wait_ms']:.3f} depth_waits={int(stats['depth_waits'])}"
    )


def _transfer_stream(device: torch.device):
    stream = _TRANSFER_STREAMS.get(device)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _TRANSFER_STREAMS[device] = stream
    return stream


def _reap_locked(block: bool = False):
    while _LIVE:
        ticket = _TICKETS.get(_LIVE[0])
        if ticket is None:
            _LIVE.popleft()
            continue
        if ticket.free_event is None:
            if not block:
                return
            raise RuntimeError("mm.fetch_start depth exceeded before fetch_free")
        if not ticket.free_event.query():
            if not block:
                return
            start = time.perf_counter()
            ticket.free_event.synchronize()
            _STATS["depth_waits"] += 1
            _STATS["wait_ms"] += (time.perf_counter() - start) * 1000.0
        _TICKETS.pop(ticket.tid, None)
        _LIVE.popleft()


def drain_fetch_runtime() -> int:
    """Abandon every outstanding fetch ticket (OOM-recovery path only).

    An OOM unwinds a forward between fetch_start and fetch_free, leaving
    tickets whose free_event never records; the next fetch_start then blocks
    on the depth limit and raises 'depth exceeded before fetch_free'. The
    recovery path (mid-denoise demote / full streamed transition) calls this
    AFTER the failed forward has fully unwound: nothing will consume the
    in-flight device buffers anymore, so waiting out the transfer streams and
    dropping the tickets is safe. Returns the number of tickets abandoned.
    """
    with _STATE_LOCK:
        for stream in _TRANSFER_STREAMS.values():
            stream.synchronize()
        abandoned = len(_LIVE)
        _LIVE.clear()
        _TICKETS.clear()
    return abandoned


def _fetch_start_impl(host_flat: torch.Tensor) -> torch.Tensor:
    if host_flat.device.type != "cpu":
        raise RuntimeError("mm.fetch_start expected a CPU host_flat tensor")
    if torch.cuda.is_available() and not pin_manager.is_host_pinned(host_flat):
        # is_host_pinned, not host_flat.is_pinned(): arena flats are pinned
        # in place with cudaHostRegister, which torch's is_pinned() does not
        # recognize (it only tracks its own caching-allocator pins).
        raise RuntimeError("mm.fetch_start expected a pinned host_flat tensor")
    device = torch.device("cuda")
    stream = _transfer_stream(device)
    with _STATE_LOCK:
        _reap_locked(block=False)
        if len(_LIVE) >= _DEPTH:
            _reap_locked(block=True)
        global _NEXT_ID
        tid = _NEXT_ID
        _NEXT_ID += 1
        _LIVE.append(tid)
    h2d_start = torch.cuda.Event(enable_timing=True)
    h2d_end = torch.cuda.Event(enable_timing=True)
    ready = torch.cuda.Event()
    with torch.cuda.stream(stream):
        h2d_start.record(stream)
        device_buffer = host_flat.to(device, non_blocking=True)
        ready.record(stream)
        h2d_end.record(stream)
    with _STATE_LOCK:
        _TICKETS[tid] = _Ticket(
            tid=tid,
            device_buffer=device_buffer,
            ready_event=ready,
            h2d_start=h2d_start,
            h2d_end=h2d_end,
            nbytes=host_flat.numel(),
        )
        _STATS["fetches"] += 1
        _STATS["bytes"] += int(host_flat.numel())
    return torch.tensor([tid], dtype=torch.int64)


@torch.library.custom_op("mm::fetch_start", mutates_args=())
def fetch_start(host_flat: torch.Tensor) -> torch.Tensor:
    return _fetch_start_impl(host_flat)


@fetch_start.register_fake
def _(host_flat):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm::fetch_start_after", mutates_args=("guard",))
def fetch_start_after(host_flat: torch.Tensor, guard: torch.Tensor) -> torch.Tensor:
    return _fetch_start_impl(host_flat)


@fetch_start_after.register_fake
def _(host_flat, guard):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm::fetch_start_gated", mutates_args=())
def fetch_start_gated(host_flat: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """fetch_start with a REAL data dependency on `gate`.

    fetch_start_after's guard is a declared mutation; that bookkeeping does
    not survive Inductor's scheduler/reinplacer, which may hoist backward
    re-fetches above frees (ring overrun). Here the gate is an ordinary
    input, so the dependency is genuine dataflow no scheduling stage can
    drop. Emitted by the post-grad ordering pass (ingraph_stream_scheduling);
    not intended for hand-written model code."""
    return _fetch_start_impl(host_flat)


@fetch_start_gated.register_fake
def _(host_flat, gate):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm::fetch_wait", mutates_args=())
def fetch_wait(token: torch.Tensor, nbytes: int) -> torch.Tensor:
    tid = int(token[0].item())
    ticket = _TICKETS.get(tid)
    if ticket is None:
        raise RuntimeError(f"mm.fetch_wait got unknown ticket {tid}")
    current = torch.cuda.current_stream()
    start = time.perf_counter()
    current.wait_event(ticket.ready_event)
    _STATS["wait_ms"] += (time.perf_counter() - start) * 1000.0
    ticket.device_buffer.record_stream(current)
    if ticket.h2d_start is not None and ticket.h2d_end is not None:
        try:
            ticket.h2d_end.synchronize()
            _STATS["h2d_ms"] += ticket.h2d_start.elapsed_time(ticket.h2d_end)
        except RuntimeError:
            pass
    return ticket.device_buffer


@fetch_wait.register_fake
def _(token, nbytes: int):
    return torch.empty(nbytes, dtype=torch.uint8, device="cuda")


def _fetch_free_impl(token: torch.Tensor) -> torch.Tensor:
    tid = int(token[0].item())
    ticket = _TICKETS.get(tid)
    if ticket is None:
        raise RuntimeError(f"mm.fetch_free got unknown ticket {tid}")
    ticket.free_event = torch.cuda.Event()
    ticket.free_event.record(torch.cuda.current_stream())
    return token.clone()


@torch.library.custom_op("mm::fetch_free", mutates_args=())
def fetch_free(token: torch.Tensor) -> torch.Tensor:
    return _fetch_free_impl(token)


@fetch_free.register_fake
def _(token):
    return token.clone()


@torch.library.custom_op("mm::fetch_free_after", mutates_args=("guard",))
def fetch_free_after(token: torch.Tensor, guard: torch.Tensor) -> torch.Tensor:
    return _fetch_free_impl(token)


@fetch_free_after.register_fake
def _(token, guard):
    return token.clone()


def _register_ordered_effects():
    """Pin the fetch ops to program order inside compiled graphs.

    Functionalized custom ops only carry data deps through their args; in the
    AOT backward graph each checkpoint unit's re-fetch depends only on the
    saved boundary activation (an immediately-available input), so Inductor
    may hoist all re-fetches above the frees -- exceeding the ring depth and
    deadlocking the host-side depth guard. Ordered effect tokens thread a
    dependency chain through every fetch op, enforcing eager program order in
    forward AND backward graphs (kernel-launch order only; stream overlap is
    unaffected)."""
    try:
        from torch._higher_order_ops.effects import (
            _EffectType,
            _register_effectful_op,
        )

        for op in (
            torch.ops.mm.fetch_start.default,
            torch.ops.mm.fetch_start_after.default,
            torch.ops.mm.fetch_wait.default,
            torch.ops.mm.fetch_free.default,
            torch.ops.mm.fetch_free_after.default,
        ):
            _register_effectful_op(op, _EffectType.ORDERED)
    except Exception as error:  # pragma: no cover - torch-version dependent
        raise RuntimeError(
            "in-graph streaming requires ordered-effect registration for its "
            f"fetch ops (torch internal API changed?): {error!r}"
        ) from error


# NOT registered at import time: in torch 2.12 ordered-effect tokens trip an
# internal token-erasure assertion inside the checkpoint HOP lowering
# (see tests/test_ingraph_training_ops.py, compiled xfail). Phase 4a S1 keeps
# this as the candidate ordering mechanism for the compiled trunk; call it
# explicitly once the HOP interaction is resolved (torch upgrade or flat-trunk
# design without the checkpoint HOP).


class _FreeOnBackwardFn(torch.autograd.Function):
    """Anchor a ticket's free event to the consuming block's BACKWARD.

    Training-mode counterpart of `fetch_free_after`: under checkpoint
    recompute, the block's backward (grad-input from the fetched weight
    views) is the true last reader of the ticket's device buffer, so a
    forward-side free lets the depth-K ring recycle the buffer under
    backward kernels that are still reading it (silent corruption).

    Wrap the block INPUT, not its output: this node's backward runs last
    in the block's backward (input side), i.e. after every weight-view
    read, and `fetch_free_after`'s declared guard mutation on the incoming
    grad keeps the free ordered after the kernels that produced it.
    """

    @staticmethod
    def forward(ctx, x, token):
        ctx.save_for_backward(token)
        return x

    @staticmethod
    def backward(ctx, grad_x):
        (token,) = ctx.saved_tensors
        if torch.compiler.is_compiling():
            # Declared guard mutation orders the free after the kernels that
            # produced grad_x; functionalization makes it version-safe.
            torch.ops.mm.fetch_free_after(token, grad_x)
        else:
            # Eager executes in program order -- and the guarded variant's
            # version bump on grad_x would trip autograd's version checks.
            torch.ops.mm.fetch_free(token)
        return grad_x, None


def free_on_backward(x: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
    """Defer a ticket's free to the consuming block's backward (training).

    Two fetch generations exist under non-reentrant checkpoint: the
    first-pass fetch (its views are dropped by the checkpoint hooks, so it
    is safe to free after the block's forward) and the recompute fetch
    (its views feed the real backward, so it must be freed after the
    block's backward). The token passed here is saved via
    ``save_for_backward`` -- checkpoint's saved-tensor machinery therefore
    swaps it for the RECOMPUTE generation's token automatically, and this
    node's backward frees exactly the ticket backward actually read.

    Canonical training block shape (see checkpoint_recompute_context):

        token = fetch_start_after(host, x)
        flat = fetch_wait(token, nbytes)
        ...views...
        x = free_on_backward(x, token)
        out = <block math>(x, views)
        if not in_recompute():
            torch.ops.mm.fetch_free_after(token, out)  # first-pass gen only
        return out
    """
    return _FreeOnBackwardFn.apply(x, token)


_IN_RECOMPUTE = threading.local()


def in_recompute() -> bool:
    """True while a checkpoint recompute pass (via checkpoint_recompute_context)
    is re-running the block fn."""
    return bool(getattr(_IN_RECOMPUTE, "value", False))


class _RecomputeMarker:
    def __enter__(self):
        self._prev = getattr(_IN_RECOMPUTE, "value", False)
        _IN_RECOMPUTE.value = True
        return self

    def __exit__(self, exc_type, exc, tb):
        _IN_RECOMPUTE.value = self._prev
        return False


def checkpoint_recompute_context():
    """``context_fn`` for torch.utils.checkpoint: null forward context, and a
    recompute context that flips in_recompute() so the block fn suppresses the
    first-pass forward free during recompute (the recompute ticket is freed by
    free_on_backward instead). EAGER ONLY -- compiled checkpoint requires
    TorchDispatchMode contexts; use compiled_checkpoint_context there."""
    return contextlib.nullcontext(), _RecomputeMarker()


def _compiled_free_policy(ctx, op, *args, **kwargs):
    from torch.utils.checkpoint import CheckpointPolicy

    if op in (
        torch.ops.mm.fetch_free_after.default,
        torch.ops.mm.fetch_free.default,
    ):
        # Keep the forward-side free OUT of the backward replay: replayed, it
        # would free the backward re-fetch's buffer before the grad kernels
        # read it. free_on_backward's op is the backward-side free.
        return CheckpointPolicy.MUST_SAVE
    if op in (
        torch.ops.mm.fetch_start.default,
        torch.ops.mm.fetch_start_after.default,
        torch.ops.mm.fetch_wait.default,
    ):
        # The design's core invariant: fetched weights are NEVER saved for
        # backward. PREFER_RECOMPUTE is advisory -- at Krea2 scale the
        # partitioner chose to save all 28 fetched flats (12.25 GiB -> OOM).
        return CheckpointPolicy.MUST_RECOMPUTE
    # Everything else replays in backward (full-checkpoint mode).
    return CheckpointPolicy.PREFER_RECOMPUTE


def compiled_checkpoint_context():
    """``context_fn`` for torch.utils.checkpoint under torch.compile."""
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    return create_selective_checkpoint_contexts(_compiled_free_policy)


def training_checkpoint_context():
    """Grad-mode checkpoint context for streamed ingraph blocks: dispatch-mode
    (SAC) contexts under compile, the in_recompute marker in eager. Both make
    the first-pass forward free stay out of the backward path so the ring is
    freed by free_on_backward at the true last read."""
    if torch.compiler.is_compiling():
        return compiled_checkpoint_context()
    return checkpoint_recompute_context()

