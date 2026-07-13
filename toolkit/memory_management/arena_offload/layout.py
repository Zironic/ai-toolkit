"""Static host layout and canonical-arena packing primitives.

This module owns no CUDA stream, execution hook, trace, queue, or transfer
lifetime. It describes host storage, wrapper reconstruction, and typed views.
"""

from __future__ import annotations
import itertools

from dataclasses import dataclass
from typing import Iterable

import torch

from toolkit.memory_management import pin_manager

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
        # I1: prepare the page-aligned buffer WITHOUT registering it yet --
        # pack_block_host copies the leaves into it (ordinary pageable
        # memcpy) before the caller commits the cudaHostRegister pin. See
        # pin_register_prepare's docstring for why population-before-pin is
        # faster than registering a virgin buffer.
        candidate, padded = pin_manager.pin_register_prepare(nbytes)
        return candidate, False, ("register_pending", padded, kind)
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
    register_pending = isinstance(pin_handle, tuple) and pin_handle[:1] == ("register_pending",)
    try:
        for leaf, offset in zip(leaves, offsets):
            nbytes = leaf.numel() * leaf.element_size()
            host[offset:offset + nbytes].view(leaf.dtype).reshape(leaf.shape).copy_(leaf)

        if register_pending:
            _, _padded, register_kind = pin_handle
            pin_handle = pin_manager.pin_register_commit(
                host, total, register_kind, required=False
            )
            host = pin_handle.tensor
            pinned = bool(pin_handle.pinned)

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


def is_streamed_module(module) -> bool:
    """The memory manager's marker for "this Linear's weights live on the host
    and are fetched per call".

    Read it BEFORE stripping compile contaminants -- the strip deletes the
    attribute, after which every leaf looks resident.
    """
    return hasattr(module, "_layer_memory_manager")


def resident_linear_tensors(module) -> "tuple[tuple, bool]":
    """``((weight, bias, scale), fp8_qualifies)`` from a Linear's live
    Parameters, wherever they are.

    Mirrors ``pack_block_host``'s leaf extraction so a resident leaf and a
    streamed leaf are interchangeable inputs to ``streamed_linear_tensors``: a
    streamed leaf's triple is a view into the fetched flat, a resident leaf's is
    the Parameter itself. Neither the block forward nor the LoRA fold can tell
    them apart.
    """
    weight = module.weight
    bias = getattr(module, "bias", None)
    w_data = weight.data if isinstance(weight, torch.nn.Parameter) else weight
    b_data = bias.data if isinstance(bias, torch.nn.Parameter) else bias
    leaves = _flatten_leaves(w_data)
    if len(leaves) == 1:
        return (leaves[0], b_data, None), False
    if (
        len(leaves) == 2
        and leaves[0].dtype == torch.float8_e4m3fn
        and leaves[1].is_floating_point()
    ):
        return (
            (leaves[0], b_data, leaves[1]),
            _fp8_rowwise_qualifies(leaves[0], leaves[1]),
        )
    raise ValueError("unsupported_quant_wrapper")


@dataclass(frozen=True)
class BlockLeafPlan:
    """Where each of a block's Linear leaves gets its weights this phase.

    The pack is a transfer-coalescing device (one H2D for N leaves), NOT a
    residency decision. The memory planner splits residency per-Linear, so a
    block is routinely part streamed / part resident. ``sources`` records, in
    the caller's canonical leaf order, whether each leaf reads from the fetched
    flat (``(True, i)`` -> ``streamed_views[i]``) or straight off its resident
    Parameter (``(False, i)`` -> ``resident_args[i]``). Both are trace-time
    constants, so the compiled block specializes on its residency pattern.

    ``pack is None`` means every leaf is resident: no flat, no fetch, no token.
    """

    block_key: str
    pack: "BlockPack | None"
    sources: tuple
    resident_args: tuple
    fp8_flags: tuple
    borrowed_from_arena: bool = False

    @property
    def streams(self) -> bool:
        return self.pack is not None


def assemble_leaf_args(plan: BlockLeafPlan, streamed_views: tuple = ()) -> tuple:
    """Interleave fetched views and resident Parameters back into the block's
    canonical leaf order. Pure Python over trace-time constants."""
    return tuple(
        streamed_views[index] if from_pack else plan.resident_args[index]
        for from_pack, index in plan.sources
    )


@dataclass
class BlockPlanResult:
    plans: "dict[str, BlockLeafPlan]"
    borrowed: int
    owned: int
    fully_resident: int
    streamed_leaves: int
    resident_leaves: int
    reasons: tuple = ()


def build_block_leaf_plans(
    arena,
    entries_by_block: dict,
    *,
    is_streamed=is_streamed_module,
    repoint: bool = False,
    pin_mechanism: str = "register",
    allow_owned_fallback: bool = True,
) -> BlockPlanResult:
    """Plan every block's leaves, packing only the ones the manager streams.

    ``entries_by_block`` maps a stable ``block_key`` to that block's FULL
    ``(name, module)`` leaf list in canonical order. This splits each block by
    ``is_streamed``, builds/borrows a pack over the streamed subset only, and
    reads the resident leaves straight off their Parameters.

    Packing only the streamed subset is what lets the trunk coexist with the
    planner's per-Linear residency: a partially-resident block yields a smaller
    flat (so a smaller prefetch ring) and skips the fetch entirely for leaves
    already on the device. Asking the arena for leaves it never offloaded is
    what produced ``borrow refused: stale_modules=3/8``.
    """
    streamed_by_block: dict = {}
    for block_key, raw_entries in entries_by_block.items():
        streamed = [(name, module) for name, module in raw_entries if is_streamed(module)]
        if streamed:
            streamed_by_block[block_key] = streamed

    result = build_or_borrow_block_packs(
        arena,
        streamed_by_block,
        repoint=repoint,
        pin_mechanism=pin_mechanism,
        allow_owned_fallback=allow_owned_fallback,
    )
    try:
        plans: "dict[str, BlockLeafPlan]" = {}
        streamed_leaves = 0
        resident_leaves = 0
        for block_key, raw_entries in entries_by_block.items():
            pack = result.packs.get(block_key)
            stream_index = {
                name: index
                for index, (name, _) in enumerate(streamed_by_block.get(block_key, ()))
            }
            sources = []
            resident_args = []
            fp8_flags = []
            for name, module in raw_entries:
                index = stream_index.get(name)
                if index is not None:
                    sources.append((True, index))
                    fp8_flags.append(pack.fp8_flags[index])
                    streamed_leaves += 1
                    continue
                try:
                    triple, qualifies = resident_linear_tensors(module)
                except ValueError as error:
                    raise IngraphPackError(
                        ("unsupported_quant_wrapper",),
                        f"unsupported_quant_wrapper ({block_key}.{name}: {error})",
                    ) from error
                sources.append((False, len(resident_args)))
                resident_args.append(triple)
                fp8_flags.append(qualifies)
                resident_leaves += 1
            plans[block_key] = BlockLeafPlan(
                block_key=block_key,
                pack=pack,
                sources=tuple(sources),
                resident_args=tuple(resident_args),
                fp8_flags=tuple(fp8_flags),
                borrowed_from_arena=bool(pack is not None and pack.borrowed_from_arena),
            )
    except BaseException:
        for pack in result.packs.values():
            release_pack(pack)
        raise
    return BlockPlanResult(
        plans=plans,
        borrowed=result.borrowed,
        owned=result.owned,
        fully_resident=sum(1 for plan in plans.values() if not plan.streams),
        streamed_leaves=streamed_leaves,
        resident_leaves=resident_leaves,
    )


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




@dataclass(frozen=True)
class LeafDescriptor:
    role: str
    offset: int
    nbytes: int
    dtype: torch.dtype
    shape: tuple[int, ...]


@dataclass(frozen=True)
class LinearLayout:
    name: str
    leaf_descriptors: tuple[LeafDescriptor, ...]
    weight_leaf_count: int
    weight_requires_grad: bool
    bias_requires_grad: bool
    native_fp8_eligible: bool
    weight_template: torch.Tensor

    def leaf(self, role: str) -> LeafDescriptor | None:
        return next((leaf for leaf in self.leaf_descriptors if leaf.role == role), None)


@dataclass(frozen=True)
class BlockLayout:
    block_key: str
    linears: tuple[LinearLayout, ...]
    nbytes: int


def flatten_leaves(value: torch.Tensor) -> list[torch.Tensor]:
    try:
        names, _ = value.__tensor_flatten__()
    except Exception:
        return [value]
    leaves = []
    for name in names:
        child = getattr(value, name, None)
        if child is not None:
            leaves.extend(flatten_leaves(child))
    return leaves


def rebuild_from_leaves(template: torch.Tensor, leaves: Iterable[torch.Tensor]):
    iterator = iter(leaves)

    def rebuild(value):
        try:
            names, context = value.__tensor_flatten__()
        except Exception:
            return next(iterator)
        children = {}
        for name in names:
            child = getattr(value, name, None)
            children[name] = None if child is None else rebuild(child)
        return type(value).__tensor_unflatten__(children, context, value.size(), value.stride())

    return rebuild(template)


def _fp8_eligible(qdata: torch.Tensor, scale: torch.Tensor) -> bool:
    return bool(
        hasattr(torch, "_scaled_mm")
        and qdata.dtype == torch.float8_e4m3fn
        and qdata.ndim == 2
        and scale.numel() == qdata.shape[0]
        and qdata.shape[0] % 16 == 0
        and qdata.shape[1] % 16 == 0
    )


def inspect_block(block_key: str, entries) -> BlockLayout:
    cursor = 0
    linears = []
    for name, module in entries:
        weight = module.weight
        bias = getattr(module, "bias", None)
        weight_data = weight.data if isinstance(weight, torch.nn.Parameter) else weight
        bias_data = bias.data if isinstance(bias, torch.nn.Parameter) else bias
        weight_leaves = flatten_leaves(weight_data)
        bias_leaves = [] if bias_data is None else flatten_leaves(bias_data)
        if len(weight_leaves) == 1:
            roles = ["weight"]
            native_fp8 = False
        elif (
            len(weight_leaves) == 2
            and weight_leaves[0].dtype == torch.float8_e4m3fn
            and weight_leaves[1].is_floating_point()
        ):
            roles = ["weight", "scale"]
            native_fp8 = _fp8_eligible(*weight_leaves)
        else:
            raise ValueError(f"unsupported_quant_wrapper:{block_key}:{name}")
        if len(bias_leaves) > 1:
            raise ValueError(f"unsupported_bias_wrapper:{block_key}:{name}")
        roles.extend("bias" for _ in bias_leaves)
        descriptors = []
        for role, leaf in zip(roles, weight_leaves + bias_leaves):
            cursor = (cursor + LEAF_ALIGN - 1) // LEAF_ALIGN * LEAF_ALIGN
            nbytes = leaf.numel() * leaf.element_size()
            descriptors.append(LeafDescriptor(role, cursor, nbytes, leaf.dtype, tuple(leaf.shape)))
            cursor += nbytes
        linears.append(LinearLayout(
            name=name,
            leaf_descriptors=tuple(descriptors),
            weight_leaf_count=len(weight_leaves),
            weight_requires_grad=bool(weight.requires_grad),
            bias_requires_grad=bool(bias.requires_grad) if bias is not None else False,
            native_fp8_eligible=native_fp8,
            weight_template=weight_data,
        ))
    return BlockLayout(block_key, tuple(linears), cursor)


def typed_view(flat: torch.Tensor, leaf: LeafDescriptor) -> torch.Tensor:
    return flat[leaf.offset:leaf.offset + leaf.nbytes].view(leaf.dtype).reshape(leaf.shape)


def linear_views(flat: torch.Tensor, layout: LinearLayout):
    views = tuple(typed_view(flat, leaf) for leaf in layout.leaf_descriptors)
    weight_views = views[:layout.weight_leaf_count]
    weight = (
        weight_views[0]
        if layout.weight_leaf_count == 1
        else rebuild_from_leaves(layout.weight_template, weight_views)
    )
    bias = views[layout.weight_leaf_count] if len(views) > layout.weight_leaf_count else None
    return weight, bias
