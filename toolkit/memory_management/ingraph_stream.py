"""In-graph weight streaming primitives.

This module contains the host-side flat block representation and the custom
ops used by the compile-visible streaming path. It is intentionally independent
from the legacy per-Linear forward hijack path so models can opt in block by
block.
"""

from __future__ import annotations

import collections
import itertools
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from toolkit.memory_management import pin_manager

from toolkit.memory_management.manager_modules import _fp8_linear_compiled



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


def _empty_host_flat(nbytes: int, *, pin: bool = True) -> tuple[torch.Tensor, bool]:
    if not pin:
        return torch.empty(nbytes, dtype=torch.uint8), False
    handle = pin_manager.pin_alloc(
        nbytes,
        "ingraph_pack",
        required=False,
        mode="sampling",
    )
    return handle.tensor, bool(handle.pinned)


def pack_block_host(block_key: str, linears, *, repoint: bool = True, pin: bool = True) -> BlockPack:
    """Pack a block's Linear weights/biases into one aligned host byte buffer.

    ``linears`` is an iterable of ``(name, module)`` or
    ``(name, weight, bias)`` entries. When ``repoint`` is true the modules'
    Parameters are replaced by views into the flat host buffer.
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
    host, pinned = _empty_host_flat(total, pin=pin)
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
            if kind == "float":
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
                kind=kind,
                weight_scale=weight_scale_spec,
                fp8_qualifies=fp8_qualifies,
            )
        )
    return BlockPack(
        block_key=block_key,
        host_flat=host,
        linears=tuple(specs),
        required_pin_bytes=int(total),
        pinned=bool(pinned),
    )


def leaf_view(flat: torch.Tensor, spec: LeafSpec) -> torch.Tensor:
    return flat[spec.offset:spec.offset + spec.nbytes].view(spec.dtype).reshape(spec.shape)


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


def functional_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None):
    if weight.dtype != x.dtype and weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
        weight = weight.to(dtype=x.dtype)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(dtype=x.dtype)
    return F.linear(x, weight, bias)


def streamed_linear(x: torch.Tensor, view: LinearView):
    if view.spec.kind == "fp8_rowwise" and view.spec.fp8_qualifies:
        if view.scale is None:
            raise RuntimeError(f"missing scale for {view.spec.name}")
        return _fp8_linear_compiled(x, view.weight.t(), view.scale.reshape(-1), view.bias)
    return functional_linear(x, view.materialized_weight(), view.bias)


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


def _fetch_start_impl(host_flat: torch.Tensor) -> torch.Tensor:
    if host_flat.device.type != "cpu":
        raise RuntimeError("mm.fetch_start expected a CPU host_flat tensor")
    if torch.cuda.is_available() and not host_flat.is_pinned():
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

