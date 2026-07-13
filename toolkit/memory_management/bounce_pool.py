"""
Slice 2B: bounded pinned CPU bounce pool.

Slice-1 measured ~134s/step of *pageable* submit stall: the training thread
blocks inside the H2D enqueue while Windows stages (and sometimes page-faults
from the pagefile) the pageable FP8 source weights. The ring depth is fine --
compute-wait was 0.8s -- so the fix is not a deeper GPU pipeline, it is moving
the pageable->pinned copy off the training thread.

This pool keeps the canonical weights pageable (as today) and, on background
worker threads, copies the upcoming layers' ordered storage leaves into a
bounded pool of reusable *pinned* buffers ahead of time. When the training
thread reaches a layer it hands the H2D a pinned source, which transfers async
without blocking. The "upcoming layers" come from the frozen slice-2A trace.

Lifecycle of one pinned slot:

    FREE -> CPU_FILLING -> CPU_READY -> IN_USE -> (H2D done) -> FREE

The pinned buffer backing a slot may not be reused until the async H2D that
read it has completed, so release() attaches a CUDA event recorded on the
transfer stream and the worker only reclaims the buffer once that event fires.

This module is CUDA-only in production (pinned memory needs CUDA). With no CUDA
it degrades to plain tensors so the scheduling/state logic stays unit-testable.
Everything here is inert unless a pool is explicitly created for a device.
"""

import collections
import json
import os
import threading
import time
import weakref
from typing import Optional

import torch
from . import pin_manager


dxgi_meminfo = None

# device -> PinnedBouncePool. The autograd staging path looks the active pool
# up here; absent an entry it behaves exactly as before.
_DEVICE_PREFETCH: dict = {}
_TRACE_CAPTURE_PATH = os.environ.get("AI_TOOLKIT_BOUNCE_TRACE_CAPTURE", "").strip()
_TRACE_CAPTURE_STEPS = max(
    0, int(os.environ.get("AI_TOOLKIT_BOUNCE_TRACE_CAPTURE_STEPS", "256"))
)

# Slot states.
FREE = "FREE"
CPU_FILLING = "CPU_FILLING"
CPU_READY = "CPU_READY"
IN_USE = "IN_USE"

HIT = "hit"
SOFT_MISS = "soft_miss"   # worker was mid-copy; we waited for it (still off-thread copy)
HARD_MISS = "hard_miss"   # not staged at all; fall back to the pageable source


def configure_trace_capture(path=None, steps=None):
    """Configure optional JSONL trace capture for current and future pools."""
    global _TRACE_CAPTURE_PATH, _TRACE_CAPTURE_STEPS
    _TRACE_CAPTURE_PATH = str(path or "").strip()
    if steps is not None:
        _TRACE_CAPTURE_STEPS = max(0, int(steps))
    for pool in list(_DEVICE_PREFETCH.values()):
        try:
            pool.configure_capture(_TRACE_CAPTURE_PATH, _TRACE_CAPTURE_STEPS)
        except Exception:
            pass


def _pin_enabled() -> bool:
    return torch.cuda.is_available()


def _is_pinned(t) -> bool:
    """True if a CPU tensor (or every leaf of a tensor-subclass wrapper, e.g.
    a quantized weight) is already page-locked. Mirrors manager_modules's
    _profile_is_pinned (False for None -- callers treat an absent bias as "no
    obstacle" explicitly); duplicated locally to avoid a circular import (that
    module imports from this one)."""
    if t is None:
        return False
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        try:
            # See manager_modules._profile_is_pinned: register-pinned arena
            # flat views report is_pinned()==False, so consult the arena
            # storage set too.
            return t.device.type == "cpu" and (
                t.is_pinned() or pin_manager.is_arena_backed(t)
            )
        except Exception:
            return False
    leaves = [getattr(t, name, None) for name in names]
    leaves = [leaf for leaf in leaves if leaf is not None]
    return bool(leaves) and all(_is_pinned(leaf) for leaf in leaves)


try:
    import psutil as _psutil
except Exception:
    _psutil = None


def _host_ram_available() -> Optional[int]:
    """Available physical RAM in bytes, or None if it cannot be determined."""
    if _psutil is None:
        return None
    try:
        return _psutil.virtual_memory().available
    except Exception:
        return None


# Process-wide ledger of bytes actually page-locked via cudaHostAlloc, shared
# across every pinning subsystem in this process: this pool's bounce buffers
# AND MemoryManager's permanent weight pins (training and sampling managers
# alike). They draw on the SAME finite resource -- on Windows/WDDM, pinned
# memory commits against the GPU's shared-memory budget (roughly a fraction of
# total RAM, opaque to psutil: "available RAM" can look fine while this
# ceiling is already exhausted). Sizing each subsystem's budget independently
# is what let training crash with a raw cudaErrorMemoryAllocation well after a
# generous auto-pin budget passed its own isolated checks -- the bounce pool's
# separate buffer budget pushed the *combined* total past the real ceiling.
# Any code path that calls .pin_memory() / pin_memory=True must register the
# bytes here, and release them when the pinning is undone.
#
# The ledger itself now lives in pin_manager (per-consumer-class); these
# functions are delegating shims kept for external callers, and the old
# module-global `_pinned_bytes_total` is served read-only via __getattr__
# below. Do not assign to it.


def register_pinned_bytes(n: int, kind: str = "unknown") -> None:
    if n <= 0:
        return
    pin_manager.register_pinned_bytes(int(n), kind=kind)


def release_pinned_bytes(n: int, kind: str = "unknown") -> None:
    if n <= 0:
        return
    pin_manager.release_pinned_bytes(int(n), kind=kind)


def __getattr__(name):
    if name == "_pinned_bytes_total":
        return pin_manager.total_pinned_bytes()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


def _cuda_device_index(device=None) -> int:
    if device is None:
        return 0
    try:
        dev = torch.device(device)
    except Exception:
        return 0
    if dev.type != "cuda":
        return 0
    if dev.index is not None:
        return int(dev.index)
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


# Shared-budget spill reserve (the WDDM NON_LOCAL margin kept free for cliff-1's
# overflow valve and the sampling-transition spike). Percentage-based:
#   margin = max(floor, pct * NON_LOCAL_Budget)
# so it scales with the OS-assigned budget instead of being a flat guess. The
# job config drives it at attach via set_spill_reserve_policy(); env vars are the
# fallback/test override only (see CLAUDE.md Training Configuration Rule).
_SPILL_RESERVE_FLOOR_GIB_OVERRIDE: Optional[float] = None
_SPILL_RESERVE_PCT_OVERRIDE: Optional[float] = None


def set_spill_reserve_policy(
    floor_gib: Optional[float] = None, pct: Optional[float] = None
) -> None:
    """Set the process-wide shared-budget spill-reserve policy (from job config).

    ``None`` leaves the corresponding term on its env/default value. Passing a
    value pins it for the process, so a real training run's margin comes from the
    resolved job config rather than an env var.
    """
    global _SPILL_RESERVE_FLOOR_GIB_OVERRIDE, _SPILL_RESERVE_PCT_OVERRIDE
    pin_manager.set_spill_reserve_policy(floor_gib=floor_gib, pct=pct)

    if floor_gib is not None:
        _SPILL_RESERVE_FLOOR_GIB_OVERRIDE = max(0.0, float(floor_gib))
    if pct is not None:
        _SPILL_RESERVE_PCT_OVERRIDE = max(0.0, float(pct))


def _spill_reserve_floor_gib() -> float:
    if _SPILL_RESERVE_FLOOR_GIB_OVERRIDE is not None:
        return _SPILL_RESERVE_FLOOR_GIB_OVERRIDE
    # Back-compat: the sensor plan's flat AI_TOOLKIT_WDDM_SPILL_RESERVE_GIB, if
    # set, still acts as the floor. Otherwise the percentage-based default floor.
    for name, default in (
        ("AI_TOOLKIT_WDDM_SPILL_RESERVE_FLOOR_GIB", None),
        ("AI_TOOLKIT_WDDM_SPILL_RESERVE_GIB", "1.0"),
    ):
        raw = os.environ.get(name)
        if raw is None:
            if default is None:
                continue
            raw = default
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            continue
    return 2.0


def _spill_reserve_pct() -> float:
    if _SPILL_RESERVE_PCT_OVERRIDE is not None:
        return _SPILL_RESERVE_PCT_OVERRIDE
    try:
        return max(0.0, float(os.environ.get("AI_TOOLKIT_WDDM_SPILL_RESERVE_PCT", "0.10")))
    except (TypeError, ValueError):
        return 0.20


def dxgi_spill_reserve_bytes(budget_bytes: Optional[int] = None) -> int:
    """Bytes of NON_LOCAL budget to keep free as the cliff-1/sampling spill margin.

    ``margin = max(floor, pct * budget)``. When ``budget_bytes`` is None or
    non-positive (caller doesn't have the DXGI budget handy) the percentage term
    drops out and the flat floor is used.
    """
    gib = 1024 ** 3
    floor_bytes = int(_spill_reserve_floor_gib() * gib)
    if budget_bytes and int(budget_bytes) > 0:
        pct_bytes = int(_spill_reserve_pct() * float(budget_bytes))
        return max(floor_bytes, pct_bytes)
    return floor_bytes


def get_dxgi_meminfo():
    if _env_bool("AI_TOOLKIT_WDDM_DXGI_DISABLE"):
        return None
    global dxgi_meminfo
    if dxgi_meminfo is None:
        try:
            from . import dxgi_meminfo as _dxgi_meminfo
        except Exception:
            return None
        dxgi_meminfo = _dxgi_meminfo
    return dxgi_meminfo


def dxgi_pinned_headroom(cuda_device_index: Optional[int] = None) -> Optional[int]:
    """Pinnable headroom from the real DXGI NON_LOCAL budget probe, or None
    when the probe is unavailable/disabled. This is the authoritative signal;
    system-RAM proxies exist only for when this returns None."""
    if _env_bool("AI_TOOLKIT_WDDM_DXGI_CONTROL_DISABLE"):
        return None
    dxgi = get_dxgi_meminfo()
    if dxgi is None:
        return None
    info = dxgi.query_non_local_video_memory_info(
        cuda_device_index=0 if cuda_device_index is None else int(cuda_device_index),
        min_interval_s=0.0,
    )
    if info is None:
        return None
    return dxgi.compute_non_local_headroom_bytes(
        info.budget_bytes,
        info.current_usage_bytes,
        dxgi_spill_reserve_bytes(info.budget_bytes),
    )


def pinned_bytes_headroom(cuda_device_index: Optional[int] = None) -> Optional[int]:
    """Bytes still safe to pin process-wide before the WDDM shared-GPU-memory
    budget is likely exhausted. None if psutil is unavailable (check skipped)
    or the proxy is disabled (AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION <= 0)."""
    headroom = dxgi_pinned_headroom(cuda_device_index)
    if headroom is not None:
        return headroom
    if _psutil is None:
        return None
    try:
        total = _psutil.virtual_memory().total
    except Exception:
        return None
    fraction = float(
        os.environ.get("AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION", "0.25")
    )
    if fraction <= 0:
        return None
    ceiling = int(total * fraction)
    return max(0, ceiling - pin_manager.total_pinned_bytes())


def _leaf_specs(t) -> list:
    """Depth-first (dtype, shape) of every physical leaf of a (maybe wrapper) tensor."""
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        return [(t.dtype, tuple(t.shape))]
    specs = []
    for name in names:
        inner = getattr(t, name, None)
        if inner is not None:
            specs.extend(_leaf_specs(inner))
    return specs


def _signature(t) -> tuple:
    """Hashable identity of a tensor's physical layout, for buffer pooling."""
    return tuple(_leaf_specs(t))


def _layer_specs(weight, bias) -> list:
    """Combined leaf specs for a layer's weight (+bias), in copy order."""
    specs = _leaf_specs(weight)
    if bias is not None:
        specs = specs + _leaf_specs(bias)
    return specs


def _shape_key_label(shape_key) -> str:
    if shape_key is None:
        return "none"
    text = repr(shape_key)
    if len(text) > 160:
        return text[:157] + "..."
    return text



def _jsonable_entry(entry):
    if isinstance(entry, tuple):
        return list(entry)
    return entry


def _schedule_layer_key(entry):
    """Layer source key for a schedule entry.

    Cold schedules are raw layer keys. Trace schedules are semantic entries:
    (layer_key, operation, occurrence).
    """
    if isinstance(entry, tuple) and entry:
        return entry[0]
    return entry


def _make_access_key(layer_key, operation="forward", occurrence=None):
    if occurrence is None:
        return (layer_key, operation)
    return (layer_key, operation, occurrence)


def _entry_matches_access(entry, access_key):
    if isinstance(entry, tuple):
        return entry == access_key
    return entry == access_key[0]


def _entry_same_layer(entry, layer_key):
    return _schedule_layer_key(entry) == layer_key


def _slot_matches_request(slot, layer_key, weight, bias) -> bool:
    if slot is None:
        return False
    if slot.layer_key != layer_key:
        return False
    try:
        return slot.signature == tuple(_layer_specs(weight, bias))
    except Exception:
        return False


def _spec_bytes(specs) -> int:
    total = 0
    for dtype, shape in specs:
        n = 1
        for s in shape:
            n *= s
        total += n * torch.empty((), dtype=dtype).element_size()
    return total


def _alloc_pinned(specs) -> tuple[list, int]:
    if not _pin_enabled():
        return [torch.empty(shape, dtype=dtype) for (dtype, shape) in specs], 0
    leaves = []
    pinned_bytes = 0
    try:
        for dtype, shape in specs:
            leaf, pinned = pin_manager.pin_empty(
                shape,
                dtype,
                "bounce",
                required=True,
            )
            leaves.append(leaf)
            if pinned:
                pinned_bytes += leaf.numel() * leaf.element_size()
    except Exception:
        # A leaf mid-signature failed: the earlier leaves' grants would leak
        # (their tensors are discarded here, and accounting is explicit).
        if pinned_bytes:
            release_pinned_bytes(pinned_bytes, kind="bounce")
        raise
    return leaves, pinned_bytes


def _rebuild_into(src, leaves_iter):
    """Copy src (leaf or wrapper) into the next preallocated pinned leaves and
    return a tensor of the same type/metadata backed by that pinned storage."""
    try:
        names, ctx = src.__tensor_flatten__()
    except Exception:
        dest = next(leaves_iter)
        dest.copy_(src)
        return dest
    moved = {}
    for name in names:
        inner = getattr(src, name, None)
        moved[name] = None if inner is None else _rebuild_into(inner, leaves_iter)
    return type(src).__tensor_unflatten__(moved, ctx, src.size(), src.stride())


class _Slot:
    __slots__ = (
        "position",
        "layer_key",
        "state",
        "signature",
        "leaves",
        "weight",
        "bias",
        "nbytes",
        "ready_event",
        "reclaim_event",
    )

    def __init__(self):
        self.position = -1
        self.layer_key = None
        self.state = FREE
        self.signature = None
        self.leaves = None
        self.weight = None
        self.bias = None
        self.nbytes = 0
        self.ready_event = threading.Event()
        self.reclaim_event = None  # CUDA event; buffer reusable once it fires


class _Ticket:
    __slots__ = ("slot",)

    def __init__(self, slot):
        self.slot = slot


class PinnedBouncePool:
    """Bounded pinned bounce buffers fed by background workers from a trace."""

    def __init__(
        self,
        device,
        budget_bytes: int,
        lookahead: int = 8,
        target_ready_bytes: Optional[int] = None,
        num_workers: int = 2,
        ram_floor_bytes: int = 2 * 1024 ** 3,
        fill_group_size: Optional[int] = None,
    ):
        self.device = torch.device(device)
        self.budget_bytes = int(budget_bytes)
        self.lookahead = int(lookahead)
        self.max_lookahead_positions = self.lookahead
        # Block streaming: how many schedulable positions a worker claims and
        # publishes per lock cycle. 1 = per-Linear (default). Set to a block's
        # Linear count to amortize the lock/CV/slot-dict overhead across a whole
        # block instead of paying it per Linear (the "small requests" cost). The
        # copies themselves are still per-Linear and happen outside the lock.
        if fill_group_size is None:
            fill_group_size = int(os.environ.get("AI_TOOLKIT_BOUNCE_FILL_GROUP", "1"))
        self.fill_group_size = max(1, int(fill_group_size))
        if target_ready_bytes is None:
            target_ready_bytes = int(
                float(os.environ.get("AI_TOOLKIT_BOUNCE_TARGET_READY_GIB", "3.5"))
                * 1024 ** 3
            )
        self.target_ready_bytes = int(target_ready_bytes)
        self.num_workers = max(1, int(num_workers))
        self.ram_floor_bytes = int(ram_floor_bytes)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._sources: dict = {}        # layer_key -> weakref(module)
        self._scheduled: list = []      # positional layer keys or semantic access tuples
        self._observed_step: list = []  # actual access order from previous step
        self._access_occurrences = collections.Counter()
        self.schedule_version = -1      # trace version this schedule came from
        self.schedule_shape_key = None  # execution shape this schedule came from
        self.schedule_confidence = "cold"
        self._slots: dict = {}          # position -> _Slot
        self._skipped_positions: set = set()
        self._free_buffers: dict = {}   # signature -> list[leaves]
        self._inflight_bytes = 0
        self._consume_pos = 0
        self._fill_pos = 0
        self._stop = False
        self._epoch = 0
        self._capture_path = _TRACE_CAPTURE_PATH
        self._capture_limit = _TRACE_CAPTURE_STEPS
        self._capture_count = 0

        # stats
        self.hits = 0
        self.soft_misses = 0
        self.hard_misses = 0
        self.cpu_wait_s = 0.0
        self.copy_s = 0.0
        self.copy_bytes = 0
        # Worker-side request accounting (the "small requests by the workers").
        # ``fills`` = individual Linear pageable->pinned copies published.
        # ``fill_batches`` = worker claim/publish cycles. With fill_group_size>1
        # one batch publishes a whole block, so fills/fill_batches ~= block size
        # and fill_batches is the per-step worker lock-cycle count Slice 1 cuts.
        self.fills = 0
        self.fill_batches = 0
        self.skips = 0
        self.resyncs = 0
        self.mismatches = 0
        self.duplicate_key_resync_blocked = 0

        self._workers = [
            threading.Thread(target=self._worker_loop, daemon=True,
                             name=f"bounce-{self.device}-{i}")
            for i in range(self.num_workers)
        ]
        for w in self._workers:
            w.start()
        pin_manager.register_evictable(self.shrink)


    # -- registration & schedule ------------------------------------------

    def configure_capture(self, path=None, steps=None):
        with self._cv:
            self._capture_path = str(path or "").strip()
            if steps is not None:
                self._capture_limit = max(0, int(steps))
            self._capture_count = 0

    def register_source(self, layer_key, module):
        if layer_key is None:
            return
        self._sources[layer_key] = weakref.ref(module)

    def sync_sources(self, sources):
        """Replace known streamable layer sources without clearing schedules."""
        with self._cv:
            self._sources = {
                key: weakref.ref(module)
                for key, module in sources
                if key is not None and module is not None
            }
            self._cv.notify_all()

    def set_schedule(self, layer_keys, confidence="exact", filter_to_sources=False):
        with self._cv:
            schedule = list(layer_keys)
            if filter_to_sources:
                source_keys = set(self._sources)
                schedule = [
                    entry for entry in schedule
                    if _schedule_layer_key(entry) in source_keys
                ]
            self._scheduled = schedule
            self._observed_step = []
            self.schedule_confidence = confidence
            self._cv.notify_all()

    def seed_schedule_from_sources(self):
        """Use registration order until a shape-specific trace is frozen.

        Deliberately does NOT clear ``_observed_step``: the manager re-seeds on
        every cold step, and the observed access order recorded last step is the
        only input ``step_begin``'s promotion has. Clearing it here would erase
        that record immediately before the promotion runs, so the pool could
        never escape the cold source-order schedule. ``_promote_observed_
        schedule_locked`` owns the buffer's lifecycle (it clears it each step)."""
        with self._cv:
            self._scheduled = list(self._sources.keys())
            self.schedule_version = -1
            self.schedule_shape_key = None
            self.schedule_confidence = "cold"
            self._cv.notify_all()

    def set_budget(self, budget_bytes: int):
        """Resize the pool budget without invalidating useful in-flight work."""
        with self._cv:
            self.budget_bytes = int(budget_bytes)
            self._trim_free_buffers_locked()
            self._cv.notify_all()

    def tune(self, *, budget_bytes=None, target_ready_bytes=None, lookahead=None,
             fill_group_size=None):
        """Adjust prefetch capacity/coverage without clearing the schedule."""
        with self._cv:
            if budget_bytes is not None:
                self.budget_bytes = int(budget_bytes)
            if target_ready_bytes is not None:
                self.target_ready_bytes = int(target_ready_bytes)
            if lookahead is not None:
                self.max_lookahead_positions = max(1, int(lookahead))
            if fill_group_size is not None:
                self.fill_group_size = max(1, int(fill_group_size))
            self._trim_free_buffers_locked()
            self._cv.notify_all()

    def step_begin(self, warmup_bytes=0, warmup_timeout_s=0.02):
        """Reset the consume/fill cursors and wake workers for a new step.

        GPU consumers from the previous step are complete at this boundary, but
        a prefetch worker may still be filling a future CPU slot outside the
        lock. Such a buffer is detached, not recycled; its worker owns it until
        the copy exits and observes the epoch change."""
        with self._cv:
            self._promote_observed_schedule_locked()
            self._reset_step_locked()
            deadline = time.perf_counter() + float(warmup_timeout_s)
            warmup_bytes = int(warmup_bytes)
            while (
                warmup_bytes > 0
                and self._ready_and_filling_bytes_locked() < warmup_bytes
                and time.perf_counter() < deadline
            ):
                self._cv.notify_all()
                self._cv.wait(timeout=0.002)

    def abort_step(self):
        """Invalidate current scheduling without recycling worker-owned buffers."""
        with self._cv:
            self._reset_step_locked()

    def _reset_step_locked(self):
        self._epoch += 1
        for slot in self._slots.values():
            if slot.state == CPU_FILLING:
                # The worker is copying outside the lock. Removing the slot is
                # enough to invalidate publication; it will recycle its own
                # leaves and decrement inflight bytes when the copy returns.
                continue
            if slot.leaves is not None:
                self._free_buffers.setdefault(slot.signature, []).append(slot.leaves)
            self._inflight_bytes -= slot.nbytes
        self._slots.clear()
        self._skipped_positions.clear()
        self._inflight_bytes = max(0, self._inflight_bytes)
        self._consume_pos = 0
        self._fill_pos = 0
        self._access_occurrences.clear()
        self._cv.notify_all()

    def _capture_observed_step_locked(self):
        if not self._capture_path or not self._observed_step:
            return
        if self._capture_limit and self._capture_count >= self._capture_limit:
            return
        record = {
            "time": time.time(),
            "device": str(self.device),
            "schedule": [_jsonable_entry(entry) for entry in self._scheduled],
            "observed": [_jsonable_entry(entry) for entry in self._observed_step],
            "schedule_confidence": self.schedule_confidence,
            "schedule_shape_key": _shape_key_label(self.schedule_shape_key),
            "schedule_version": self.schedule_version,
            "lookahead": self.max_lookahead_positions,
            "consume_pos": self._consume_pos,
            "resyncs": self.resyncs,
            "mismatches": self.mismatches,
            "duplicate_key_resync_blocked": self.duplicate_key_resync_blocked,
            "hits": self.hits,
            "soft_misses": self.soft_misses,
            "hard_misses": self.hard_misses,
            "skips": self.skips,
        }
        try:
            directory = os.path.dirname(self._capture_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self._capture_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._capture_count += 1
        except Exception:
            self._capture_path = ""

    def _promote_observed_schedule_locked(self):
        """Recover when the cold source-order schedule never matched reality."""
        self._capture_observed_step_locked()
        observed_len = len(self._observed_step)
        total = self.hits + self.soft_misses + self.hard_misses
        hard_miss_rate = self.hard_misses / max(1, total)
        stale_or_short = observed_len > len(self._scheduled) + max(1, self.max_lookahead_positions)
        cold_schedule = self.schedule_shape_key is None
        if observed_len and cold_schedule and stale_or_short and (total == 0 or hard_miss_rate > 0.25):
            self._scheduled = list(self._observed_step)
            self.schedule_version = -2
            self.schedule_shape_key = "observed"
            self.schedule_confidence = "observed"
        self._observed_step = []

    def _discard_positions_locked(self, start, end):
        for stale_pos in range(start, end):
            self._skipped_positions.add(stale_pos)
            slot = self._slots.get(stale_pos)
            if slot is not None and slot.state == CPU_READY:
                if slot.leaves is not None:
                    self._free_buffers.setdefault(slot.signature, []).append(slot.leaves)
                self._inflight_bytes -= slot.nbytes
                del self._slots[stale_pos]

    def _align_locked(self, access_key):
        pos = self._consume_pos
        if not self._scheduled:
            return pos, "mismatch"
        if pos < len(self._scheduled) and _entry_matches_access(
            self._scheduled[pos], access_key
        ):
            return pos, "aligned"

        layer_key = access_key[0]
        end = min(len(self._scheduled), pos + 1 + self.max_lookahead_positions)
        duplicate_layer_seen = False
        for j in range(pos + 1, end):
            entry = self._scheduled[j]
            if _entry_matches_access(entry, access_key):
                self._discard_positions_locked(pos, j)
                return j, "resynced"
            if _entry_same_layer(entry, layer_key):
                duplicate_layer_seen = True
        if duplicate_layer_seen:
            self.duplicate_key_resync_blocked += 1
        return pos, "mismatch"

    def _free_buffer_bytes_locked(self):
        total = 0
        for signature, buffers in self._free_buffers.items():
            total += _spec_bytes(signature) * len(buffers)
        return total

    def _pop_one_free_buffer_locked(self):
        for signature, buffers in list(self._free_buffers.items()):
            if not buffers:
                del self._free_buffers[signature]
                continue
            buffers.pop()
            release_pinned_bytes(_spec_bytes(signature), kind="bounce")
            if not buffers:
                del self._free_buffers[signature]
            return True
        return False

    def _trim_free_buffers_locked(self):
        while (
            self._inflight_bytes + self._free_buffer_bytes_locked()
            > self.budget_bytes
        ):
            if not self._pop_one_free_buffer_locked():
                break

    # -- training-thread API ----------------------------------------------

    def acquire(self, layer_key, weight_cpu, bias_cpu, operation="forward"):
        """Return (weight_src, bias_src, ticket). ticket is None on a hard miss
        (caller uses the pageable originals unchanged)."""
        with self._cv:
            occurrence = self._access_occurrences[layer_key]
            self._access_occurrences[layer_key] += 1
            access_key = _make_access_key(layer_key, operation, occurrence)
            pos, status = self._align_locked(access_key)
            self._consume_pos = pos + 1
            self._observed_step.append(access_key)
            if status == "resynced":
                self.resyncs += 1
            elif status == "mismatch":
                self.mismatches += 1
                self.hard_misses += 1
                self._cv.notify_all()
                return weight_cpu, bias_cpu, None
            slot = self._slots.get(pos)
            if slot is not None and not _slot_matches_request(
                slot, layer_key, weight_cpu, bias_cpu
            ):
                if slot.state == CPU_READY:
                    if slot.leaves is not None:
                        self._free_buffers.setdefault(slot.signature, []).append(
                            slot.leaves
                        )
                    self._inflight_bytes -= slot.nbytes
                    del self._slots[pos]
                self.hard_misses += 1
                self._cv.notify_all()
                return weight_cpu, bias_cpu, None
            if slot is not None and slot.state == CPU_READY:
                slot.state = IN_USE
                self.hits += 1
                self._cv.notify_all()
                return slot.weight, slot.bias, _Ticket(slot)
            if slot is not None and slot.state == CPU_FILLING:
                self.soft_misses += 1
                event = slot.ready_event
            else:
                # not staged: let the worker know, but don't block the train
                # thread on a synchronous pin -- use the pageable source.
                self.hard_misses += 1
                self._cv.notify_all()
                return weight_cpu, bias_cpu, None
        # wait for the worker to finish the in-flight copy (off-thread copy,
        # we only pay the wait, not the page faults).
        wait_t0 = time.perf_counter()
        event.wait()
        with self._cv:
            self.cpu_wait_s += time.perf_counter() - wait_t0
            slot = self._slots.get(pos)
            if slot is not None and not _slot_matches_request(
                slot, layer_key, weight_cpu, bias_cpu
            ):
                if slot.state == CPU_READY:
                    if slot.leaves is not None:
                        self._free_buffers.setdefault(slot.signature, []).append(
                            slot.leaves
                        )
                    self._inflight_bytes -= slot.nbytes
                    del self._slots[pos]
                self._cv.notify_all()
                return weight_cpu, bias_cpu, None
            if slot is not None and slot.state == CPU_READY:
                slot.state = IN_USE
                self._cv.notify_all()
                return slot.weight, slot.bias, _Ticket(slot)
            self._cv.notify_all()
        return weight_cpu, bias_cpu, None

    def consume_without_transfer(self, layer_key, operation="forward"):
        """Advance one trace position when the GPU ring already owns the weight.

        This keeps positional prefetch aligned without waiting for, or issuing,
        a CPU bounce copy that the consumer no longer needs.
        """
        with self._cv:
            occurrence = self._access_occurrences[layer_key]
            self._access_occurrences[layer_key] += 1
            access_key = _make_access_key(layer_key, operation, occurrence)
            pos, status = self._align_locked(access_key)
            self._consume_pos = pos + 1
            self._observed_step.append(access_key)
            self.skips += 1
            if status == "resynced":
                self.resyncs += 1
            elif status == "mismatch":
                self.mismatches += 1
            self._skipped_positions.add(pos)
            slot = self._slots.get(pos)
            if slot is not None and slot.state == CPU_READY:
                if slot.leaves is not None:
                    self._free_buffers.setdefault(slot.signature, []).append(
                        slot.leaves
                    )
                self._inflight_bytes -= slot.nbytes
                del self._slots[pos]
            self._cv.notify_all()

    def on_h2d_submitted(self, ticket, transfer_stream):
        """Record a reclaim barrier: the pinned buffer is reusable once the H2D
        enqueued on transfer_stream completes."""
        if ticket is None:
            return
        slot = ticket.slot
        event = torch.cuda.Event()
        event.record(transfer_stream)
        with self._cv:
            slot.reclaim_event = event
            self._cv.notify_all()

    # -- worker side -------------------------------------------------------

    def _reclaim_locked(self):
        """Return finished IN_USE slots (and stale slots behind the cursor) to FREE.

        An IN_USE slot is reclaimable only once its reclaim event has been
        recorded (by on_h2d_submitted, after the H2D was enqueued) AND fired.
        Reclaiming earlier would let a worker overwrite the pinned buffer while
        the async H2D is still reading it."""
        for pos, slot in list(self._slots.items()):
            # Never reclaim CPU_FILLING here: its worker is copying outside the
            # lock. A skipped filling slot is reclaimed by that worker below.
            stale = pos < self._consume_pos and slot.state == CPU_READY
            done = (
                slot.state == IN_USE
                and slot.reclaim_event is not None
                and slot.reclaim_event.query()
            )
            if stale or done:
                if slot.leaves is not None:
                    self._free_buffers.setdefault(slot.signature, []).append(slot.leaves)
                self._inflight_bytes -= slot.nbytes
                del self._slots[pos]

    def _next_fill_target_locked(self):
        """Pick the next schedulable position within byte and position limits."""
        target_ready_bytes = min(self.budget_bytes, self.target_ready_bytes)
        if self._ready_and_filling_bytes_locked() >= target_ready_bytes:
            return None
        end = min(
            len(self._scheduled),
            self._consume_pos + self.max_lookahead_positions,
        )
        for pos in range(max(self._consume_pos, self._fill_pos), end):
            if pos in self._slots:
                continue
            if pos in self._skipped_positions:
                continue
            return pos
        return None

    def _ready_and_filling_bytes_locked(self):
        total = 0
        for slot in self._slots.values():
            if slot.state in (CPU_FILLING, CPU_READY):
                total += slot.nbytes
        return total

    def _slot_bytes_by_state_locked(self):
        by_state = {
            CPU_FILLING: 0,
            CPU_READY: 0,
            IN_USE: 0,
            FREE: 0,
        }
        for slot in self._slots.values():
            by_state[slot.state] = by_state.get(slot.state, 0) + slot.nbytes
        return by_state

    def _take_buffers_locked(self, signature, nbytes):
        if self._inflight_bytes + nbytes > self.budget_bytes:
            return None
        pooled = self._free_buffers.get(signature)
        if pooled:
            leaves = pooled.pop()
        else:
            # New pinned allocation locks physical RAM. On a RAM-starved host
            # that deepens the very paging we are trying to avoid, so respect a
            # floor of free RAM and fall back to demand-load when it is tight.
            avail = _host_ram_available()
            if avail is not None and avail - nbytes < self.ram_floor_bytes:
                return None
            # Second, independent ceiling: the process-wide pinned-bytes ledger
            # (shared with MemoryManager's weight pins) approximates the WDDM
            # shared-GPU-memory budget, which plain "available RAM" cannot see.
            headroom = pinned_bytes_headroom(_cuda_device_index(self.device))
            if headroom is not None and nbytes > headroom:
                return None
            try:
                leaves, pinned_bytes = _alloc_pinned(signature)
            except Exception:
                # The proxy above is a heuristic, not a guarantee -- if the
                # driver still refuses (cudaErrorMemoryAllocation), demand-load
                # this position from the pageable source instead of taking the
                # worker thread down.
                return None
            if _pin_enabled() and pinned_bytes <= 0:
                return None
        self._inflight_bytes += nbytes
        return leaves

    def _claim_one_fill_locked(self):
        """Reserve the next fillable position as a CPU_FILLING slot.

        Returns ``(status, job)`` where status is one of:
          "job"  -> job = (pos, slot, weight, bias, signature); slot registered.
          "skip" -> position unfillable (no source/weight/too big/already
                    pinned -- the consumer bypasses the pool for it); advanced.
          "full" -> budget/buffers exhausted; caller should wait for a reclaim.
          "none" -> no schedulable position right now (target_ready met or end).
        """
        pos = self._next_fill_target_locked() if self._scheduled else None
        if pos is None:
            return "none", None
        layer_key = _schedule_layer_key(self._scheduled[pos])
        src_ref = self._sources.get(layer_key)
        module = src_ref() if src_ref is not None else None
        if module is None:
            # cannot stage an unknown source; skip it permanently
            self._fill_pos = pos + 1
            return "skip", None
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if weight is None:
            self._fill_pos = pos + 1
            return "skip", None
        if _is_pinned(weight) and (bias is None or _is_pinned(bias)):
            # Already page-locked (e.g. under the pinned-weight auto-budget):
            # the consumer transfers straight from it via consume_without_transfer,
            # so bouncing a redundant copy into a pool buffer here would only
            # burn a worker cycle and budget for nothing.
            self._fill_pos = pos + 1
            return "skip", None
        specs = _layer_specs(weight, bias)
        signature = tuple(specs)
        nbytes = _spec_bytes(specs)
        if nbytes > self.budget_bytes:
            # never fits the pool; always demand-load this one
            self._fill_pos = pos + 1
            return "skip", None
        leaves = self._take_buffers_locked(signature, nbytes)
        if leaves is None:
            # budget full; wait for a reclaim
            return "full", None
        slot = _Slot()
        slot.position = pos
        slot.layer_key = layer_key
        slot.state = CPU_FILLING
        slot.signature = signature
        slot.leaves = leaves
        slot.nbytes = nbytes
        slot.ready_event.clear()
        self._slots[pos] = slot
        self._fill_pos = pos + 1
        return "job", (pos, slot, weight, bias, signature)

    def _worker_loop(self):
        while True:
            jobs = []
            with self._cv:
                if self._stop:
                    return
                self._reclaim_locked()
                if not self._scheduled:
                    self._cv.wait(timeout=0.05)
                    continue
                # Claim up to fill_group_size positions under a single lock so a
                # whole block's worth of fills shares one lock/CV cycle instead
                # of one per Linear.
                full = False
                for _ in range(self.fill_group_size):
                    status, job = self._claim_one_fill_locked()
                    if status == "job":
                        jobs.append(job)
                        continue
                    if status == "skip":
                        continue
                    full = status == "full"
                    break
                if not jobs:
                    self._cv.wait(timeout=0.02 if full else 0.05)
                    continue
                epoch = self._epoch

            # Copy outside the lock so page faults / memcpy overlap the train
            # thread. PyTorch's copy_ releases the GIL for the actual transfer.
            results = []
            for (pos, slot, weight, bias, signature) in jobs:
                copy_t0 = time.perf_counter()
                pinned_weight = None
                pinned_bias = None
                try:
                    leaves_iter = iter(slot.leaves)
                    pinned_weight = _rebuild_into(weight, leaves_iter)
                    if bias is not None:
                        pinned_bias = next(leaves_iter)
                        pinned_bias.copy_(bias)
                    ok = True
                except Exception:
                    ok = False
                copy_dt = time.perf_counter() - copy_t0
                results.append(
                    (pos, slot, signature, pinned_weight, pinned_bias, ok, copy_dt)
                )

            with self._cv:
                published = 0
                for (pos, slot, signature, pinned_weight, pinned_bias, ok, copy_dt) in results:
                    if (
                        not ok
                        or self._epoch != epoch
                        or self._slots.get(pos) is not slot
                        or pos in self._skipped_positions
                    ):
                        # step rolled over or copy failed: discard this slot.
                        if self._slots.get(pos) is slot:
                            del self._slots[pos]
                        self._free_buffers.setdefault(signature, []).append(slot.leaves)
                        self._inflight_bytes -= slot.nbytes
                        slot.ready_event.set()  # release any waiter -> falls back
                        continue
                    slot.weight = pinned_weight
                    slot.bias = pinned_bias
                    slot.state = CPU_READY
                    self.copy_s += copy_dt
                    self.copy_bytes += slot.nbytes
                    self.fills += 1
                    published += 1
                    slot.ready_event.set()
                if published:
                    self.fill_batches += 1
                self._cv.notify_all()

    # -- diagnostics & teardown -------------------------------------------

    def _reset_stats_locked(self):
        self.hits = self.soft_misses = self.hard_misses = 0
        self.skips = 0
        self.resyncs = 0
        self.mismatches = 0
        self.duplicate_key_resync_blocked = 0
        self.cpu_wait_s = self.copy_s = 0.0
        self.copy_bytes = 0
        self.fills = self.fill_batches = 0

    def stats(self, reset: bool = False) -> dict:
        with self._lock:
            total = self.hits + self.soft_misses + self.hard_misses
            gib = 1024 ** 3
            copy_gbps = (
                (self.copy_bytes / gib) / self.copy_s if self.copy_s > 0 else 0.0
            )
            states = self._slot_bytes_by_state_locked()
            result = {
                "acquires": total,
                "hits": self.hits,
                "soft_misses": self.soft_misses,
                "hard_misses": self.hard_misses,
                "skips": self.skips,
                "resyncs": self.resyncs,
                "mismatches": self.mismatches,
                "duplicate_key_resync_blocked": self.duplicate_key_resync_blocked,
                "hit_rate": (self.hits / total) if total else 0.0,
                "cpu_wait_s": self.cpu_wait_s,
                "copy_s": self.copy_s,
                "copy_gbps": copy_gbps,
                "fills": self.fills,
                "fill_batches": self.fill_batches,
                "fill_group_size": self.fill_group_size,
                "fills_per_batch": (
                    self.fills / self.fill_batches if self.fill_batches else 0.0
                ),
                "budget_gib": self.budget_bytes / gib,
                "ready_gib": states.get(CPU_READY, 0) / gib,
                "filling_gib": states.get(CPU_FILLING, 0) / gib,
                "in_use_gib": states.get(IN_USE, 0) / gib,
                "free_buffer_gib": self._free_buffer_bytes_locked() / gib,
                "target_ready_gib": min(
                    self.budget_bytes, self.target_ready_bytes
                ) / gib,
                "inflight_gib": self._inflight_bytes / gib,
                "live_slots": len(self._slots),
                "consume_pos": self._consume_pos,
                "fill_pos": self._fill_pos,
                "schedule_len": len(self._scheduled),
                "observed_len": len(self._observed_step),
                "schedule_shape_key": _shape_key_label(self.schedule_shape_key),
                "schedule_confidence": self.schedule_confidence,
                "lookahead": self.max_lookahead_positions,
            }
            if reset:
                self._reset_stats_locked()
            return result

    def report(self, reset: bool = False) -> str:
        s = self.stats(reset=reset)
        return (
            f"[BouncePool] acquires={s['acquires']} "
            f"skipped_gpu_resident={s['skips']} "
            f"resync={s['resyncs']} mismatch={s['mismatches']} "
            f"dup_block={s['duplicate_key_resync_blocked']} "
            f"hit={s['hits']} soft_miss={s['soft_misses']} hard_miss={s['hard_misses']} "
            f"hit_rate={s['hit_rate']:.1%} cpu_wait={s['cpu_wait_s']:.1f}s "
            f"copy={s['copy_s']:.1f}s@{s['copy_gbps']:.2f}GB/s "
            f"fills={s['fills']} batches={s['fill_batches']} "
            f"group={s['fill_group_size']}({s['fills_per_batch']:.1f}/batch) "
            f"budget={s['budget_gib']:.2f}GiB ready={s['ready_gib']:.2f}GiB "
            f"filling={s['filling_gib']:.2f}GiB in_use={s['in_use_gib']:.2f}GiB "
            f"free_buffer={s['free_buffer_gib']:.2f}GiB "
            f"target_ready={s['target_ready_gib']:.2f}GiB "
            f"inflight={s['inflight_gib']:.2f}GiB slots={s['live_slots']} "
            f"pos={s['consume_pos']}/{s['fill_pos']} "
            f"schedule={s['schedule_len']} shape={s['schedule_shape_key']} "
            f"confidence={s['schedule_confidence']} "
            f"lookahead={s['lookahead']}"
        )

    def shrink(self, target_bytes: int = 0) -> int:
        """Drop idle free buffers for pin-manager reconciliation."""
        freed = 0
        with self._cv:
            for signature, buffers in list(self._free_buffers.items()):
                while buffers and (target_bytes <= 0 or freed < target_bytes):
                    buffers.pop()
                    nbytes = _spec_bytes(signature)
                    freed += nbytes
                    release_pinned_bytes(nbytes, kind="bounce")
                if not buffers:
                    del self._free_buffers[signature]
                if target_bytes > 0 and freed >= target_bytes:
                    break
            self._cv.notify_all()
        return freed

    def shutdown(self):
        pin_manager.unregister_evictable(self.shrink)

        with self._cv:
            self._stop = True
            self._reset_step_locked()
            self._cv.notify_all()
        for w in self._workers:
            # A worker may be inside a pageable->pinned copy. Do not release
            # its destination storage until it has left that copy section.
            w.join()
        with self._cv:
            release_pinned_bytes(self._inflight_bytes + self._free_buffer_bytes_locked(), kind="bounce")
            self._slots.clear()
            self._free_buffers.clear()
            self._inflight_bytes = 0


def create_pool(device, **kwargs) -> PinnedBouncePool:
    device = torch.device(device)
    existing = _DEVICE_PREFETCH.get(device)
    if existing is not None:
        return existing
    pool = PinnedBouncePool(device, **kwargs)
    _DEVICE_PREFETCH[device] = pool
    return pool


def get_pool(device) -> Optional[PinnedBouncePool]:
    return _DEVICE_PREFETCH.get(torch.device(device))


def all_pools() -> list:
    return list(_DEVICE_PREFETCH.values())


def destroy_all_pools():
    """Stop and remove every process-local pool at a job boundary."""
    for device in list(_DEVICE_PREFETCH):
        destroy_pool(device)


def destroy_pool(device):
    device = torch.device(device)
    pool = _DEVICE_PREFETCH.pop(device, None)
    if pool is not None:
        pool.shutdown()
