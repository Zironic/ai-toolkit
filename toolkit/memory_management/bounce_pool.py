"""
Slice 2B: bounded pinned CPU bounce pool.

Slice-1 measured ~134s/step of *pageable* submit stall: the training thread
blocks inside the H2D enqueue while Windows stages (and sometimes page-faults
from the pagefile) the pageable FP8 source weights. The ring depth is fine --
compute-wait was 0.8s -- so the fix is not a deeper GPU pipeline, it is moving
the pageable->pinned copy off the training thread.

This pool keeps the canonical weights pageable (as today) and, on background
worker threads, copies the upcoming layers' FP8 qdata / scales / bias into a
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

import os
import threading
import time
import weakref
from typing import Optional

import torch

# device -> PinnedBouncePool. The autograd staging path looks the active pool
# up here; absent an entry it behaves exactly as before.
_DEVICE_PREFETCH: dict = {}

# Slot states.
FREE = "FREE"
CPU_FILLING = "CPU_FILLING"
CPU_READY = "CPU_READY"
IN_USE = "IN_USE"

HIT = "hit"
SOFT_MISS = "soft_miss"   # worker was mid-copy; we waited for it (still off-thread copy)
HARD_MISS = "hard_miss"   # not staged at all; fall back to the pageable source


def _pin_enabled() -> bool:
    return torch.cuda.is_available()


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


def _spec_bytes(specs) -> int:
    total = 0
    for dtype, shape in specs:
        n = 1
        for s in shape:
            n *= s
        total += n * torch.empty((), dtype=dtype).element_size()
    return total


def _alloc_pinned(specs) -> list:
    pin = _pin_enabled()
    return [
        torch.empty(shape, dtype=dtype, pin_memory=pin)
        for (dtype, shape) in specs
    ]


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
        num_workers: int = 2,
        ram_floor_bytes: int = 2 * 1024 ** 3,
    ):
        self.device = torch.device(device)
        self.budget_bytes = int(budget_bytes)
        self.lookahead = int(lookahead)
        self.num_workers = max(1, int(num_workers))
        self.ram_floor_bytes = int(ram_floor_bytes)

        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._sources: dict = {}        # layer_key -> weakref(module)
        self._scheduled: list = []      # positional layer_key access order
        self.schedule_version = -1      # trace version this schedule came from
        self._slots: dict = {}          # position -> _Slot
        self._skipped_positions: set = set()
        self._free_buffers: dict = {}   # signature -> list[leaves]
        self._inflight_bytes = 0
        self._consume_pos = 0
        self._fill_pos = 0
        self._stop = False
        self._epoch = 0

        # stats
        self.hits = 0
        self.soft_misses = 0
        self.hard_misses = 0
        self.cpu_wait_s = 0.0
        self.copy_s = 0.0
        self.copy_bytes = 0
        self.skips = 0

        self._workers = [
            threading.Thread(target=self._worker_loop, daemon=True,
                             name=f"bounce-{self.device}-{i}")
            for i in range(self.num_workers)
        ]
        for w in self._workers:
            w.start()

    # -- registration & schedule ------------------------------------------

    def register_source(self, layer_key, module):
        if layer_key is None:
            return
        self._sources[layer_key] = weakref.ref(module)

    def set_schedule(self, layer_keys):
        with self._cv:
            self._scheduled = list(layer_keys)

    def step_begin(self):
        """Reset the consume/fill cursors and wake workers for a new step.

        GPU consumers from the previous step are complete at this boundary, but
        a prefetch worker may still be filling a future CPU slot outside the
        lock. Such a buffer is detached, not recycled; its worker owns it until
        the copy exits and observes the epoch change."""
        with self._cv:
            self._reset_step_locked()

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
        self._cv.notify_all()

    # -- training-thread API ----------------------------------------------

    def acquire(self, layer_key, weight_cpu, bias_cpu):
        """Return (weight_src, bias_src, ticket). ticket is None on a hard miss
        (caller uses the pageable originals unchanged)."""
        with self._cv:
            pos = self._consume_pos
            self._consume_pos += 1
            scheduled_match = (
                self._scheduled
                and pos < len(self._scheduled)
                and self._scheduled[pos] == layer_key
            )
            if not scheduled_match:
                self.hard_misses += 1
                self._cv.notify_all()
                return weight_cpu, bias_cpu, None
            slot = self._slots.get(pos)
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
            if slot is not None and slot.state == CPU_READY:
                slot.state = IN_USE
                self._cv.notify_all()
                return slot.weight, slot.bias, _Ticket(slot)
            self._cv.notify_all()
        return weight_cpu, bias_cpu, None

    def consume_without_transfer(self, layer_key):
        """Advance one trace position when the GPU ring already owns the weight.

        This keeps positional prefetch aligned without waiting for, or issuing,
        a CPU bounce copy that the consumer no longer needs.
        """
        with self._cv:
            pos = self._consume_pos
            self._consume_pos += 1
            self.skips += 1
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
        """Pick the next schedulable position within the lookahead window."""
        end = min(len(self._scheduled), self._consume_pos + self.lookahead)
        for pos in range(max(self._consume_pos, self._fill_pos), end):
            if pos in self._slots:
                continue
            return pos
        return None

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
            leaves = _alloc_pinned(signature)
        self._inflight_bytes += nbytes
        return leaves

    def _worker_loop(self):
        while True:
            with self._cv:
                if self._stop:
                    return
                self._reclaim_locked()
                pos = self._next_fill_target_locked() if self._scheduled else None
                if pos is None:
                    self._cv.wait(timeout=0.05)
                    continue
                layer_key = self._scheduled[pos]
                src_ref = self._sources.get(layer_key)
                module = src_ref() if src_ref is not None else None
                if module is None:
                    # cannot stage an unknown source; skip it permanently
                    self._fill_pos = pos + 1
                    continue
                weight = getattr(module, "weight", None)
                bias = getattr(module, "bias", None)
                if weight is None:
                    self._fill_pos = pos + 1
                    continue
                specs = _layer_specs(weight, bias)
                signature = tuple(specs)
                nbytes = _spec_bytes(specs)
                if nbytes > self.budget_bytes:
                    # never fits the pool; always demand-load this one
                    self._fill_pos = pos + 1
                    continue
                leaves = self._take_buffers_locked(signature, nbytes)
                if leaves is None:
                    # budget full; wait for a reclaim
                    self._cv.wait(timeout=0.02)
                    continue
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
                epoch = self._epoch

            # Copy outside the lock so page faults / memcpy overlap the train
            # thread. PyTorch's copy_ releases the GIL for the actual transfer.
            copy_t0 = time.perf_counter()
            try:
                leaves_iter = iter(slot.leaves)
                pinned_weight = _rebuild_into(weight, leaves_iter)
                pinned_bias = None
                if bias is not None:
                    pinned_bias = next(leaves_iter)
                    pinned_bias.copy_(bias)
                ok = True
            except Exception:
                ok = False
            copy_dt = time.perf_counter() - copy_t0

            with self._cv:
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
                slot.ready_event.set()
                self._cv.notify_all()

    # -- diagnostics & teardown -------------------------------------------

    def _reset_stats_locked(self):
        self.hits = self.soft_misses = self.hard_misses = 0
        self.skips = 0
        self.cpu_wait_s = self.copy_s = 0.0
        self.copy_bytes = 0

    def stats(self, reset: bool = False) -> dict:
        with self._lock:
            total = self.hits + self.soft_misses + self.hard_misses
            gib = 1024 ** 3
            copy_gbps = (
                (self.copy_bytes / gib) / self.copy_s if self.copy_s > 0 else 0.0
            )
            result = {
                "acquires": total,
                "hits": self.hits,
                "soft_misses": self.soft_misses,
                "hard_misses": self.hard_misses,
                "skips": self.skips,
                "hit_rate": (self.hits / total) if total else 0.0,
                "cpu_wait_s": self.cpu_wait_s,
                "copy_s": self.copy_s,
                "copy_gbps": copy_gbps,
                "inflight_gib": self._inflight_bytes / gib,
                "live_slots": len(self._slots),
            }
            if reset:
                self._reset_stats_locked()
            return result

    def report(self, reset: bool = False) -> str:
        s = self.stats(reset=reset)
        return (
            f"[BouncePool] acquires={s['acquires']} "
            f"skipped_gpu_resident={s['skips']} "
            f"hit={s['hits']} soft_miss={s['soft_misses']} hard_miss={s['hard_misses']} "
            f"hit_rate={s['hit_rate']:.1%} cpu_wait={s['cpu_wait_s']:.1f}s "
            f"copy={s['copy_s']:.1f}s@{s['copy_gbps']:.2f}GB/s "
            f"inflight={s['inflight_gib']:.2f}GiB slots={s['live_slots']}"
        )

    def shutdown(self):
        with self._cv:
            self._stop = True
            self._reset_step_locked()
            self._cv.notify_all()
        for w in self._workers:
            # A worker may be inside a pageable->pinned copy. Do not release
            # its destination storage until it has left that copy section.
            w.join()
        with self._cv:
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
