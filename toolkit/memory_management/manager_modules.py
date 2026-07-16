"""
This code was heavily inspired by the work of Lodestone-Rock, pretty much all credit goes
to them. The original code can be found here:
https://github.com/lodestone-rock/RamTorch/blob/main/ramtorch/modules/linear.py

I simply modified it to work with a memory management model and with AI Toolkit's models
"""

import atexit
import collections
import gc
import os
import time
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional, Tuple
from torch.overrides import has_torch_function_unary  # (ADD) torchao detection
from toolkit.quantization.fp8_linear import (
    FP8_STATS as _FP8_STATS,
    _fp8_grad_input_compute,
    _fp8_linear_compiled,
    _fp8_linear_training,
    dequantize_into as _dequantize_into,
    dequantize_to as _dequantize_to,
    fast_dequantize as _fast_fp8_dequant,
    fast_dequantize_into as _fast_fp8_dequant_into,
    fp8_linear_inference,
    fp8_sampling_qualifies,
    grad_input as _fp8_grad_input,
    grad_input_supported_weight as _fp8_grad_input_supported,
    reference_dequantize_to as _reference_dequantize_to,
    set_fp8_grad_input_enabled,
)
from . import pin_manager


from .bounce_pool import (
    get_pool as get_prefetch_pool,
    pinned_bytes_headroom,
    register_pinned_bytes,
    release_pinned_bytes,
    get_dxgi_meminfo,
    dxgi_spill_reserve_bytes,
    _cuda_device_index,
)

if TYPE_CHECKING:
    from .manager import MemoryManager

# --- Per-device global state registry ---
_DEVICE_STATE = {}

# How many layers deep to prefetch weights. The old ping-pong used 2 slots, which
# only lets one transfer overlap one compute (1-deep). A deeper ring lets Python
# enqueue several layers ahead so the H2D stream stays saturated instead of
# stalling on a per-layer sync. Override with AI_TOOLKIT_OFFLOAD_DEPTH.
PIPELINE_DEPTH = int(os.environ.get("AI_TOOLKIT_OFFLOAD_DEPTH", "4"))

def _fp8_stats_enabled():
    return _FP8_STATS["enabled"] or _FP8_STATS["training_enabled"]


# ===========================================================================
# Slice 1 instrumentation: localize where a streamed step spends its time.
#
# Three suspects produce very different fixes, so we separate them:
#   * submit_s   - CPU wall-clock blocked inside the H2D enqueue. For a pinned
#                  source `.to(non_blocking=True)` returns immediately; for a
#                  *pageable* source it blocks on the staging copy, and on a
#                  pagefile-backed page it blocks on disk. This is the column
#                  that exposes Windows paging stalls.
#   * staging_ms - GPU time for the H2D + on-GPU dequant (CUDA events on the
#                  transfer stream). This is PCIe + dequant bandwidth.
#   * wait_ms    - GPU time the compute stream sat blocked on fwd_slot_ready,
#                  i.e. compute that outran the transfer ring.
#
# Enable with AI_TOOLKIT_OFFLOAD_PROFILE=1. Off by default and the hot path
# pays only a single boolean check.
# ===========================================================================

_PROFILE_ENABLED = os.environ.get("AI_TOOLKIT_OFFLOAD_PROFILE", "0").lower() not in (
    "0",
    "false",
    "no",
    "off",
    "",
)


def set_offload_profile_enabled(enabled: bool, reset: bool = True) -> None:
    global _PROFILE_ENABLED
    _PROFILE_ENABLED = bool(enabled)
    if reset:
        _OFFLOAD_PROFILE.clear()
        _PENDING_EVENTS.clear()
        for state in _DEVICE_STATE.values():
            state["backward_reuse_hits"] = 0
            state["backward_reuse_misses"] = 0
            state["backward_reuse_bytes"] = 0


class _LayerProfile:
    __slots__ = (
        "label",
        "operation",
        "bytes",
        "pinned",
        "count",
        "submit_s",
        "staging_ms",
        "h2d_ms",
        "dequant_ms",
        "wait_ms",
    )

    def __init__(self, label, operation, nbytes, pinned):
        self.label = label
        self.operation = operation
        self.bytes = nbytes
        self.pinned = pinned
        self.count = 0
        self.submit_s = 0.0
        self.staging_ms = 0.0
        self.h2d_ms = 0.0
        self.dequant_ms = 0.0
        self.wait_ms = 0.0


_OFFLOAD_PROFILE: dict[int, _LayerProfile] = {}
# (profile, field, start_event, end_event) awaiting completion before we can
# read elapsed_time(). Drained opportunistically with query() so timing never
# perturbs the pipeline; force-synced only at summary time.
_PENDING_EVENTS: list = []
_EVENT_POOL: list = []


def _profile_bytes(t: Optional[torch.Tensor]) -> int:
    if t is None:
        return 0
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        return t.numel() * t.element_size()
    return sum(_profile_bytes(getattr(t, name, None)) for name in names)


def _ring_current_bytes(state) -> int:
    seen = set()
    total = 0
    for key in ("w_buffers", "b_buffers", "w_grad_buffers", "b_grad_buffers"):
        for tensor in state.get(key, ()) or ():
            if tensor is None or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            total += _profile_bytes(tensor)
    for entry in state.get("block_ring", ()) or ():
        try:
            total += int(entry.get("bytes", 0) or 0)
        except (TypeError, ValueError):
            pass
    return int(total)


def _record_ring_peak(state) -> None:
    current = _ring_current_bytes(state)
    state["ring_live_bytes"] = current
    state["ring_peak_bytes"] = max(int(state.get("ring_peak_bytes", 0)), current)


def _profile_is_pinned(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        try:
            # is_arena_backed: a register-pinned arena flat view reports
            # is_pinned()==False (torch only tracks its own caching-allocator
            # pins), so consult the arena storage set too or the streaming
            # bypass would needlessly bounce-stage already-pinned weights.
            return t.device.type == "cpu" and (
                t.is_pinned() or pin_manager.is_arena_backed(t)
            )
        except Exception:
            return False
    leaves = [getattr(t, name, None) for name in names]
    leaves = [leaf for leaf in leaves if leaf is not None]
    return bool(leaves) and all(_profile_is_pinned(leaf) for leaf in leaves)


def _profile_event() -> "torch.cuda.Event":
    if _EVENT_POOL:
        return _EVENT_POOL.pop()
    return torch.cuda.Event(enable_timing=True)


def _drain_pending_events(force: bool = False) -> None:
    if not _PENDING_EVENTS:
        return
    keep = []
    for record in _PENDING_EVENTS:
        prof, field, start_evt, end_evt = record
        if not force and not end_evt.query():
            keep.append(record)
            continue
        try:
            if force:
                end_evt.synchronize()
            elapsed_ms = start_evt.elapsed_time(end_evt)
        except (RuntimeError, ValueError):
            # The process can abort between recording the start/end events. Exit
            # diagnostics should skip that partial timing instead of obscuring
            # the real training error.
            if not force:
                keep.append(record)
            continue
        setattr(prof, field, getattr(prof, field) + elapsed_ms)
        _EVENT_POOL.append(start_evt)
        _EVENT_POOL.append(end_evt)
    _PENDING_EVENTS[:] = keep


def _begin_layer_profile(
    weight_cpu, bias_cpu, layer_key=None, operation="unknown"
) -> _LayerProfile:
    pinned = _profile_is_pinned(weight_cpu)
    key = (layer_key or id(weight_cpu), operation, pinned)
    prof = _OFFLOAD_PROFILE.get(key)
    if prof is None:
        try:
            shape = tuple(weight_cpu.shape)
        except Exception:
            shape = ()
        nbytes = _profile_bytes(weight_cpu) + _profile_bytes(bias_cpu)
        prof = _LayerProfile(
            layer_key or str(shape), operation, nbytes, pinned
        )
        _OFFLOAD_PROFILE[key] = prof
    return prof


def summarize_offload_profile(reset: bool = False) -> Optional[str]:
    """Aggregate the streamed-step timings into a one-shot human report."""
    if not _OFFLOAD_PROFILE:
        return None
    _drain_pending_events(force=True)
    gib = 1024 ** 3
    layers = list(_OFFLOAD_PROFILE.values())
    unique_layer_labels = {p.label for p in layers}
    fetches = sum(p.count for p in layers)
    submit = sum(p.submit_s for p in layers)
    h2d = sum(p.h2d_ms for p in layers) / 1000.0
    dequant = sum(p.dequant_ms for p in layers) / 1000.0
    unsplit_staging = sum(p.staging_ms for p in layers) / 1000.0
    staging = h2d + dequant + unsplit_staging
    wait = sum(p.wait_ms for p in layers) / 1000.0
    pageable = [p for p in layers if not p.pinned]
    pinned = [p for p in layers if p.pinned]
    submit_pageable = sum(p.submit_s for p in pageable)
    submit_pinned = sum(p.submit_s for p in pinned)
    operation_counts = {}
    operation_bytes = {}
    for p in layers:
        operation_counts[p.operation] = operation_counts.get(p.operation, 0) + p.count
        operation_bytes[p.operation] = (
            operation_bytes.get(p.operation, 0) + p.bytes * p.count
        )
    moved_bytes = sum(p.bytes * p.count for p in layers)
    reuse_hits = sum(s.get("backward_reuse_hits", 0) for s in _DEVICE_STATE.values())
    reuse_misses = sum(s.get("backward_reuse_misses", 0) for s in _DEVICE_STATE.values())
    reuse_bytes = sum(s.get("backward_reuse_bytes", 0) for s in _DEVICE_STATE.values())
    staging_gbps = (moved_bytes / gib) / staging if staging > 0 else 0.0
    top = sorted(layers, key=lambda p: p.submit_s, reverse=True)[:8]
    lines = [
        f"[OffloadProfile] fetches={fetches} unique_layers={len(unique_layer_labels)}",
        f"  submit (CPU wall / paging stall): {submit:.1f}s "
        f"[pageable={submit_pageable:.1f}s pinned={submit_pinned:.1f}s]",
        f"  staging (H2D+dequant GPU): {staging:.1f}s  "
        f"effective={staging_gbps:.2f} GB/s over {moved_bytes / gib:.2f} GiB moved",
        f"    split: H2D={h2d:.1f}s dequant/materialize={dequant:.1f}s "
        f"unsplit={unsplit_staging:.1f}s",
        f"  compute wait on ready: {wait:.1f}s",
        f"  source layers: pageable={len(pageable)} "
        f"({sum(p.bytes for p in pageable) / gib:.2f} GiB) "
        f"pinned={len(pinned)} ({sum(p.bytes for p in pinned) / gib:.2f} GiB)",
        "  accesses by operation: " + ", ".join(
            f"{op}={operation_counts[op]} "
            f"({operation_bytes[op] / gib:.2f} GiB)"
            for op in sorted(operation_counts)
        ),
        f"  backward GPU-ring reuse: hits={reuse_hits} misses={reuse_misses} "
        f"hit_rate={100.0 * reuse_hits / max(1, reuse_hits + reuse_misses):.1f}% "
        f"avoided={reuse_bytes / gib:.2f} GiB",
        f"  native FP8 linear: enabled={_FP8_STATS['training_enabled']} "
        f"calls={_FP8_STATS['kernel_calls']} "
        f"fallbacks={_FP8_STATS['fallback_calls']}",
        "  top submit stalls (layer / submit_s / pinned):",
    ]
    for p in top:
        lines.append(
            f"    {p.label:<36} {p.operation:<8} {p.submit_s:7.2f}s  pinned={p.pinned} "
            f"x{p.count} {p.bytes / 1024 ** 2:.1f} MiB"
        )
    report = "\n".join(lines)
    if reset:
        _OFFLOAD_PROFILE.clear()
        _PENDING_EVENTS.clear()
        for state in _DEVICE_STATE.values():
            state["backward_reuse_hits"] = 0
            state["backward_reuse_misses"] = 0
            state["backward_reuse_bytes"] = 0
        _FP8_STATS["kernel_calls"] = 0
        _FP8_STATS["fallback_calls"] = 0
    return report


@atexit.register
def _print_offload_profile_atexit() -> None:
    if not _PROFILE_ENABLED:
        return
    try:
        report = summarize_offload_profile()
    except (RuntimeError, ValueError):
        # CUDA may be in an error state after a fatal allocation failure. Exit
        # diagnostics must never obscure the actual training exception.
        return
    if report:
        print(report)


# ===========================================================================
# Slice 2A: stable execution trace.
#
# A streamed step is not a simple forward/reverse walk. One Krea step contains
# the normal forward, DOP forward(s), gradient-checkpoint recomputation, DOP
# backward, normal backward, and reused layers -- the slice-1 numbers showed
# the 6.12 GiB offload set fetched ~7x. The scheduler (slice 2C) can only
# prefetch a layer it can name *ahead of time*, so we first record the exact
# ordered sequence of weight accesses for one step and replay it thereafter.
#
# This slice is pure observation: it records, freezes after the first step,
# then on later steps compares actual accesses against the frozen trace and
# counts divergences. It does not yet drive any prefetch. Demand loading still
# executes every fetch, so a wrong trace cannot produce wrong weights here.
#
# Enable with AI_TOOLKIT_OFFLOAD_TRACE=1. Layer identity is the module path
# (see MemoryManager.attach), NOT id(weight): sampling detach/restore replaces
# Parameter objects, so an id-based key would not survive a sample.
# ===========================================================================

_TRACE_ENABLED = os.environ.get("AI_TOOLKIT_OFFLOAD_TRACE", "0").lower() not in (
    "0",
    "false",
    "no",
    "off",
    "",
)

# How many consecutive diverging steps before we discard the frozen trace and
# re-record. A stable schedule should never hit this; it exists so a genuine
# change in access pattern (e.g. a config switch mid-run) self-heals.
_TRACE_REDISCOVER_AFTER = int(os.environ.get("AI_TOOLKIT_OFFLOAD_TRACE_REDISCOVER", "3"))

# schedule_by_shape_key holds one full per-step access list (~1700+ entries for
# a Krea2-sized model) per distinct (resolution, config-flags) shape key ever
# seen. Bucketed aspect-ratio datasets can produce dozens to hundreds of
# distinct shape keys over a run with no natural end -- without a cap this
# dict only ever grows for the life of the process. Evict least-recently-used
# once the cap is hit.
_TRACE_MAX_SHAPE_KEYS = int(os.environ.get("AI_TOOLKIT_OFFLOAD_TRACE_MAX_SHAPES", "64"))

_Access = collections.namedtuple(
    "_Access", ("layer_key", "operation", "fp8_bytes", "materialized_bytes")
)


class _OffloadTrace:
    """Records, freezes, and replay-validates one step's weight-access order."""

    def __init__(self):
        self.enabled = _TRACE_ENABLED
        self.mode = "idle"            # idle | recording | replaying
        self.recording: list = []
        self.frozen: Optional[list] = None
        self.schedule_by_shape_key: "collections.OrderedDict" = collections.OrderedDict()
        self.compatible_fallback_blocked_shape_keys = set()
        self.current_shape_key = None
        self.cursor = 0
        self.step_index = 0
        self.total_divergences = 0
        self.consecutive_divergences = 0
        self.diverged_this_step = False
        self.first_divergence: Optional[str] = None
        # Bumped every time a trace is frozen, so consumers (the bounce pool)
        # know to refresh a schedule learned from an earlier, different shape
        # (e.g. the DOP warmup step before the steady-state shape settles).
        self.version = 0

    def step_begin(self, shape_key=None):
        if not self.enabled:
            return
        self.current_shape_key = shape_key
        self.frozen = self.schedule_by_shape_key.get(shape_key)
        if self.frozen is not None:
            self.schedule_by_shape_key.move_to_end(shape_key)
        self.cursor = 0
        self.diverged_this_step = False
        self.first_divergence = None
        if self.frozen is None:
            self.mode = "recording"
            self.recording = []
        else:
            self.mode = "replaying"

    def record(self, layer_key, operation, fp8_bytes, materialized_bytes):
        if not self.enabled or self.mode == "idle":
            return
        if self.mode == "recording":
            self.recording.append(
                _Access(layer_key, operation, fp8_bytes, materialized_bytes)
            )
            return
        # replaying: compare against the frozen trace without altering execution.
        frozen = self.frozen
        if self.cursor < len(frozen):
            expected = frozen[self.cursor]
            if (
                expected.layer_key != layer_key
                or expected.operation != operation
            ):
                if not self.diverged_this_step:
                    self.first_divergence = (
                        f"pos {self.cursor}: expected "
                        f"{expected.operation}:{expected.layer_key}, got "
                        f"{operation}:{layer_key}"
                    )
                self.diverged_this_step = True
        else:
            if not self.diverged_this_step:
                self.first_divergence = (
                    f"pos {self.cursor}: trace exhausted "
                    f"(len {len(frozen)}), got {operation}:{layer_key}"
                )
            self.diverged_this_step = True
        self.cursor += 1

    def step_end(self):
        if not self.enabled:
            return
        if self.mode == "recording":
            self.frozen = self.recording
            self.schedule_by_shape_key[self.current_shape_key] = self.frozen
            self.schedule_by_shape_key.move_to_end(self.current_shape_key)
            while len(self.schedule_by_shape_key) > _TRACE_MAX_SHAPE_KEYS:
                self.schedule_by_shape_key.popitem(last=False)
            self.compatible_fallback_blocked_shape_keys.discard(self.current_shape_key)
            self.recording = []
            self.version += 1
            self._report_frozen()
        elif self.mode == "replaying":
            length_mismatch = self.cursor != len(self.frozen)
            if self.diverged_this_step or length_mismatch:
                self.total_divergences += 1
                self.consecutive_divergences += 1
                detail = self.first_divergence or (
                    f"length {self.cursor} != {len(self.frozen)}"
                )
                print(
                    f"[OffloadTrace] step {self.step_index} diverged "
                    f"({self.consecutive_divergences} in a row): {detail}"
                )
                if self.consecutive_divergences >= _TRACE_REDISCOVER_AFTER:
                    print(
                        "[OffloadTrace] re-recording trace after "
                        f"{self.consecutive_divergences} divergences"
                    )
                    self.schedule_by_shape_key.pop(self.current_shape_key, None)
                    self.compatible_fallback_blocked_shape_keys.add(self.current_shape_key)
                    self.frozen = None
                    self.version += 1
                    self.consecutive_divergences = 0
            else:
                self.consecutive_divergences = 0
        self.step_index += 1
        self.mode = "idle"

    def step_abort(self):
        """Discard the in-flight step (e.g. OOM) without freezing/validating."""
        if not self.enabled:
            return
        if self.mode == "recording":
            self.recording = []
        self.mode = "idle"

    def _report_frozen(self):
        frozen = self.frozen
        gib = 1024 ** 3
        unique = {a.layer_key for a in frozen}
        by_op = collections.Counter(a.operation for a in frozen)
        fp8_bytes = sum(a.fp8_bytes for a in frozen)
        mat_bytes = sum(a.materialized_bytes for a in frozen)
        print(
            f"[OffloadTrace] froze step trace: accesses={len(frozen)} "
            f"unique_layers={len(unique)} "
            f"ops={{{', '.join(f'{k}={v}' for k, v in sorted(by_op.items()))}}} "
            f"fp8={fp8_bytes / gib:.2f} GiB materialized={mat_bytes / gib:.2f} GiB"
        )

    def report(self) -> Optional[str]:
        if not self.schedule_by_shape_key:
            return None
        frozen = self.frozen or next(iter(self.schedule_by_shape_key.values()))
        unique = {a.layer_key for a in frozen}
        return (
            f"[OffloadTrace] frozen accesses={len(frozen)} "
            f"unique_layers={len(unique)} shapes={len(self.schedule_by_shape_key)} "
            f"steps={self.step_index} divergences={self.total_divergences}"
        )


_OFFLOAD_TRACE = _OffloadTrace()


def offload_step_begin(shape_key=None) -> None:
    _OFFLOAD_TRACE.step_begin(shape_key=shape_key)
    for state in _DEVICE_STATE.values():
        if isinstance(state, dict):
            current = _ring_current_bytes(state)
            state["ring_live_bytes"] = current
            state["ring_peak_bytes"] = current


def offload_step_end() -> None:
    _OFFLOAD_TRACE.step_end()


def offload_step_abort() -> None:
    _OFFLOAD_TRACE.step_abort()


def record_weight_access(
    layer_key, operation, fp8_bytes=0, materialized_bytes=0
) -> None:
    if layer_key is None:
        return
    _OFFLOAD_TRACE.record(layer_key, operation, fp8_bytes, materialized_bytes)



def _trace_execution_order_key(shape_key):
    """Part of an offload shape key that controls access order, not memory size."""
    if (
        isinstance(shape_key, tuple)
        and len(shape_key) == 2
        and isinstance(shape_key[1], tuple)
    ):
        return shape_key[1]
    return shape_key


def _compatible_frozen_trace(shape_key):
    if shape_key in _OFFLOAD_TRACE.compatible_fallback_blocked_shape_keys:
        return None
    target_order_key = _trace_execution_order_key(shape_key)
    for candidate_key, frozen in reversed(_OFFLOAD_TRACE.schedule_by_shape_key.items()):
        if _trace_execution_order_key(candidate_key) == target_order_key:
            return frozen
    return None

def offload_trace_report() -> Optional[str]:
    return _OFFLOAD_TRACE.report()


def offload_trace_schedule(shape_key=None) -> Optional[list]:
    """Positional semantic access order from the frozen trace, or None.

    When a new resolution has not frozen its own trace yet, replay the latest
    full trace from another shape instead of the cold unique-layer list. Schedule
    entries deliberately include operation and per-layer occurrence so local
    resync cannot jump from a forward access to a later backward/recompute access
    of the same layer key.
    """
    frozen = _OFFLOAD_TRACE.schedule_by_shape_key.get(shape_key)
    if frozen is None and shape_key == _OFFLOAD_TRACE.current_shape_key:
        frozen = _OFFLOAD_TRACE.frozen
    if frozen is None:
        frozen = _compatible_frozen_trace(shape_key)
    if frozen is None:
        return None
    occurrences = collections.Counter()
    schedule = []
    for access in frozen:
        occurrence = occurrences[access.layer_key]
        occurrences[access.layer_key] += 1
        schedule.append((access.layer_key, access.operation, occurrence))
    return schedule


def offload_trace_schedule_confidence(shape_key=None) -> str:
    if _OFFLOAD_TRACE.schedule_by_shape_key.get(shape_key) is not None:
        return "exact"
    if shape_key == _OFFLOAD_TRACE.current_shape_key and _OFFLOAD_TRACE.frozen is not None:
        return "exact"
    if _compatible_frozen_trace(shape_key) is not None:
        return "compatible"
    return "cold"

def offload_trace_version() -> int:
    """Monotonic id of the current frozen trace (changes on re-record)."""
    return _OFFLOAD_TRACE.version


def mark_transfer_plan_dirty(reason: str = "") -> None:
    """Mark transfer planning dirty without dropping execution traces.

    Layout changes such as layer promotion/demotion alter which accesses require
    H2D transfer, but they normally do not alter the execution order. Keep frozen
    schedules so the bounce pool can continue using them as hints; bump the
    version so consumers refresh derived transfer state at the next boundary.
    """
    trace = _OFFLOAD_TRACE
    if not trace.enabled:
        return
    trace.version += 1
    trace.cursor = 0
    trace.diverged_this_step = False
    trace.first_divergence = None


def invalidate_offload_trace_for_shape(shape_key=None) -> None:
    """Drop only one exact trace and block compatible fallback for that shape.

    Used when a best-effort compatible schedule proves bad for the current
    resolution. Other shapes with the same execution policy keep their traces.
    """
    trace = _OFFLOAD_TRACE
    if not trace.enabled:
        return
    key = trace.current_shape_key if shape_key is None else shape_key
    trace.schedule_by_shape_key.pop(key, None)
    trace.compatible_fallback_blocked_shape_keys.add(key)
    if key == trace.current_shape_key:
        trace.frozen = None
        trace.recording = []
        trace.cursor = 0
        trace.diverged_this_step = False
        trace.first_divergence = None
        trace.consecutive_divergences = 0
        trace.mode = "recording"
    trace.version += 1

def invalidate_execution_trace(reason: str = "") -> None:
    """Drop execution traces after a real access-order policy change."""
    trace = _OFFLOAD_TRACE
    if not trace.enabled:
        return
    trace.schedule_by_shape_key.clear()
    trace.compatible_fallback_blocked_shape_keys.clear()
    trace.version += 1
    trace.frozen = None
    trace.recording = []
    trace.cursor = 0
    trace.diverged_this_step = False
    trace.first_divergence = None
    trace.consecutive_divergences = 0
    trace.mode = "recording"

def set_offload_trace_enabled(enabled: bool) -> None:
    # _TRACE_ENABLED gates the per-access recording on the autograd hot path;
    # _OFFLOAD_TRACE.enabled gates the step boundaries. They must move together,
    # or (as happened) the step freezes an empty trace because recording was
    # skipped while step_begin/end still ran.
    global _TRACE_ENABLED
    _TRACE_ENABLED = bool(enabled)
    _OFFLOAD_TRACE.enabled = bool(enabled)


def _get_device_state(device: torch.device):
    """Get or initialize per-device state."""
    if isinstance(device, str):
        device = torch.device(device)

    # CPU path needs no CUDA state
    if device.type != "cuda":
        if device not in _DEVICE_STATE:
            _DEVICE_STATE[device] = {}
        return _DEVICE_STATE[device]

    if device not in _DEVICE_STATE:
        d = max(2, PIPELINE_DEPTH)
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                "depth": d,
                # streams
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                # forward weight ring: slot_ready = H2D done, slot_free = compute
                # that consumed the slot done (so it can be overwritten).
                "w_buffers": [None] * d,
                "b_buffers": [None] * d,
                "w_layer_keys": [None] * d,
                # Slice 3: persistent per-slot bf16 dequant destination, grown to
                # the largest layer that lands in the slot and then reused. This
                # is what stops the TorchAO dequant from allocating a fresh bf16
                # output on every fetch (the churn that fragments VRAM into the
                # WDDM shared-memory spill). Bounded at depth * max_layer_bf16.
                "w_dest": [None] * d,
                "fwd_slot_ready": [torch.cuda.Event() for _ in range(d)],
                "fwd_slot_free": [torch.cuda.Event() for _ in range(d)],
                "forward_clk": 0,
                "backward_reuse_hits": 0,
                "backward_reuse_misses": 0,
                "backward_reuse_bytes": 0,
                # backward grad-staging ring (device-side grads -> CPU).
                "w_grad_buffers": [None] * d,
                "b_grad_buffers": [None] * d,
                "grad_compute_done": [torch.cuda.Event() for _ in range(d)],
                "grad_xfer_done": [torch.cuda.Event() for _ in range(d)],
                # block-coalesced forward staging (Slice 2, opt-in; inert until
                # set_block_stream_enabled). block_resident: layer_key ->
                # (w_gpu, b_gpu, ready_event); block_ring: FIFO of staged blocks.
                "block_stream_enabled": False,
                "block_depth": 2,
                "block_resident": {},
                "block_ring": [],
                "block_h2d_count": 0,
                "block_layer_count": 0,
                "ring_live_bytes": 0,
                "ring_peak_bytes": 0,
            }
    return _DEVICE_STATE[device]


# ---- ring-buffer staging helpers -----------------------------------------
#
# Each transfer waits only on the event for the *specific slot* it is about to
# overwrite (the compute that used that slot D layers ago), not on a single
# global "compute started" event. With D slots that prior compute is long done,
# so the transfer stream never actually stalls and stays D layers ahead of
# compute. This is the deeper-pipeline + relaxed-dependency change in one.


def _stage_forward_weight(
    state, device, materialize, weight_cpu, bias_cpu,
    layer_key=None, operation="forward"
):
    """H2D the next forward weight (+bias) into its ring slot; return (idx, w, b).
    Caller runs compute, then calls _release_forward_slot(state, idx)."""
    d = state["depth"]
    idx = state["forward_clk"]
    state["forward_clk"] = (idx + 1) % d
    ts = state["transfer_stream"]
    # Slice 2B: if a bounce pool has the source pre-pinned, transfer from the
    # pinned copy (async, no pageable stall) instead of the pageable weight.
    pool = get_prefetch_pool(device)
    ticket = None
    src_w, src_b = weight_cpu, bias_cpu
    if pool is not None:
        # Pinned-source bypass: a weight that is already page-locked needs no
        # bounce copy — the H2D below runs async straight from it. Skip the pool
        # (no redundant pinned->pinned copy, worker stays idle) but advance its
        # cursor so prefetch stays aligned for the pageable layers that do need it.
        if _profile_is_pinned(weight_cpu) and (
            bias_cpu is None or _profile_is_pinned(bias_cpu)
        ):
            pool.consume_without_transfer(layer_key, operation=operation)
        else:
            src_w, src_b, ticket = pool.acquire(
                layer_key, weight_cpu, bias_cpu, operation=operation
            )
    prof = (
        _begin_layer_profile(src_w, src_b, layer_key, operation)
        if _PROFILE_ENABLED else None
    )

    def _dest_fn(shape, dtype):
        # Slice 3: hand the materializer this slot's reusable bf16 buffer so the
        # dequant writes in place instead of allocating a new output each fetch.
        if device.type != "cuda":
            return None
        return _slot_bf16_dest(state, idx, shape, dtype, device)

    profile_marks = {}

    def _profile_midpoint():
        if prof is None or "h2d_end" in profile_marks:
            return
        # Separate events avoid sharing one pooled event between two records.
        h2d_end = _profile_event()
        dequant_start = _profile_event()
        h2d_end.record(ts)
        dequant_start.record(ts)
        profile_marks["h2d_end"] = h2d_end
        profile_marks["dequant_start"] = dequant_start

    with torch.cuda.stream(ts):
        ts.wait_event(state["fwd_slot_free"][idx])
        if prof is not None:
            stage_start = _profile_event()
            stage_start.record(ts)
            submit_t0 = time.perf_counter()
        state["w_buffers"][idx] = materialize(
            src_w, device, _dest_fn, _profile_midpoint
        )
        state["w_layer_keys"][idx] = layer_key
        state["b_buffers"][idx] = (
            src_b.to(device, non_blocking=True) if src_b is not None else None
        )
        _record_ring_peak(state)
        if prof is not None:
            # CPU wall inside the enqueue: ~0 for pinned, = staging/pagefile
            # copy time for a pageable source.
            prof.submit_s += time.perf_counter() - submit_t0
            stage_end = _profile_event()
            stage_end.record(ts)
            if "h2d_end" in profile_marks:
                _PENDING_EVENTS.append(
                    (prof, "h2d_ms", stage_start, profile_marks["h2d_end"])
                )
                _PENDING_EVENTS.append(
                    (
                        prof,
                        "dequant_ms",
                        profile_marks["dequant_start"],
                        stage_end,
                    )
                )
            else:
                _PENDING_EVENTS.append(
                    (prof, "staging_ms", stage_start, stage_end)
                )
        state["fwd_slot_ready"][idx].record()
        if ticket is not None:
            # The pinned source is reusable once this H2D completes.
            pool.on_h2d_submitted(ticket, ts)
    if prof is not None:
        cs = torch.cuda.current_stream()
        wait_start = _profile_event()
        wait_start.record(cs)
        cs.wait_event(state["fwd_slot_ready"][idx])
        wait_end = _profile_event()
        wait_end.record(cs)
        # wait_end can only complete after the slot-ready wait clears, so the
        # elapsed time is exactly how long compute sat blocked on the transfer.
        _PENDING_EVENTS.append((prof, "wait_ms", wait_start, wait_end))
        prof.count += 1
        _drain_pending_events()
    else:
        torch.cuda.current_stream().wait_event(state["fwd_slot_ready"][idx])
    return idx, state["w_buffers"][idx], state["b_buffers"][idx]


def _release_forward_slot(state, idx):
    # Slot is reusable once the compute stream finishes the op that read it.
    state["fwd_slot_free"][idx].record()


def _stage_backward_weight(
    state, device, materialize, weight_cpu, layer_key=None, expected_dtype=None
):
    """Re-fetch for grad-input through the shared bidirectional weight ring."""
    # Gradient-checkpoint recompute is immediately followed by the block's
    # backward. Its most recent weights are therefore still materialized in
    # the bounded GPU ring. Reuse them before scheduling another H2D+dequant.
    # Search newest-to-oldest; repeated module calls can leave several copies.
    if layer_key is not None:
        d = state["depth"]
        clk = state["forward_clk"]
        for distance in range(1, d + 1):
            reuse_idx = (clk - distance) % d
            if state["w_layer_keys"][reuse_idx] != layer_key:
                continue
            candidate = state["w_buffers"][reuse_idx]
            if candidate is None:
                continue
            if expected_dtype is not None and getattr(candidate, "dtype", None) != expected_dtype:
                continue
            torch.cuda.current_stream().wait_event(
                state["fwd_slot_ready"][reuse_idx]
            )
            state["backward_reuse_hits"] += 1
            state["backward_reuse_bytes"] += _profile_bytes(weight_cpu)
            pool = get_prefetch_pool(device)
            if pool is not None:
                pool.consume_without_transfer(layer_key, operation="backward")
            return reuse_idx, candidate
    state["backward_reuse_misses"] += 1
    idx, weight, _ = _stage_forward_weight(
        state,
        device,
        lambda source, _device, _dest_fn=None, _profile_midpoint=None: materialize(
            source, _dest_fn, _profile_midpoint
        ),
        weight_cpu,
        None,
        layer_key,
        "backward",
    )
    return idx, weight


def _release_backward_weight_slot(state, idx):
    _release_forward_slot(state, idx)


# ---- block-coalesced forward staging (Slice 2, opt-in) -------------------
#
# True block streaming. Instead of one H2D + ring-slot/event pair per Linear, a
# block forward-pre-hook stages all of a block's streamed weights in ONE
# transfer-stream burst under a single ready event, into a small ring of
# whole-block buffers (default 2 blocks: one executing, one prefetching). Each
# Linear's forward then consumes its already-resident weight via
# ``consume_block_resident`` instead of issuing its own H2D.
#
# Safe by construction: this is forward-only. If nothing staged a layer,
# ``consume_block_resident`` returns None and the Linear takes the unchanged
# per-Linear path. Backward staging is never touched, so gradients are computed
# from the same weight values either way. Quantized weights are dequantized per
# Linear at stage time (no wrapper repacking); fp8-native forward and conv keep
# the per-Linear path.


def set_block_stream_enabled(device, enabled, depth=2):
    state = _get_device_state(device)
    state["block_stream_enabled"] = bool(enabled)
    state["block_depth"] = max(1, int(depth))
    if not enabled:
        reset_block_stream(device)


def reset_block_stream(device):
    state = _DEVICE_STATE.get(torch.device(device))
    if not state:
        return
    state["block_resident"] = {}
    state["block_ring"] = []
    state["block_h2d_count"] = 0
    state["block_layer_count"] = 0
    state["ring_live_bytes"] = _ring_current_bytes(state)


def block_stream_stats(device):
    """(h2d_count, layer_count): block H2D copies vs layers they covered."""
    state = _DEVICE_STATE.get(torch.device(device))
    if not state:
        return 0, 0
    return state.get("block_h2d_count", 0), state.get("block_layer_count", 0)


def _flatten_leaves(t):
    """Depth-first list of the physical leaf tensors of a (maybe wrapper) tensor.
    A plain tensor is its own single leaf; a tensor subclass yields its physical
    storage in ``__tensor_flatten__`` order."""
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
    """Rebuild a tensor with ``src``'s type/metadata, backed by the next device
    leaves from ``leaves_iter`` (already filled by the block H2D) — no copy."""
    try:
        names, ctx = src.__tensor_flatten__()
    except Exception:
        return next(leaves_iter)
    moved = {}
    for name in names:
        inner = getattr(src, name, None)
        moved[name] = None if inner is None else _rebuild_from_leaves(inner, leaves_iter)
    return type(src).__tensor_unflatten__(moved, ctx, src.size(), src.stride())


_BLOCK_LEAF_ALIGN = 256  # generous alignment so every leaf's uint8 slice .view()s cleanly


def stage_block_forward(device, block_key, linears, compute_dtype=None):
    """Stage a whole block's streamed weights with a SINGLE H2D copy.

    The block's ordered weight/bias storage leaves are packed into one
    contiguous pinned host buffer, copied
    to GPU in one ``cudaMemcpyAsync``, then sliced back into per-Linear tensors
    that view the single device buffer. This is the point of block streaming:
    one transfer (one CPU submit) per block instead of one per Linear.

    ``linears``: ``(layer_key, weight_cpu, bias_cpu)`` list. Stored in
    ``block_resident`` as ``(w_gpu, b_gpu, ready_event)`` — still quantized for
    quantized weights; the Linear forward dequantizes per layer as usual.
    """
    device = torch.device(device)
    if device.type != "cuda":
        return
    state = _get_device_state(device)
    if not state.get("block_stream_enabled"):
        return
    resident = state["block_resident"]
    ring = state["block_ring"]
    depth = state["block_depth"]
    ts = state["transfer_stream"]

    # 1) Flatten every weight/bias to physical leaves and lay them out in one
    #    aligned byte buffer.
    items = []        # (layer_key, weight_cpu, bias_cpu, n_w_leaves, n_b_leaves)
    src_leaves = []
    for (layer_key, weight_cpu, bias_cpu) in linears:
        w_leaves = _flatten_leaves(weight_cpu)
        b_leaves = _flatten_leaves(bias_cpu) if bias_cpu is not None else []
        items.append((layer_key, weight_cpu, bias_cpu, len(w_leaves), len(b_leaves)))
        src_leaves.extend(w_leaves)
        src_leaves.extend(b_leaves)
    if not src_leaves:
        return
    align = _BLOCK_LEAF_ALIGN
    offsets = []
    total = 0
    for leaf in src_leaves:
        total = (total + align - 1) // align * align
        offsets.append(total)
        total += leaf.numel() * leaf.element_size()

    # 2) Pack the leaves into one pinned host buffer (host memcpy, no CUDA submit).
    host_handle = pin_manager.pin_alloc(total, "block_stream", required=True)
    host = host_handle.tensor
    for leaf, off in zip(src_leaves, offsets):
        nb = leaf.numel() * leaf.element_size()
        host[off:off + nb].view(leaf.dtype).reshape(leaf.shape).copy_(leaf)

    ready_event = torch.cuda.Event()
    free_event = torch.cuda.Event()
    staged_keys = []
    with torch.cuda.stream(ts):
        while len(ring) >= depth:
            old = ring.pop(0)
            if old["free"] is not None:
                ts.wait_event(old["free"])
            for k in old["layer_keys"]:
                resident.pop(k, None)
        # 3) ONE H2D for the whole block.
        dev = host.to(device, non_blocking=True)
        state["block_h2d_count"] = state.get("block_h2d_count", 0) + 1
        state["block_layer_count"] = state.get("block_layer_count", 0) + len(items)
        # 4) Slice the device buffer back into per-leaf views, then rebuild each
        #    weight/bias tensor (sharing the one device buffer's storage).
        dev_leaves = []
        for leaf, off in zip(src_leaves, offsets):
            nb = leaf.numel() * leaf.element_size()
            dev_leaves.append(
                dev[off:off + nb].view(leaf.dtype).reshape(leaf.shape)
            )
        it = iter(dev_leaves)
        for (layer_key, weight_cpu, bias_cpu, n_w, n_b) in items:
            w_gpu = _rebuild_from_leaves(weight_cpu, it)
            b_gpu = _rebuild_from_leaves(bias_cpu, it) if n_b else None
            resident[layer_key] = (w_gpu, b_gpu, ready_event)
            staged_keys.append(layer_key)
        ready_event.record(ts)
    ring.append({
        "key": block_key,
        "layer_keys": staged_keys,
        "free": free_event,
        "bytes": int(total),
    })
    _record_ring_peak(state)
    # The staging buffer is transient: it dies at function exit and returns to
    # torch's caching host allocator (which keeps the storage safe until the
    # async H2D completes, and keeps the commitment -- that cached footprint is
    # the host-cache reserve's territory, not this ledger's). Without this
    # release the ledger grows monotonically, one block per staging call.
    pin_manager.release(host_handle)


def block_forward_done(device, block_key):
    """Record a block's compute-free event (forward post-hook) so its buffers are
    only reclaimed once the block's matmuls have finished reading them."""
    state = _DEVICE_STATE.get(torch.device(device))
    if not state or not state.get("block_stream_enabled"):
        return
    cs = torch.cuda.current_stream()
    for entry in reversed(state.get("block_ring", [])):
        if entry["key"] == block_key and entry["free"] is not None:
            entry["free"].record(cs)
            break


def consume_block_resident(device, layer_key):
    """Return ``(w_gpu, b_gpu)`` for a block-staged layer after making the compute
    stream wait on the block's H2D, or None if the layer was not block-staged."""
    state = _DEVICE_STATE.get(torch.device(device))
    if not state or not state.get("block_stream_enabled"):
        return None
    entry = state["block_resident"].get(layer_key)
    if entry is None:
        return None
    w_gpu, b_gpu, ready_event = entry
    torch.cuda.current_stream().wait_event(ready_event)
    return w_gpu, b_gpu


def _stage_grads_to_cpu(state, idx, grad_w_gpu, grad_b_gpu):
    """Copy freshly-computed device grads (in staging slot idx) to CPU on the
    grad stream, overlapping the next H2D. Returns (grad_w_cpu, grad_b_cpu)."""
    gs = state["transfer_grad_stream"]
    state["grad_compute_done"][idx].record()  # on the compute stream
    grad_w_cpu = grad_b_cpu = None
    with torch.cuda.stream(gs):
        gs.wait_event(state["grad_compute_done"][idx])
        if grad_w_gpu is not None:
            grad_w_cpu = grad_w_gpu.to("cpu", non_blocking=True)
        if grad_b_gpu is not None:
            grad_b_cpu = grad_b_gpu.to("cpu", non_blocking=True)
        state["grad_xfer_done"][idx].record()
    return grad_w_cpu, grad_b_cpu


# (ADD) detect torchao wrapper tensors
def _is_ao_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    try:
        if has_torch_function_unary(t):
            return t.__class__.__module__.startswith("torchao.")
    except Exception:
        pass
    for attr in (
        "_scale",
        "_scales",
        "_zero_point",
        "_zp",
        "_block_size",
        "_group_size",
        "_pack_dim",
    ):
        if hasattr(t, attr):
            return True
    return False


def _is_quantized_tensor(t: Optional[torch.Tensor]) -> bool:
    if t is None:
        return False
    # torch quantized tensors
    try:
        if torch.is_quantized(t):  # type: ignore[attr-defined]
            return True
    except Exception:
        pass
    # (ADD) torchao quantized wrappers
    if _is_ao_quantized_tensor(t):
        return True
    # packed/int formats (weight-only)
    return not t.dtype.is_floating_point


def _wrapper_to_async(t, device):
    """Move a tensor-subclass (e.g. TorchAO float8) to device, forwarding
    non_blocking=True to every inner leaf.

    TorchAO's own AffineQuantizedTensor.to(device, non_blocking=True) does not
    propagate non_blocking to its physical storage leaves, so even a fully
    pinned source copies synchronously and the training thread blocks inside the
    H2D enqueue (the measured ~13s/step of "pinned submit"). Rebuilding the
    wrapper here with explicit non_blocking leaf moves makes the transfer truly
    async so it overlaps compute on the transfer stream.
    """
    try:
        names, ctx = t.__tensor_flatten__()
    except Exception:
        return t.to(device, non_blocking=True)
    moved = {}
    for name in names:
        inner = getattr(t, name, None)
        if inner is None:
            moved[name] = None
        elif hasattr(inner, "__tensor_flatten__"):
            moved[name] = _wrapper_to_async(inner, device)
        else:
            moved[name] = inner.to(device, non_blocking=True)
    return type(t).__tensor_unflatten__(moved, ctx, t.size(), t.stride())


def _slot_bf16_dest(state, idx, shape, dtype, device):
    """Persistent per-slot buffer reused across fetches, grown to the max shape."""
    numel = 1
    for s in shape:
        numel *= s
    buf = state["w_dest"][idx]
    if buf is None or buf.numel() < numel or buf.dtype != dtype:
        buf = torch.empty(numel, dtype=dtype, device=device)
        state["w_dest"][idx] = buf
    return buf.narrow(0, 0, numel).view(*shape)


_REGISTERED_HOST_PIN_LOCK = threading.Lock()
_REGISTERED_HOST_PINS: dict[int, int] = {}


def _pin_tensor_in_place(t: torch.Tensor) -> bool:
    return pin_manager.pin_tensor_in_place(t, kind="weights")


def _unpin_tensor_in_place(t: torch.Tensor) -> bool:
    return pin_manager.unpin_tensor_in_place(t, kind="weights")


def _pin_inner_tensors(t: torch.Tensor, budget: int) -> int:
    """Pin the leaf storage of a tensor-subclass (e.g. torchao float8) in place.

    Quantized wrappers can't be pin_memory()'d directly, but they expose their
    real data as inner tensors via __tensor_flatten__. Pinning those lets the
    per-layer H2D bounce run async and overlap with compute instead of blocking.
    """
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        return 0
    pinned = 0
    for name in names:
        inner = getattr(t, name, None)
        if inner is None:
            continue
        if hasattr(inner, "__tensor_flatten__"):
            used = _pin_inner_tensors(inner, max(0, budget - pinned))
            pinned += used
        elif (
            isinstance(inner, torch.Tensor)
            and inner.device.type == "cpu"
            and not inner.is_pinned()
        ):
            size = inner.numel() * inner.element_size()
            if size > budget - pinned:
                continue
            # Process-wide ledger: caps against the WDDM shared-memory-budget
            # proxy, which "budget" alone (a per-call byte cap) cannot see.
            headroom = pinned_bytes_headroom()
            if headroom is not None and size > headroom:
                continue
            if _pin_tensor_in_place(inner):
                pinned += size
            else:
                handle = pin_manager.pin_alloc(size, "weights", required=False)
                if handle.pinned:
                    view = handle.tensor.view(inner.dtype).reshape(inner.shape)
                    view.copy_(inner)
                    setattr(t, name, view)
                    pinned += size
    return pinned


def _empty_host_pin_cache() -> None:
    """Best-effort flush for PyTorch's cached pinned-host allocator."""
    try:
        fn = getattr(torch._C, "_host_emptyCache", None)
        if fn is not None:
            fn()
    except Exception:
        pass
    try:
        fn = getattr(torch._C, "_accelerator_emptyHostCache", None)
        if fn is not None:
            fn()
    except Exception:
        pass


def _unpin_inner_tensors(t: torch.Tensor) -> bool:
    """Replace pinned wrapper leaves with ordinary pageable CPU tensors."""
    try:
        names, _ = t.__tensor_flatten__()
    except Exception:
        return False
    changed = False
    for name in names:
        inner = getattr(t, name, None)
        if inner is None:
            continue
        if hasattr(inner, "__tensor_flatten__"):
            changed = _unpin_inner_tensors(inner) or changed
        elif (
            isinstance(inner, torch.Tensor)
            and inner.device.type == "cpu"
            and inner.is_pinned()
        ):
            size = inner.numel() * inner.element_size()
            if not _unpin_tensor_in_place(inner):
                release_pinned_bytes(size, kind="weights")
                setattr(t, name, inner.clone())
            changed = True
    return changed


def _unpin_module_weights(module: nn.Module, manager) -> int:
    """Undo the page-locking on a module's weights, in place (pinned -> pageable).

    Shared core of ``unpin_layer`` and the transactional attach rollback.
    ``_unpin_inner_tensors`` already releases the process-wide bounce ledger for
    quantized inner tensors; this additionally reconciles the manager-side
    ``pinned_weight_bytes`` counter and the per-layer ``_mm_pinned_bytes`` tag so
    later pins can reuse the budget. Returns bytes released (0 if none held).

    """
    tracked = int(getattr(module, "_mm_pinned_bytes", 0) or 0)
    changed = False
    with torch.no_grad():
        for name in ("weight", "bias"):
            # OstrisLinear.weight is a dequantizing property. Reading the
            # registration table avoids materializing a full float weight.
            param = module._parameters.get(name, None)
            if not isinstance(param, nn.Parameter):
                continue
            data = param.data
            if _is_quantized_tensor(data) or hasattr(data, "__tensor_flatten__"):
                # Releases the bounce ledger per inner tensor internally.
                changed = _unpin_inner_tensors(data) or changed
            elif (
                isinstance(data, torch.Tensor)
                and data.device.type == "cpu"
                and data.is_pinned()
            ):
                size = data.numel() * data.element_size()
                if not _unpin_tensor_in_place(data):
                    release_pinned_bytes(size, kind="weights")
                    param.data = data.clone()
                changed = True
        if getattr(module, "is_ostris_quantized", False):
            for name, data in list(module._buffers.items()):
                if not isinstance(data, torch.Tensor):
                    continue
                if _is_quantized_tensor(data) or hasattr(data, "__tensor_flatten__"):
                    changed = _unpin_inner_tensors(data) or changed
                elif data.device.type == "cpu" and data.is_pinned():
                    size = data.numel() * data.element_size()
                    if not _unpin_tensor_in_place(data):
                        release_pinned_bytes(size, kind="weights")
                        module._buffers[name] = data.clone()
                    changed = True
    if not changed:
        return 0
    # Drop Python references to the old pinned tensors promptly. WDDM NON_LOCAL
    # reclaim can lag, but holding references here guarantees it cannot happen.
    try:
        gc.collect()
    except Exception:
        pass
    _empty_host_pin_cache()
    # The bounce ledger was already released above (do NOT release it again
    # here, unlike promote_layer which never calls _unpin_inner_tensors).
    if manager is not None and tracked:
        manager.pinned_weight_bytes = max(0, manager.pinned_weight_bytes - tracked)
    module._mm_pinned_bytes = 0
    return tracked


def unpin_layer(child: nn.Module) -> int:
    """Unpin a still-streamed layer's weights back to pageable CPU, in place.

    Shared-cliff relief primitive (§4 of the pin-for-speed policy): unlike
    ``promote_layer`` (pinned -> resident, which spends *dedicated* VRAM), this
    reverts a pinned layer to bounce-streamed pageable CPU, freeing WDDM shared
    budget at zero dedicated-VRAM cost. The streaming forward stays installed
    (the layer remains ``_layer_memory_manager``-attached); only the page-locking
    is undone, so the very next forward re-streams it through the bounce pool.
    """
    lmm = getattr(child, "_layer_memory_manager", None)
    if lmm is None:
        return 0
    return _unpin_module_weights(child, getattr(lmm, "manager", None))


def _dxgi_signed_headroom_bytes(device=None) -> Optional[int]:
    """Fresh NON_LOCAL ``Budget - CurrentUsage - spill_reserve`` (signed), or None.

    Signed (may go negative) unlike ``pinned_bytes_headroom`` which clamps to 0,
    so the transactional attach can tell "exactly at margin" from "overshot past
    it" and roll back only in the latter case. Reads DXGI fresh (min_interval_s=0)
    -- this is an allocation gate, not telemetry.
    """
    dxgi = get_dxgi_meminfo()
    if dxgi is None:
        return None
    cuda_index = _cuda_device_index(device)
    try:
        if not dxgi.control_is_eligible(cuda_index):
            return None
    except Exception:
        return None
    info = dxgi.query_non_local_video_memory_info(
        cuda_device_index=cuda_index,
        min_interval_s=0.0,
    )
    if info is None:
        return None
    reserve = dxgi_spill_reserve_bytes(info.budget_bytes)
    return int(info.budget_bytes) - int(info.current_usage_bytes) - int(reserve)


def _ensure_cpu_pinned(
    t: Optional[torch.Tensor], budget: int
) -> Tuple[Optional[torch.Tensor], int]:
    if t is None:
        return None, 0
    if t.device.type != "cpu":
        try:
            t = t.to("cpu", copy=True)
        except Exception:
            t = t.to("cpu")
    # Tensor-subclass wrappers (torchao AQT, quanto QBytesTensor) must have
    # their inner storage pinned in place: wrapper-level pin_memory() falls
    # back to a dequantize round-trip that burns ~2x the storage in host RAM
    # and leaves the real data unpinned. quanto is deliberately not
    # _is_quantized_tensor (the streaming forward depends on that), so gate
    # on __tensor_flatten__ as well.
    if _is_quantized_tensor(t) or hasattr(t, "__tensor_flatten__"):
        if torch.cuda.is_available():
            return t, _pin_inner_tensors(t, budget)
        return t, 0
    size = t.numel() * t.element_size()
    if torch.cuda.is_available() and size <= budget:
        headroom = pinned_bytes_headroom()
        if headroom is None or size <= headroom:
            if _pin_tensor_in_place(t):
                return t, size
            handle = pin_manager.pin_alloc(size, "weights", required=False)
            if handle.pinned:
                pinned_t = handle.tensor.view(t.dtype).reshape(t.shape)
                pinned_t.copy_(t)
                return pinned_t, size
    return t, 0



def _move_params_to_cpu_and_pin(module: nn.Module, manager: "MemoryManager"):
    """Force parameters to CPU (+pinned) so we can 'bounce' them per forward/backward."""
    dxgi_before = _dxgi_signed_headroom_bytes(getattr(manager, "process_device", None))
    pinned_before = int(getattr(module, "_mm_pinned_bytes", 0) or 0)
    with torch.no_grad():
        for name in ("weight", "bias"):
            param = getattr(module, name, None)
            if not isinstance(param, nn.Parameter):
                continue
            remaining = max(
                0,
                manager.pinned_weight_budget_bytes - manager.pinned_weight_bytes,
            )
            if dxgi_before is not None:
                remaining = min(remaining, max(0, int(dxgi_before)))
            cpu_data, pinned = _ensure_cpu_pinned(param.data, remaining)
            manager.pinned_weight_bytes += pinned
            # Track per-module pinned bytes so promote_layer can give the budget
            # back when it moves this weight to GPU (otherwise the counter leaks
            # upward across promote/demote cycles and later demotions can't pin).
            module._mm_pinned_bytes = getattr(module, "_mm_pinned_bytes", 0) + pinned
            cpu_data = cpu_data.detach()
            if _is_quantized_tensor(param.data) or hasattr(param.data, "__tensor_flatten__"):
                # Tensor-subclass weights (e.g. torchao float8 AffineQuantizedTensor)
                # ignore `param.data = ...`: the wrapper reports CPU but its inner
                # storage stays on the GPU, so the weight never actually offloads.
                # Replace the whole Parameter so the device move sticks.
                setattr(
                    module,
                    name,
                    nn.Parameter(cpu_data, requires_grad=param.requires_grad),
                )
            else:
                param.data = cpu_data
    pinned_delta = int(getattr(module, "_mm_pinned_bytes", 0) or 0) - pinned_before
    if dxgi_before is not None and pinned_delta > 0:
        dxgi_after = _dxgi_signed_headroom_bytes(getattr(manager, "process_device", None))
        if dxgi_after is not None and dxgi_after < 0:
            released = _unpin_module_weights(module, manager)
            if released:
                try:
                    print(
                        "[MemoryManager] DXGI pin rollback: "
                        f"layer={getattr(module, '_mm_layer_key', module.__class__.__name__)} "
                        f"released={released / 1024 ** 3:.2f} GiB "
                        f"headroom_before={dxgi_before / 1024 ** 3:.2f} GiB "
                        f"headroom_after={dxgi_after / 1024 ** 3:.2f} GiB"
                    )
                except Exception:
                    pass


# ==========================
# Autograd functions (CUDA)
# ==========================


class _BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight_cpu,
        bias_cpu,
        device: torch.device,
        fp8_sampling=False,
        layer_key=None,
    ):
        ctx.layer_key = layer_key
        if _TRACE_ENABLED:
            record_weight_access(
                layer_key,
                "forward",
                _profile_bytes(weight_cpu),
                weight_cpu.numel() * 2,
            )
        # choose compute dtype to match activations
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_linear_weight(
            cpu_w, dev, dest_fn=None, profile_midpoint=None
        ):
            if _is_quantized_tensor(cpu_w):
                if fp8_sampling:
                    w_q_gpu = _wrapper_to_async(cpu_w, dev)
                    if profile_midpoint is not None:
                        profile_midpoint()
                    return w_q_gpu
                # move quantized wrapper to GPU -> dequantize on GPU -> cast on GPU
                w_q_gpu = _wrapper_to_async(cpu_w, dev)
                if profile_midpoint is not None:
                    profile_midpoint()
                dest = dest_fn(w_q_gpu.shape, target_dtype) if dest_fn else None
                reused = _dequantize_into(w_q_gpu, dest)
                if reused is not None:
                    return reused
                try:
                    w_fp_gpu = _dequantize_to(w_q_gpu, target_dtype)
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w_gpu = cpu_w.to(dev, non_blocking=True)
            if profile_midpoint is not None:
                profile_midpoint()
            return w_gpu

        if device.type != "cuda":
            bias_compute = (
                bias_cpu.to(dtype=target_dtype) if bias_cpu is not None else None
            )
            out = F.linear(
                x.to("cpu"),
                _materialize_linear_weight(weight_cpu, torch.device("cpu")),
                bias_compute,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.device = torch.device("cpu")
            return out.to(x.device)

        # Block streaming (opt-in): if a block pre-hook already staged this
        # weight to GPU, consume it instead of issuing a per-Linear H2D. Only the
        # plain dequant/float forward takes this path; fp8-native forward keeps
        # the per-Linear staging below. Backward is unchanged either way, so the
        # gradient is computed from the same weight value.
        if not fp8_sampling:
            block_resident = consume_block_resident(device, layer_key)
            if block_resident is not None:
                w_gpu, b_gpu = block_resident
                pool = get_prefetch_pool(device)
                if pool is not None:
                    pool.consume_without_transfer(layer_key, operation="forward")
                if _is_quantized_tensor(w_gpu):
                    w_gpu = _dequantize_to(w_gpu, target_dtype)
                if w_gpu.dtype != target_dtype:
                    w_gpu = w_gpu.to(dtype=target_dtype)
                if b_gpu is not None and b_gpu.dtype != target_dtype:
                    b_gpu = b_gpu.to(dtype=target_dtype)
                out = F.linear(x, w_gpu, b_gpu)
                ctx.save_for_backward(x, weight_cpu, bias_cpu)
                ctx.device = device
                ctx.target_dtype = target_dtype
                return out

        state = _get_device_state(device)
        idx, w_gpu, b_gpu = _stage_forward_weight(
            state, device, _materialize_linear_weight, weight_cpu, bias_cpu,
            layer_key
        )
        out = fp8_linear_inference(x, w_gpu, b_gpu) if fp8_sampling else None
        if out is None:
            if _is_quantized_tensor(w_gpu):
                w_gpu = _dequantize_to(w_gpu, target_dtype)
            if b_gpu is not None and b_gpu.dtype != target_dtype:
                b_gpu = b_gpu.to(dtype=target_dtype)
            out = F.linear(x, w_gpu, b_gpu)
        _release_forward_slot(state, idx)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        ctx.target_dtype = target_dtype
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        target_dtype = getattr(ctx, "target_dtype", grad_out.dtype)
        if _TRACE_ENABLED:
            record_weight_access(
                getattr(ctx, "layer_key", None),
                "backward",
                _profile_bytes(weight_cpu),
                weight_cpu.numel() * 2,
            )

        if device.type != "cuda":
            go_cpu = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_mat = (
                _dequantize_to(weight_cpu, target_dtype)
                if _is_quantized_tensor(weight_cpu)
                else weight_cpu
            )
            if w_mat.dtype != target_dtype and target_dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                w_mat = w_mat.to(target_dtype)
            grad_input = go_cpu @ w_mat
            grad_weight = (
                go_cpu.flatten(0, -2).T @ x_cpu.flatten(0, -2)
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go_cpu.sum(dim=tuple(range(go_cpu.ndim - 1)))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return (
                grad_input.to(grad_out.device),
                grad_weight,
                grad_bias,
                None,
                None,
                None,
            )

        state = _get_device_state(device)

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_for_bwd(cpu_w, dest_fn=None, profile_midpoint=None):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = _wrapper_to_async(cpu_w, device)
                if profile_midpoint is not None:
                    profile_midpoint()
                # Let the bound quantized operation decide whether it can keep
                # the wrapped representation for grad-input execution.
                if _fp8_grad_input_supported(w_q_gpu, grad_out):
                    return w_q_gpu
                dest = dest_fn(w_q_gpu.shape, target_dtype) if dest_fn else None
                reused = _dequantize_into(w_q_gpu, dest)
                if reused is not None:
                    return reused
                try:
                    w_fp_gpu = _dequantize_to(w_q_gpu, target_dtype)
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w = cpu_w.to(device, non_blocking=True)
            if profile_midpoint is not None:
                profile_midpoint()
            return w

        idx, w_bwd = _stage_backward_weight(
            state, device, _materialize_for_bwd, weight_cpu,
            getattr(ctx, "layer_key", None), target_dtype
        )

        # grad wrt input (GPU)
        grad_input = _fp8_grad_input(grad_out, w_bwd, target_dtype)
        if grad_input is None:
            grad_input = grad_out.to(dtype=target_dtype) @ w_bwd
        _release_backward_weight_slot(state, idx)

        # compute grads if float masters exist (frozen/quantized bases skip this)
        grad_weight = None
        grad_bias = None
        need_w = (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        )
        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
        if need_w or need_b:
            # ensure the prior grad D2H using this staging slot finished
            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
            w_grad_gpu = b_grad_gpu = None
            if need_w:
                w_grad_gpu = grad_out.flatten(0, -2).T @ x.flatten(0, -2)
                state["w_grad_buffers"][idx] = w_grad_gpu
            if need_b:
                b_grad_gpu = grad_out.sum(dim=tuple(range(grad_out.ndim - 1)))
                state["b_grad_buffers"][idx] = b_grad_gpu
            _record_ring_peak(state)
            grad_weight, grad_bias = _stage_grads_to_cpu(
                state, idx, w_grad_gpu, b_grad_gpu
            )

        return (
            grad_input.to(dtype=grad_out.dtype),
            grad_weight,
            grad_bias,
            None,
            None,
            None,
        )


class _BouncingConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight_cpu,
        bias_cpu,
        device: torch.device,
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        dilation: Tuple[int, int],
        groups: int,
        layer_key=None,
    ):
        ctx.layer_key = layer_key
        if _TRACE_ENABLED:
            record_weight_access(
                layer_key,
                "forward",
                _profile_bytes(weight_cpu),
                weight_cpu.numel() * 2,
            )
        target_dtype = (
            x.dtype
            if x.dtype in (torch.bfloat16, torch.float16, torch.float32)
            else torch.bfloat16
        )

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_conv_weight(
            cpu_w, dev, dest_fn=None, profile_midpoint=None
        ):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = _wrapper_to_async(cpu_w, dev)
                if profile_midpoint is not None:
                    profile_midpoint()
                dest = dest_fn(w_q_gpu.shape, target_dtype) if dest_fn else None
                reused = _dequantize_into(w_q_gpu, dest)
                if reused is not None:
                    return reused
                try:
                    w_fp_gpu = _dequantize_to(w_q_gpu, target_dtype)
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w_gpu = cpu_w.to(dev, non_blocking=True)
            if profile_midpoint is not None:
                profile_midpoint()
            return w_gpu

        if device.type != "cuda":
            bias_compute = (
                bias_cpu.to(dtype=target_dtype) if bias_cpu is not None else None
            )
            out = F.conv2d(
                x.to("cpu"),
                _materialize_conv_weight(weight_cpu, torch.device("cpu")),
                bias_compute,
                stride,
                padding,
                dilation,
                groups,
            )
            ctx.save_for_backward(x.to("cpu"), weight_cpu, bias_cpu)
            ctx.meta = ("cpu", stride, padding, dilation, groups, target_dtype)
            return out.to(x.device)

        state = _get_device_state(device)
        idx, w_gpu, b_gpu = _stage_forward_weight(
            state, device, _materialize_conv_weight, weight_cpu, bias_cpu,
            layer_key
        )
        if b_gpu is not None and b_gpu.dtype != target_dtype:
            b_gpu = b_gpu.to(dtype=target_dtype)
        out = F.conv2d(x, w_gpu, b_gpu, stride, padding, dilation, groups)
        _release_forward_slot(state, idx)

        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.meta = (device, stride, padding, dilation, groups, target_dtype)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device, stride, padding, dilation, groups, target_dtype = ctx.meta
        if _TRACE_ENABLED:
            record_weight_access(
                getattr(ctx, "layer_key", None),
                "backward",
                _profile_bytes(weight_cpu),
                weight_cpu.numel() * 2,
            )

        if (
            isinstance(device, torch.device) and device.type != "cuda"
        ) or device == "cpu":
            go = grad_out.to("cpu")
            x_cpu = x.to("cpu")
            w_cpu = (
                _dequantize_to(weight_cpu, target_dtype)
                if _is_quantized_tensor(weight_cpu)
                else weight_cpu
            )
            if w_cpu.dtype != target_dtype and target_dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float32,
            ):
                w_cpu = w_cpu.to(target_dtype)
            from torch.nn.grad import conv2d_input, conv2d_weight  # type: ignore

            grad_input = conv2d_input(
                x_cpu.shape,
                w_cpu,
                go,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            )
            grad_weight = (
                conv2d_weight(
                    x_cpu,
                    w_cpu.shape,
                    go,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                if getattr(weight_cpu, "requires_grad", False)
                and weight_cpu.dtype.is_floating_point
                else None
            )
            grad_bias = (
                go.sum(dim=(0, 2, 3))
                if (bias_cpu is not None and getattr(bias_cpu, "requires_grad", False))
                else None
            )
            return (
                grad_input.to(grad_out.device),
                grad_weight,
                grad_bias,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        state = _get_device_state(device)

        # GPU-side dequant/cast for quantized; float path unchanged
        def _materialize_for_bwd(cpu_w, dest_fn=None, profile_midpoint=None):
            if _is_quantized_tensor(cpu_w):
                w_q_gpu = _wrapper_to_async(cpu_w, device)
                if profile_midpoint is not None:
                    profile_midpoint()
                dest = dest_fn(w_q_gpu.shape, target_dtype) if dest_fn else None
                reused = _dequantize_into(w_q_gpu, dest)
                if reused is not None:
                    return reused
                try:
                    w_fp_gpu = _dequantize_to(w_q_gpu, target_dtype)
                except Exception:
                    w_fp_gpu = w_q_gpu.to(dtype=torch.float32, non_blocking=True)
                if w_fp_gpu.dtype != target_dtype:
                    w_fp_gpu = w_fp_gpu.to(target_dtype, non_blocking=True)
                return w_fp_gpu
            # float path (preserve original behavior: NO dtype cast)
            w = cpu_w.to(device, non_blocking=True)
            if profile_midpoint is not None:
                profile_midpoint()
            return w

        idx, w_bwd = _stage_backward_weight(
            state, device, _materialize_for_bwd, weight_cpu,
            getattr(ctx, "layer_key", None), target_dtype
        )

        from torch.nn.grad import conv2d_input, conv2d_weight  # type: ignore

        grad_input = conv2d_input(
            x.shape,
            w_bwd,
            grad_out.to(dtype=target_dtype),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        _release_backward_weight_slot(state, idx)

        # Compute heavy grads on GPU into staging buffers (frozen bases skip this)
        grad_weight = None
        grad_bias = None
        need_w = (
            getattr(weight_cpu, "requires_grad", False)
            and weight_cpu.dtype.is_floating_point
        )
        need_b = bias_cpu is not None and getattr(bias_cpu, "requires_grad", False)
        if need_w or need_b:
            torch.cuda.current_stream().wait_event(state["grad_xfer_done"][idx])
            w_grad_gpu = b_grad_gpu = None
            if need_w:
                w_grad_gpu = conv2d_weight(
                    x,
                    weight_cpu.shape,
                    grad_out,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                state["w_grad_buffers"][idx] = w_grad_gpu
            if need_b:
                b_grad_gpu = grad_out.sum(dim=(0, 2, 3))
                state["b_grad_buffers"][idx] = b_grad_gpu
            _record_ring_peak(state)
            grad_weight, grad_bias = _stage_grads_to_cpu(
                state, idx, w_grad_gpu, b_grad_gpu
            )

        return (
            grad_input.to(dtype=grad_out.dtype),
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class BaseLayerMemoryManager:
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        self.module: nn.Module = module
        self.manager: "MemoryManager" = manager

    def _capture_base_forward(self):
        """Find the base forward slot without bypassing an attached LoRA."""
        if hasattr(self.module, "ara_lora_ref"):
            owner = self.module.ara_lora_ref()
            if owner is not None and hasattr(owner, "org_forward"):
                self._forward_container = owner
                self._forward_attribute = "org_forward"
                return getattr(owner, "org_forward")
        owner = getattr(getattr(self.module, "forward", None), "__self__", None)
        if (
            owner is not None
            and owner is not self.module
            and hasattr(owner, "org_forward")
        ):
            self._forward_container = owner
            self._forward_attribute = "org_forward"
            return getattr(owner, "org_forward")
        self._forward_container = self.module
        self._forward_attribute = "forward"
        return getattr(self.module, "forward")

    def _install_base_forward(self, forward):
        self._installed_forward = forward
        setattr(self._forward_container, self._forward_attribute, forward)

    def _locate_installed_forward_slot(self):
        """Find where OUR streaming forward lives RIGHT NOW.

        The slot recorded at attach time can go stale: a LoRA applied AFTER
        attach hijacks ``module.forward`` and chains to our forward through its
        ``org_forward`` -- so the recorded ``(module, 'forward')`` slot now
        holds the LoRA hijack, and writing the original there on detach eats
        the LoRA (the lora_hijack_missing boundary crash). Walk the live
        forward chain from the module and return the (container, attribute)
        that actually holds our installed forward, or None if we are not in
        the chain (already unwound, or stripped by the in-graph enable).
        """
        installed = getattr(self, "_installed_forward", None)
        if installed is None:
            return None
        fwd = self.module.__dict__.get("forward")
        if fwd is installed:
            return self.module, "forward"
        seen = set()
        owner = getattr(fwd, "__self__", None)
        while owner is not None and id(owner) not in seen:
            seen.add(id(owner))
            org = getattr(owner, "org_forward", None)
            if org is installed:
                return owner, "org_forward"
            owner = getattr(org, "__self__", None)
        return None

    def _uninstall_base_forward(self):
        """Remove our streaming forward from the live chain and put the
        captured original back -- the inverse of ``_install_base_forward``,
        robust to a LoRA hijack applied after attach. Falls back to the
        attach-time slot when our forward is not in the live chain."""
        slot = self._locate_installed_forward_slot()
        if slot is None:
            container = getattr(self, "_forward_container", None)
            attribute = getattr(self, "_forward_attribute", None)
            if container is None or attribute is None:
                return False
            slot = (container, attribute)
        setattr(slot[0], slot[1], self._original_forward)
        return True

    @classmethod
    def attach(cls, module: nn.Module, manager: "MemoryManager"):
        if hasattr(module, "_layer_memory_manager"):
            return
        module._layer_memory_manager = cls(module, manager)

        # mark parameters as memory managed
        for param in module.parameters(recurse=False):
            param._is_memory_managed = True


class LinearLayerMemoryManager(BaseLayerMemoryManager):
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module, self.manager)

        # 2) Hijack forward
        self._original_forward = self._capture_base_forward()

        # @torch.compiler.disable ensures Dynamo never traces into this function.
        # _BouncingLinearFn is a custom autograd fn that does CPU→GPU staging,
        # optional dequant, and FP8 branching — none of which is compilable.
        # If this forward were inside a torch.compile region it would cause
        # graph-break storms or a multi-minute freeze.
        @torch.compiler.disable
        def _mm_forward(x, *args, **kwargs):
            # ensure we only use expected signature (Linear: x)
            if args or kwargs:
                # fall back to original if a custom signature is used
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            # NOTE: do NOT move params to device here; autograd fn streams & bounces them
            return _BouncingLinearFn.apply(
                x,
                weight_cpu,
                bias_cpu,
                device,
                bool(
                    getattr(self.module, "_memory_management_fp8_sampling", False)
                    or getattr(self.module, "_memory_management_fp8_training", False)
                ),
                getattr(self.module, "_mm_layer_key", None),
            )

        self._install_base_forward(_mm_forward)
        
        self.module._memory_management_device = self.manager.process_device


class OstrisLinearLayerMemoryManager(BaseLayerMemoryManager):
    """Offload manager for OstrisLinear (custom-quantized) layers.

    The generic linear bounce is wrong for these: module.weight is a property that
    fully dequantizes on access, so bouncing it ships a full-precision weight over
    PCIe every forward and bypasses the quantizer's hardware kernels. Instead this
    keeps the (much smaller) quantized buffers pinned on CPU, stages them H2D into
    the same forward ring the float path uses, swaps them onto the module, and runs
    the quantizer's own forward on device — so fp4/int8 GEMM paths and the STE
    training path work unchanged under offloading. Buffers are read live off the
    module each forward (not cached) so requantize_ during merge/reset stays valid.
    """

    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move quantized buffers + bias to CPU and pin within the same
        # manager/DXGI budget used by ordinary streamed weights.
        dxgi_before = _dxgi_signed_headroom_bytes(manager.process_device)
        pinned_before = int(getattr(module, "_mm_pinned_bytes", 0) or 0)
        with torch.no_grad():
            for name, buf in list(module._buffers.items()):
                if buf is None:
                    continue
                if buf.device.type != "cpu":
                    buf = buf.to("cpu")
                remaining = max(
                    0,
                    manager.pinned_weight_budget_bytes - manager.pinned_weight_bytes,
                )
                if dxgi_before is not None:
                    remaining = min(remaining, max(0, int(dxgi_before)))
                cpu_buf, pinned = _ensure_cpu_pinned(buf, remaining)
                manager.pinned_weight_bytes += pinned
                module._mm_pinned_bytes = (
                    getattr(module, "_mm_pinned_bytes", 0) + pinned
                )
                module._buffers[name] = cpu_buf.detach()
            bias = module._parameters.get("bias", None)
            if bias is not None:
                remaining = max(
                    0,
                    manager.pinned_weight_budget_bytes - manager.pinned_weight_bytes,
                )
                if dxgi_before is not None:
                    remaining = min(remaining, max(0, int(dxgi_before)))
                cpu_bias, pinned = _ensure_cpu_pinned(bias.data, remaining)
                manager.pinned_weight_bytes += pinned
                module._mm_pinned_bytes = (
                    getattr(module, "_mm_pinned_bytes", 0) + pinned
                )
                bias.data = cpu_bias.detach()

        pinned_delta = int(getattr(module, "_mm_pinned_bytes", 0) or 0) - pinned_before
        if dxgi_before is not None and pinned_delta > 0:
            dxgi_after = _dxgi_signed_headroom_bytes(manager.process_device)
            if dxgi_after is not None and dxgi_after < 0:
                _unpin_module_weights(module, manager)

        # 2) Hijack forward
        self._original_forward = self._capture_base_forward()

        @torch.compiler.disable
        def _mm_forward(x, *args, **kwargs):
            # ensure we only use expected signature (Linear: x)
            if args or kwargs:
                return self._original_forward(x, *args, **kwargs)

            module = self.module
            device = self.manager.process_device
            if device.type != "cuda":
                return self._original_forward(x)

            cpu_bufs = {
                n: b
                for n, b in module._buffers.items()
                if b is not None and b.device.type == "cpu"
            }
            bias = module._parameters.get("bias", None)
            bias_cpu = (
                bias.data
                if bias is not None and bias.data.device.type == "cpu"
                else None
            )
            if not cpu_bufs and bias_cpu is None:
                # already resident on device
                return self._original_forward(x)

            state = _get_device_state(device)
            d = state["depth"]
            idx = state["forward_clk"]
            state["forward_clk"] = (idx + 1) % d
            ts = state["transfer_stream"]
            # the guard makes current_stream() resolve to the process device and
            # keeps that device's context active for the quantizer's triton
            # kernels (nothing sets the global current device, so it is 0 even
            # when training on another gpu)
            with torch.cuda.device(device):
                with torch.cuda.stream(ts):
                    ts.wait_event(state["fwd_slot_free"][idx])
                    gpu_bufs = {
                        n: b.to(device, non_blocking=True) for n, b in cpu_bufs.items()
                    }
                    gpu_bias = (
                        bias_cpu.to(device, non_blocking=True)
                        if bias_cpu is not None
                        else None
                    )
                    state["w_buffers"][idx] = gpu_bufs
                    state["b_buffers"][idx] = gpu_bias
                    state["fwd_slot_ready"][idx].record()
                torch.cuda.current_stream().wait_event(state["fwd_slot_ready"][idx])

                # swap the quantized state onto the device, run the quantizer's own
                # forward, then swap the pinned CPU state back
                for n, t in gpu_bufs.items():
                    module._buffers[n] = t
                if gpu_bias is not None:
                    bias.data = gpu_bias
                try:
                    out = self._original_forward(x)
                finally:
                    for n, t in cpu_bufs.items():
                        module._buffers[n] = t
                    if bias_cpu is not None:
                        bias.data = bias_cpu
                _release_forward_slot(state, idx)
            return out

        self._install_base_forward(_mm_forward)

        self.module._memory_management_device = self.manager.process_device


class ConvLayerMemoryManager(BaseLayerMemoryManager):
    def __init__(
        self,
        module: nn.Module,
        manager: "MemoryManager",
    ):
        super().__init__(module, manager)

        # 1) Move params to CPU + pin memory for fast H2D
        _move_params_to_cpu_and_pin(self.module, self.manager)

        # Cache static conv attributes from the module
        stride = (
            self.module.stride
            if isinstance(self.module.stride, tuple)
            else (self.module.stride, self.module.stride)
        )
        padding = (
            self.module.padding
            if isinstance(self.module.padding, tuple)
            else (self.module.padding, self.module.padding)
        )
        dilation = (
            self.module.dilation
            if isinstance(self.module.dilation, tuple)
            else (self.module.dilation, self.module.dilation)
        )
        groups = self.module.groups

        # 2) Hijack forward
        self._original_forward = self._capture_base_forward()

        def _mm_forward(x, *args, **kwargs):
            # Support the typical Conv2d(x) call; if user passes uncommon extras, fallback.
            if args or kwargs:
                return self._original_forward(x, *args, **kwargs)

            weight_cpu = self.module.weight
            bias_cpu = getattr(self.module, "bias", None)
            device = self.manager.process_device

            return _BouncingConv2dFn.apply(
                x,
                weight_cpu,
                bias_cpu,
                device,
                stride,
                padding,
                dilation,
                groups,
                getattr(self.module, "_mm_layer_key", None),
            )

        self._install_base_forward(_mm_forward)
        
        self.module._memory_management_device = self.manager.process_device
