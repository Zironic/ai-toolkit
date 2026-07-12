"""Arena-native planning signals. S1 observes only; it performs no actions."""

from __future__ import annotations

from dataclasses import dataclass

_ALLOC = ("num_alloc_retries", "num_device_alloc", "num_device_free")
_COMPILE = ("frames", "graphs", "graph_breaks")


def _deltas(previous, current, keys, cast=int):
    result = {}
    for key in keys:
        now = cast((current or {}).get(key, 0) or 0)
        before = cast((previous or {}).get(key, 0) or 0)
        result[key] = now - before if now >= before else now
    return result


@dataclass(frozen=True)
class ShapePeak:
    steps: int = 0
    warmup_steps: int = 1
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0


class TrainingSignalWindow:
    """Runtime-owned, CPU-testable training policy observations."""

    def __init__(self, *, transfer_window_steps=4):
        self.transfer_window_steps = max(2, int(transfer_window_steps))
        self._allocator_previous = None
        self._compile_previous = None
        self._transfer_previous = None
        self._transfer_steps = 0
        self._transfer_wall_ms = 0.0
        self._transfer_h2d_ms = 0.0
        self._transfer_bytes = 0
        self._shape_peaks = {}
        self._last_signal = None

    @property
    def shape_peaks(self):
        return dict(self._shape_peaks)

    @property
    def last_signal(self):
        return None if self._last_signal is None else dict(self._last_signal)

    @property
    def transfer_snapshot_due(self):
        return self._transfer_steps + 1 >= self.transfer_window_steps

    def invalidate_shape_peaks(self):
        self._shape_peaks.clear()

    def observe(
        self, *, shape_key, step_num, allocator_counters,
        peak_allocated_bytes, peak_reserved_bytes, device_free_bytes,
        resident_bytes, ring_bytes, compile_counters=None,
        transfer_counters=None, step_wall_ms=0.0,
    ):
        alloc_delta = _deltas(self._allocator_previous, allocator_counters, _ALLOC)
        self._allocator_previous = {
            key: int((allocator_counters or {}).get(key, 0) or 0) for key in _ALLOC
        }
        compile_delta = _deltas(
            self._compile_previous, compile_counters, _COMPILE
        )
        compile_invalid = compile_counters is not None and compile_delta["frames"] > 0
        if compile_counters is not None:
            self._compile_previous = {
                key: int(compile_counters.get(key, 0) or 0) for key in _COMPILE
            }
        if compile_invalid:
            self.invalidate_shape_peaks()

        key = _shape_key(shape_key)
        if not compile_invalid:
            self._record_shape_peak(
                key, int(peak_allocated_bytes or 0), int(peak_reserved_bytes or 0)
            )
        transfer = self._observe_transfer(transfer_counters, float(step_wall_ms or 0.0))
        allocated = int(peak_allocated_bytes or 0)
        reserved = int(peak_reserved_bytes or 0)
        resident = int(resident_bytes or 0)
        ring = int(ring_bytes or 0)
        signal = {
            "shape_key": key,
            "step_num": None if step_num is None else int(step_num),
            "allocator": {
                "alloc_retries_delta": alloc_delta["num_alloc_retries"],
                "alloc_count_delta": alloc_delta["num_device_alloc"],
                "free_count_delta": alloc_delta["num_device_free"],
            },
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
            "reclaimable_at_peak_bytes": max(0, reserved - allocated),
            "device_free_bytes": int(device_free_bytes or 0),
            "resident_bytes": resident,
            "ring_bytes": ring,
            "live_bytes": resident + ring,
            "compile_invalid": bool(compile_invalid),
            "compile_delta": compile_delta,
            "transfer": transfer,
        }
        self._last_signal = signal
        return dict(signal)

    def diagnostics(self):
        return {
            "last_signal": self.last_signal,
            "shape_peaks": [
                {
                    "shape_key": key,
                    "steps": peak.steps,
                    "warmup_steps": peak.warmup_steps,
                    "peak_allocated_bytes": peak.peak_allocated_bytes,
                    "peak_reserved_bytes": peak.peak_reserved_bytes,
                }
                for key, peak in self._shape_peaks.items()
            ],
            "transfer_window_steps": self.transfer_window_steps,
            "transfer_pending_steps": self._transfer_steps,
        }

    def _record_shape_peak(self, key, allocated, reserved):
        previous = self._shape_peaks.get(key)
        if previous is None:
            self._shape_peaks[key] = ShapePeak()
            return
        self._shape_peaks[key] = ShapePeak(
            steps=previous.steps + 1,
            warmup_steps=previous.warmup_steps,
            peak_allocated_bytes=max(previous.peak_allocated_bytes, allocated),
            peak_reserved_bytes=max(previous.peak_reserved_bytes, reserved),
        )

    def _observe_transfer(self, counters, wall_ms):
        self._transfer_steps += 1
        self._transfer_wall_ms += max(0.0, wall_ms)
        if counters is not None:
            h2d = _deltas(
                self._transfer_previous, counters, ("h2d_ms",), float
            )["h2d_ms"]
            byte_count = _deltas(
                self._transfer_previous, counters, ("bytes",)
            )["bytes"]
            self._transfer_previous = {
                "h2d_ms": float(counters.get("h2d_ms", 0.0) or 0.0),
                "bytes": int(counters.get("bytes", 0) or 0),
            }
            self._transfer_h2d_ms += h2d
            self._transfer_bytes += byte_count
        if self._transfer_steps < self.transfer_window_steps:
            return None
        duty = (
            None if self._transfer_wall_ms <= 0.0
            else 100.0 * self._transfer_h2d_ms / self._transfer_wall_ms
        )
        gbps = (
            None if self._transfer_h2d_ms <= 0.0
            else self._transfer_bytes / (self._transfer_h2d_ms * 1_000_000.0)
        )
        result = {
            "steps": self._transfer_steps,
            "step_wall_ms": self._transfer_wall_ms,
            "h2d_ms": self._transfer_h2d_ms,
            "bytes": self._transfer_bytes,
            "h2d_duty_pct": duty,
            "h2d_duty_overflow": bool(duty is not None and duty > 100.0),
            "achieved_gbps": gbps,
        }
        self._transfer_steps = 0
        self._transfer_wall_ms = 0.0
        self._transfer_h2d_ms = 0.0
        self._transfer_bytes = 0
        return result


def _shape_key(value):
    if value is None:
        return ("unknown",)
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)
