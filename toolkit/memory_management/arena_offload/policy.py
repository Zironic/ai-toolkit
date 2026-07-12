"""Arena-native training signals and two-timescale residency policy."""

from __future__ import annotations

from dataclasses import dataclass

from .. import vram_budget

_ALLOC = ("num_alloc_retries", "num_device_alloc", "num_device_free")
_COMPILE = ("frames", "graphs", "graph_breaks")

DEFAULT_SLACK_PAD_BYTES = 256 * 1024**2
def transfer_benefits_from_residency(transfer) -> bool:
    """Return whether a valid window proves that weights are still streaming."""
    if not transfer or transfer.get("h2d_duty_overflow"):
        return False
    return (
        int(transfer.get("bytes", 0) or 0) > 0
        and float(transfer.get("h2d_ms", 0.0) or 0.0) > 0.0
    )


@dataclass(frozen=True)
class PolicyDecision:
    action: str
    block_key: str | None = None
    block_bytes: int = 0
    target_cap_bytes: int | None = None
    reason: str = ""


class ArenaResidencyController:
    """Stateful wiring around the pure two-timescale residency FSM."""

    def __init__(self, *, slack_pad_bytes=DEFAULT_SLACK_PAD_BYTES):
        self.state = vram_budget.ResidencyFsmState()
        self.slack_pad_bytes = max(0, int(slack_pad_bytes))
        self.last_action = "hold"
        self.last_reason = "cold_start"
        self.last_promoted_key = None
        self.last_block_key = None
        self.last_block_bytes = 0
        self.last_target_cap_bytes = None
        self.last_worst_shape_margin_bytes = None
        self.last_throughput_gate = None
        self.last_promote_gate = None
        self.last_cap_covers_promo = None
        self.bootstrapped = False

    def step(self, signal, *, candidate, demote_candidate, cliff_cap_bytes,
             worst_shape_free_bytes, current_cap_bytes=None):
        if not self.bootstrapped:
            self.bootstrapped = True
            if demote_candidate is not None:
                return self._decision(
                    "demote",
                    demote_candidate,
                    reason="approach_from_below",
                )
        if signal is None:
            return self._hold("awaiting_signal")

        block_bytes = 0 if candidate is None else int(candidate["block_bytes"])
        retries = int(
            (signal.get("allocator") or {}).get("alloc_retries_delta", 0) or 0
        )
        reclaimable = int(signal.get("reclaimable_at_peak_bytes", 0) or 0)
        live = int(
            signal.get("peak_allocated_bytes", signal.get("live_bytes", 0)) or 0
        )
        throughput_ok = transfer_benefits_from_residency(signal.get("transfer"))
        worst_ok = candidate is not None and int(worst_shape_free_bytes) >= 0
        promote_ok = (
            candidate is not None
            and throughput_ok
            and worst_ok
            and vram_budget.residency_promote_ok(
                retries, reclaimable, block_bytes, self.slack_pad_bytes
            )
        )
        active_cap = (
            int(cliff_cap_bytes)
            if current_cap_bytes is None
            else int(current_cap_bytes)
        )
        cap_covers = (
            candidate is not None
            and vram_budget.cap_can_host_promotion(
                live, block_bytes, self.slack_pad_bytes, active_cap
            )
        )
        binding = retries > 0 or int(worst_shape_free_bytes) < 0
        self.last_worst_shape_margin_bytes = int(worst_shape_free_bytes)
        self.last_throughput_gate = bool(throughput_ok)
        self.last_promote_gate = bool(promote_ok)
        self.last_cap_covers_promo = bool(cap_covers)
        cap_raise_bytes = (
            block_bytes
            if promote_ok and block_bytes > 0
            else self.slack_pad_bytes
        )
        needed_cap = min(
            int(cliff_cap_bytes),
            active_cap + max(0, int(cap_raise_bytes)),
        )
        self.state, action = vram_budget.residency_fsm_step(
            self.state,
            {
                "measurements_invalid": bool(signal.get("compile_invalid")),
                "binding": binding,
                "cap_can_relieve": (
                    active_cap < needed_cap <= int(cliff_cap_bytes)
                ),
                "promote_gate": promote_ok,
                "cap_covers_promo": cap_covers,
            },
        )
        if action == vram_budget.ACT_PROMOTE and candidate is not None:
            self.last_promoted_key = candidate["block_key"]
            return self._decision(
                action, candidate, reason="safe_transfer_benefit"
            )
        if (
            action == vram_budget.ACT_ROLLBACK
            and self.last_promoted_key is not None
        ):
            key = self.last_promoted_key
            self.last_promoted_key = None
            return self._decision(
                action, {"block_key": key, "block_bytes": 0},
                reason="promotion_bound",
            )
        if action == vram_budget.ACT_DEMOTE and demote_candidate is not None:
            return self._decision(
                action, demote_candidate, reason="live_pressure"
            )
        if action == vram_budget.ACT_RAISE_CAP:
            self.last_action = action
            self.last_reason = "prefund_or_relieve"
            self.last_block_key = None
            self.last_block_bytes = 0
            self.last_target_cap_bytes = needed_cap
            return PolicyDecision(
                action, target_cap_bytes=needed_cap, reason=self.last_reason
            )
        reason = (
            "worst_shape_veto"
            if candidate is not None and not worst_ok
            else (
                "throughput_gate"
                if candidate is not None and not throughput_ok
                else "fsm_hold"
            )
        )
        return self._hold(reason, candidate=candidate)

    def _decision(self, action, candidate, *, reason):
        self.last_action = action
        self.last_reason = reason
        self.last_block_key = candidate["block_key"]
        self.last_block_bytes = int(candidate["block_bytes"])
        self.last_target_cap_bytes = None
        return PolicyDecision(
            action,
            candidate["block_key"],
            int(candidate["block_bytes"]),
            reason=reason,
        )

    def _hold(self, reason, *, candidate=None):
        self.last_action = "hold"
        self.last_reason = reason
        self.last_block_key = (
            None if candidate is None else candidate["block_key"]
        )
        self.last_block_bytes = (
            0 if candidate is None else int(candidate["block_bytes"])
        )
        self.last_target_cap_bytes = None
        return PolicyDecision("hold", reason=reason)

    def diagnostics(self):
        return {
            "state": self.state.name,
            "windows_in_state": self.state.windows_in_state,
            "last_action": self.last_action,
            "last_reason": self.last_reason,
            "last_promoted_key": self.last_promoted_key,
            "last_block_key": self.last_block_key,
            "last_block_bytes": self.last_block_bytes,
            "last_target_cap_bytes": self.last_target_cap_bytes,
            "last_worst_shape_margin_bytes": self.last_worst_shape_margin_bytes,
            "last_throughput_gate": self.last_throughput_gate,
            "last_promote_gate": self.last_promote_gate,
            "last_cap_covers_promo": self.last_cap_covers_promo,
            "slack_pad_bytes": self.slack_pad_bytes,
        }


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
        self._last_signal = None

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
