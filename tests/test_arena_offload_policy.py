import contextlib
import json
from types import SimpleNamespace

import pytest

from toolkit.memory_management.arena_offload.policy import (
    ArenaResidencyController,
    TrainingSignalWindow,
    transfer_benefits_from_residency,
)
from toolkit.memory_management.arena_offload.runtime import ArenaOffloadRuntime


def observe(window, **overrides):
    values = {
        "shape_key": (512, 512),
        "step_num": 1,
        "allocator_counters": {
            "num_alloc_retries": 10,
            "num_device_alloc": 20,
            "num_device_free": 30,
        },
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 140,
        "device_free_bytes": 500,
        "resident_bytes": 200,
        "ring_bytes": 80,
        "compile_counters": None,
        "transfer_counters": None,
        "step_wall_ms": 10.0,
    }
    values.update(overrides)
    return window.observe(**values)


def test_allocator_deltas_tolerate_counter_reset():
    window = TrainingSignalWindow()
    first = observe(window)
    assert first["allocator"] == {
        "alloc_retries_delta": 10,
        "alloc_count_delta": 20,
        "free_count_delta": 30,
    }
    second = observe(
        window,
        allocator_counters={
            "num_alloc_retries": 12,
            "num_device_alloc": 25,
            "num_device_free": 37,
        },
    )
    assert second["allocator"] == {
        "alloc_retries_delta": 2,
        "alloc_count_delta": 5,
        "free_count_delta": 7,
    }
    reset = observe(
        window,
        allocator_counters={
            "num_alloc_retries": 1,
            "num_device_alloc": 2,
            "num_device_free": 3,
        },
    )
    assert reset["allocator"] == {
        "alloc_retries_delta": 1,
        "alloc_count_delta": 2,
        "free_count_delta": 3,
    }


def test_per_shape_peaks_skip_warmup_and_track_independently():
    window = TrainingSignalWindow()
    observe(window, peak_allocated_bytes=100, peak_reserved_bytes=140)
    observe(window, peak_allocated_bytes=110, peak_reserved_bytes=150)
    observe(
        window,
        shape_key=(768, 768),
        peak_allocated_bytes=200,
        peak_reserved_bytes=260,
    )
    peaks = window.shape_peaks
    assert peaks[(512, 512)].steps == 1
    assert peaks[(512, 512)].peak_allocated_bytes == 110
    assert peaks[(512, 512)].peak_reserved_bytes == 150
    assert peaks[(768, 768)].steps == 0


def test_compile_activity_invalidates_shape_measurements():
    window = TrainingSignalWindow()
    observe(window)
    observe(window, peak_allocated_bytes=110)
    assert window.shape_peaks[(512, 512)].steps == 1

    invalid = observe(
        window,
        compile_counters={"frames": 4, "graphs": 2, "graph_breaks": 0},
    )
    assert invalid["compile_invalid"] is True
    assert window.shape_peaks == {}

    stable = observe(
        window,
        compile_counters={"frames": 4, "graphs": 2, "graph_breaks": 0},
    )
    assert stable["compile_invalid"] is False
    assert window.shape_peaks[(512, 512)].warmup_steps == 1


def test_transfer_metrics_require_settled_multi_step_window_and_handle_reset():
    window = TrainingSignalWindow(transfer_window_steps=3)
    assert observe(window)["transfer"] is None
    assert observe(window)["transfer"] is None
    settled = observe(
        window,
        transfer_counters={"h2d_ms": 12.0, "bytes": 96_000_000},
    )["transfer"]
    assert settled["steps"] == 3
    assert settled["h2d_duty_pct"] == pytest.approx(40.0)
    assert settled["achieved_gbps"] == pytest.approx(8.0)
    assert settled["h2d_duty_overflow"] is False

    observe(window)
    observe(window)
    after_reset = observe(
        window,
        transfer_counters={"h2d_ms": 6.0, "bytes": 42_000_000},
    )["transfer"]
    assert after_reset["h2d_duty_pct"] == pytest.approx(20.0)
    assert after_reset["achieved_gbps"] == pytest.approx(7.0)




def test_runtime_invalidates_shape_peaks_after_layout_change():
    signals = TrainingSignalWindow()
    observe(signals)
    observe(signals, peak_allocated_bytes=110)
    assert signals.shape_peaks

    next_plan = object()
    executor = SimpleNamespace(
        transition_training_block=lambda key, resident: {
            "changed": True,
            "block_key": key,
            "resident": resident,
            "plan": next_plan,
        }
    )
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._executor = executor
    runtime._signals = signals
    runtime._training_plan = object()

    result = runtime.transition_training_block("blocks.3", resident=True)
    assert result["plan"] is next_plan
    assert runtime._training_plan is next_plan
    assert signals.shape_peaks == {}


def test_training_cap_binding_uses_configured_phase_margin(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.runtime."
        "allocator_cap.apply_wddm_hard_allocator_cap",
        lambda device, hard, **kwargs: calls.append((device, hard, kwargs)),
    )
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._device = "cuda:1"
    runtime._config = SimpleNamespace(
        legacy=SimpleNamespace(wddm_hard_gib=1.25)
    )

    runtime._bind_training_cap()
    assert calls == [
        ("cuda:1", 1.25, {"log_prefix": "[ArenaOffload]"})
    ]
def test_signal_contains_memory_accounting():
    signal = observe(TrainingSignalWindow())
    assert signal["reclaimable_at_peak_bytes"] == 40
    assert signal["device_free_bytes"] == 500
    assert signal["resident_bytes"] == 200
    assert signal["ring_bytes"] == 80
    assert signal["live_bytes"] == 280
    json.dumps(TrainingSignalWindow().diagnostics())

def test_transfer_benefit_gate_requires_valid_nonzero_streaming():
    assert not transfer_benefits_from_residency(None)
    assert not transfer_benefits_from_residency({"bytes": 0, "h2d_ms": 20.0})
    assert not transfer_benefits_from_residency({"bytes": 100, "h2d_ms": 0.0})
    assert not transfer_benefits_from_residency(
        {"bytes": 100, "h2d_ms": 20.0, "h2d_duty_overflow": True}
    )
    assert transfer_benefits_from_residency(
        {"bytes": 100, "h2d_ms": 20.0, "h2d_duty_pct": 1.0}
    )


def test_controller_promotes_exact_candidate_then_rolls_it_back():
    controller = ArenaResidencyController(slack_pad_bytes=10)
    candidate = {"block_key": "blocks.3", "block_bytes": 20}
    clean = {
        "allocator": {"alloc_retries_delta": 0},
        "peak_allocated_bytes": 100,
        "reclaimable_at_peak_bytes": 100,
        "compile_invalid": False,
        "transfer": {
            "bytes": 100,
            "h2d_ms": 20.0,
            "h2d_duty_pct": 80.0,
            "achieved_gbps": 10.0,
        },
    }
    actions = []
    for _ in range(4):
        decision = controller.step(
            clean,
            candidate=candidate,
            demote_candidate=None,
            cliff_cap_bytes=1000,
            worst_shape_free_bytes=100,
        )
        actions.append(decision)
    promoted = next(item for item in actions if item.action == "promote")
    assert promoted.block_key == "blocks.3"
    diagnostics = controller.diagnostics()
    assert diagnostics["last_block_key"] == "blocks.3"
    assert diagnostics["last_block_bytes"] == 20
    assert diagnostics["last_target_cap_bytes"] is None

    controller.step(
        clean,
        candidate=None,
        demote_candidate=None,
        cliff_cap_bytes=1000,
        worst_shape_free_bytes=0,
    )
    dirty = {**clean, "allocator": {"alloc_retries_delta": 1}}
    rollback = controller.step(
        dirty,
        candidate=None,
        demote_candidate=None,
        cliff_cap_bytes=1000,
        worst_shape_free_bytes=0,
    )
    assert rollback.action == "rollback"
    assert rollback.block_key == "blocks.3"


def test_controller_cold_starts_one_whole_block_below():
    controller = ArenaResidencyController()
    decision = controller.step(
        None,
        candidate={"block_key": "blocks.4", "block_bytes": 20},
        demote_candidate={"block_key": "blocks.2", "block_bytes": 30},
        cliff_cap_bytes=1000,
        worst_shape_free_bytes=100,
    )
    assert decision.action == "demote"
    assert decision.block_key == "blocks.2"
    assert decision.reason == "approach_from_below"


def test_controller_raises_cap_by_fixed_fsm_increment():
    controller = ArenaResidencyController(slack_pad_bytes=10)
    controller.bootstrapped = True
    controller.state = type(controller.state)("stable", 2)
    signal = {
        "allocator": {"alloc_retries_delta": 1},
        "peak_allocated_bytes": 400,
        "reclaimable_at_peak_bytes": 0,
        "compile_invalid": False,
        "transfer": None,
    }
    decision = controller.step(
        signal,
        candidate=None,
        demote_candidate={"block_key": "blocks.0", "block_bytes": 100},
        cliff_cap_bytes=1000,
        current_cap_bytes=700,
        worst_shape_free_bytes=100,
    )
    assert decision.action == "raise_cap"
    assert decision.target_cap_bytes == 710


def test_failed_training_step_does_not_publish_partial_peak(monkeypatch):
    runtime = ArenaOffloadRuntime.__new__(ArenaOffloadRuntime)
    runtime._closed = False
    runtime._last_shape_key = None
    runtime._last_step_num = None
    runtime._device = "cpu"
    runtime._apply_training_policy = lambda: None
    runtime._executor = SimpleNamespace(
        TRAIN="train",
        execution=lambda _mode: contextlib.nullcontext(),
    )
    observed = []
    runtime._observe_training_step = lambda **kwargs: observed.append(kwargs)

    with pytest.raises(RuntimeError, match="synthetic OOM"):
        with runtime.training_step(shape_key=(768, 768), step_num=2):
            raise RuntimeError("synthetic OOM")

    assert observed == []
