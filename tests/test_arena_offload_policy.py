import json

import pytest

from toolkit.memory_management.arena_offload.policy import TrainingSignalWindow


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


def test_signal_contains_memory_accounting():
    signal = observe(TrainingSignalWindow())
    assert signal["reclaimable_at_peak_bytes"] == 40
    assert signal["device_free_bytes"] == 500
    assert signal["resident_bytes"] == 200
    assert signal["ring_bytes"] == 80
    assert signal["live_bytes"] == 280
    json.dumps(TrainingSignalWindow().diagnostics())
