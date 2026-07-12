import pytest

from scripts.digest_perf_log import summarize_record
from toolkit.memory_management import ingraph_stream
from toolkit.memory_management.ingraph_stream import fetch_performance_metrics


def test_fetch_performance_metrics_uses_device_time_and_decimal_bandwidth():
    metrics = fetch_performance_metrics(
        {"bytes": 12_000_000_000, "h2d_ms": 1_500.0, "wait_ms": 99_000.0},
        step_wall_ms=2_600.0,
    )

    assert metrics["h2d_duty_pct"] == pytest.approx(57.6923)
    assert metrics["achieved_gbps"] == pytest.approx(8.0)
    assert metrics["h2d_duty_overflow"] is False


def test_fetch_performance_metrics_flags_window_carryover():
    metrics = fetch_performance_metrics(
        {"bytes": 1_000_000_000, "h2d_ms": 1_100.0},
        step_wall_ms=1_000.0,
    )

    assert metrics["h2d_duty_pct"] == pytest.approx(110.0)
    assert metrics["h2d_duty_overflow"] is True


def test_fetch_performance_metrics_handles_missing_denominators():
    metrics = fetch_performance_metrics(
        {"bytes": 0, "h2d_ms": 0.0, "wait_ms": 123.0},
        step_wall_ms=None,
    )

    assert metrics["h2d_duty_pct"] is None
    assert metrics["achieved_gbps"] is None
    assert metrics["h2d_duty_overflow"] is False


def test_fetch_report_emits_raw_and_derived_window_metrics(monkeypatch):
    monkeypatch.setattr(
        ingraph_stream,
        "fetch_stats",
        lambda reset=False: {
            "fetches": 4,
            "copies": 8,
            "bytes": 12_000_000_000,
            "h2d_ms": 1_500.0,
            "wait_ms": 321.0,
            "depth_waits": 2,
        },
    )
    report = ingraph_stream.fetch_report(reset=True, step_wall_ms=2_600.0)
    assert "h2d_ms=1500.000" in report
    assert "step_wall_ms=2600.000" in report
    assert "h2d_duty_pct=57.7" in report
    assert "h2d_duty_overflow=0" in report
    assert "achieved_gbps=8.00" in report
    assert "wait_ms=321.000" in report


def test_digest_surfaces_duty_and_marks_host_wait_diagnostic_only():
    lines = summarize_record(
        {
            "step": 10,
            "window_steps": 4,
            "entire_training_step_s": 0.65,
            "ingraph_stream": (
                "[InGraphStream] fetches=4 copies=8 bytes=11.18 GiB "
                "h2d_ms=1500.000 step_wall_ms=2600.000 "
                "h2d_duty_pct=57.7 h2d_duty_overflow=0 "
                "achieved_gbps=8.00 wait_ms=321.000 depth_waits=2"
            ),
        },
        full=False,
    )
    joined = "\n".join(lines)
    assert "duty=57.7%" in joined
    assert "achieved=8.00GB/s" in joined
    assert "host_wait_ms=321.000 (diagnostic only)" in joined
