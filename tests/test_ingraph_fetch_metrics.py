import pytest

from scripts.digest_perf_log import summarize_record, summarize_records
from toolkit.memory_management import ingraph_stream
from toolkit.memory_management.ingraph_stream import fetch_performance_metrics


def test_lifetime_fetch_stats_survive_report_reset(monkeypatch):
    window = {
        "fetches": 2,
        "bytes": 100,
        "copies": 2,
        "h2d_ms": 5.0,
        "wait_ms": 1.0,
        "depth_waits": 0,
    }
    lifetime = dict(window)
    monkeypatch.setattr(ingraph_stream, "_STATS", window)
    monkeypatch.setattr(ingraph_stream, "_LIFETIME_STATS", lifetime)

    ingraph_stream.reset_fetch_stats()

    assert ingraph_stream.fetch_stats() == {
        key: 0 for key in window
    }
    assert ingraph_stream.lifetime_fetch_stats() == lifetime


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


def test_digest_surfaces_arena_policy_window_and_run_summary():
    arena = {
        "resident_bytes": 2 * 1024 ** 3,
        "singleton_resident_bytes": 512 * 1024 ** 2,
        "canonical_resident_bytes": 1536 * 1024 ** 2,
        "plan_fingerprint": "abc123",
        "policy_error": None,
        "policy": {
            "last_signal": {
                "allocator": {"alloc_retries_delta": 0},
                "transfer": {
                    "h2d_duty_pct": 75.0,
                    "achieved_gbps": 9.5,
                },
            },
            "controller": {
                "state": "promotion_verify",
                "windows_in_state": 1,
                "last_action": "promote",
                "last_reason": "safe_transfer_benefit",
                "last_block_key": "blocks.3",
                "last_block_bytes": 512 * 1024 ** 2,
                "last_target_cap_bytes": None,
                "last_worst_shape_margin_bytes": 1024 ** 3,
                "last_throughput_gate": True,
                "last_promote_gate": True,
                "last_cap_covers_promo": True,
                "slack_pad_bytes": 256 * 1024 ** 2,
            },
        },
    }
    record = {
        "step": 10,
        "window_steps": 1,
        "entire_training_step_s": 1.0,
        "arena_offload": arena,
    }

    window = "\n".join(summarize_record(record, full=False))
    assert "arena_policy: state=promotion_verify" in window
    assert "action=promote" in window
    assert "block=blocks.3" in window
    assert "duty=75.0%" in window
    assert "achieved=9.50GB/s" in window
    assert "resident_gib=2.00 singleton_gib=0.50 canonical_gib=1.50" in window
    assert "worst_margin_gib=1.00" in window
    assert "allocator_slack_gib=- headband_gib=0.25" in window
    assert "throughput_gate=True promote_gate=True cap_covers=True" in window

    summary = "\n".join(summarize_records([record]))
    assert "Arena policy: windows=1 state=promotion_verify" in summary
    assert "resident=2.00GiB (singleton=0.50 canonical=1.50)" in summary
    assert "layout_actions=1" in summary
    assert "policy_errors=0" in summary
    assert "actions=[promote=1]" in summary
