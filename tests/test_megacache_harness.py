from pathlib import Path

from scripts.bench_full_model_megacache_residency import (
    _command as residency_command,
    build_sequence,
)
from scripts.megacache_diagnostics import cache_evidence, numeric_delta
from scripts.run_full_model_megacache_matrix import (
    Arm,
    _command,
    _phase_parity_failures,
    build_arms,
)
from scripts.smoke_transformer_train_cuda import _validate_megacache_evidence


def test_numeric_delta_and_cache_evidence_distinguish_restoration():
    counters = {
        "aot_autograd.autograd_cache_hit": 4,
        "inductor.fxgraph_cache_hit": 5,
        "stats.unique_graphs": 3,
    }
    backend = numeric_delta(
        {"triton_compile": 7, "inductor_codegen": 2},
        {"triton_compile": 7, "inductor_codegen": 2},
    )
    evidence = cache_evidence(counters, backend)
    assert evidence["aot_hit"] == 4
    assert evidence["fx_hit"] == 5
    assert evidence["unique_graphs"] == 3
    assert evidence["backend_codegen_observed"] is False
    assert _validate_megacache_evidence("megacache", evidence, 3) == []


def test_megacache_validator_rejects_warm_backend_work():
    evidence = cache_evidence(
        {
            "aot_autograd.autograd_cache_hit": 3,
            "inductor.fxgraph_cache_hit": 3,
            "stats.unique_graphs": 3,
        },
        {"triton_compile": 1},
    )
    failures = _validate_megacache_evidence("shared-disk", evidence, 3)
    assert any("backend compiler work" in failure for failure in failures)


def test_megacache_validator_allows_only_measured_nonserializable_tail():
    evidence = cache_evidence(
        {
            "aot_autograd.autograd_cache_hit": 3,
            "aot_autograd.autograd_cache_miss": 1,
            "inductor.fxgraph_cache_hit": 5,
            "stats.unique_graphs": 3,
        },
        {"benchmark_all_configs": 1},
    )
    assert _validate_megacache_evidence("megacache", evidence, 3) == []

    evidence["aot_miss"] = 2
    evidence["autotune_benchmark_calls"] = 2
    failures = _validate_megacache_evidence("megacache", evidence, 3)
    assert any("non-serializable lookup" in failure for failure in failures)


def test_variant_count_is_unique_graph_count_not_cache_lookup_count():
    evidence = cache_evidence(
        {
            "aot_autograd.autograd_cache_hit": 6,
            "inductor.fxgraph_cache_hit": 7,
            "stats.unique_graphs": 4,
        },
        {},
    )
    assert _validate_megacache_evidence("megacache", evidence, 4) == []


def test_full_model_matrix_has_isolated_four_arm_layout(tmp_path):
    arms = build_arms(Path(tmp_path), "true", include_invalidation=True)
    assert [arm.expected for arm in arms] == [
        "cold",
        "empty-control",
        "shared-disk",
        "megacache",
        "invalidation",
    ]
    assert arms[0].cache_dir == arms[2].cache_dir
    assert len({arms[0].cache_dir, arms[1].cache_dir, arms[3].cache_dir}) == 3
    assert arms[-1].compile_dynamic == "false"


def test_matrix_freezes_residency_to_isolate_compiler_cache(tmp_path):
    args = type(
        "Args",
        (),
        {
            "smoke_args": ["--profile", "zimage"],
            "compile_coordinate_descent": "false",
            "expected_variants": 3,
        },
    )()
    arm = Arm("cold", "produce", "cold", Path(tmp_path) / "cache", "true")
    command = _command(
        args,
        arm,
        Path(tmp_path) / "artifact.bin",
        Path(tmp_path) / "result.json",
    )
    assert "--freeze-arena-residency" in command


def test_phase_parity_uses_field_specific_fp8_tolerances():
    cold = [
        {
            "index": 0,
            "phase": "train",
            "pred_checksum": "cold-pred",
            "pred_norm": 100.0,
            "pred_shape": [1, 2],
            "grad_checksum": "cold-grad",
            "grad_norm": 10.0,
            "grad_tensors": 4,
            "loss": 2.0,
        }
    ]
    close = [
        {
            **cold[0],
            "pred_checksum": "different-pred-bits",
            "grad_checksum": "different-grad-bits",
            "pred_norm": 100.1,
            "grad_norm": 10.2,
            "loss": 2.003,
        }
    ]
    assert _phase_parity_failures(cold, close, "megacache") == []

    far = [{**close[0], "grad_norm": 10.4}]
    failures = _phase_parity_failures(cold, far, "megacache")
    assert any("grad_norm exceeds" in failure for failure in failures)


def test_residency_benchmark_uses_cumulative_five_run_sequence(tmp_path):
    runs = build_sequence(Path(tmp_path))
    assert [run.transition for run in runs] == [
        "cold->mixed",
        "mixed->mixed",
        "mixed->full",
        "full->full",
        "full->mixed",
    ]
    assert runs[2].update_artifact is not None
    assert runs[3].input_artifact == runs[2].update_artifact
    assert runs[4].input_artifact == runs[2].update_artifact
    assert len({run.name for run in runs}) == 5


def test_residency_benchmark_forces_mixed_with_simulated_card(tmp_path):
    args = type(
        "Args",
        (),
        {
            "smoke_args": ["--profile", "zimage"],
            "compile_dynamic": "true",
            "compile_coordinate_descent": "false",
            "mixed_simulated_vram_gib": 10.0,
            "full_working_reserve_gib": 3.0,
        },
    )()
    runs = build_sequence(Path(tmp_path))
    mixed = residency_command(args, runs[1], Path(tmp_path) / "mixed.json")
    full = residency_command(args, runs[2], Path(tmp_path) / "full.json")
    assert mixed[mixed.index("--simulated-vram-gib") + 1] == "10.0"
    assert mixed[mixed.index("--working-reserve-gib") + 1] == "-1"
    assert mixed[mixed.index("--expected-residency") + 1] == "mixed"
    assert full[full.index("--simulated-vram-gib") + 1] == "0"
    assert full[full.index("--working-reserve-gib") + 1] == "3.0"
    assert full[full.index("--expected-residency") + 1] == "full"
    assert "--megacache-measure-only" in full
    assert "--megacache-update-artifact" in full
