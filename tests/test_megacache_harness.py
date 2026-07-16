from pathlib import Path

from scripts.megacache_diagnostics import cache_evidence, numeric_delta
from scripts.run_full_model_megacache_matrix import build_arms
from scripts.smoke_transformer_train_cuda import _validate_megacache_evidence


def test_numeric_delta_and_cache_evidence_distinguish_restoration():
    counters = {
        "aot_autograd.autograd_cache_hit": 3,
        "inductor.fxgraph_cache_hit": 3,
    }
    backend = numeric_delta(
        {"triton_compile": 7, "inductor_codegen": 2},
        {"triton_compile": 7, "inductor_codegen": 2},
    )
    evidence = cache_evidence(counters, backend)
    assert evidence["aot_hit"] == 3
    assert evidence["fx_hit"] == 3
    assert evidence["backend_codegen_observed"] is False
    assert _validate_megacache_evidence("megacache", evidence, 3) == []


def test_megacache_validator_rejects_warm_backend_work():
    evidence = cache_evidence(
        {
            "aot_autograd.autograd_cache_hit": 3,
            "inductor.fxgraph_cache_hit": 3,
        },
        {"triton_compile": 1},
    )
    failures = _validate_megacache_evidence("shared-disk", evidence, 3)
    assert any("compiler work" in failure for failure in failures)


def test_variant_count_is_aot_graph_count_not_fx_artifact_count():
    evidence = cache_evidence(
        {
            "aot_autograd.autograd_cache_hit": 1,
            "inductor.fxgraph_cache_hit": 2,
        },
        {},
    )
    assert _validate_megacache_evidence("megacache", evidence, 1) == []


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
