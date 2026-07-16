from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from jobs.process.BaseSDTrainProcess import (
    BaseSDTrainProcess,
    _set_dynamo_cache_size_limit,
)
from toolkit.compile_shape_bounds import ObservedInputShape, SequenceLayout


def _process(*, dynamic, limit, shapes):
    transformer = SimpleNamespace(config=SimpleNamespace(patch=2))
    process = object.__new__(BaseSDTrainProcess)
    process.model_config = SimpleNamespace(
        compile=True,
        cache_size_limit=limit,
    )
    process.sd = SimpleNamespace(
        unet=transformer,
        get_compile_sequence_layout=lambda: SequenceLayout(
            sequence_alignment=1,
            includes_text=False,
        ),
    )
    process._observed_input_shapes = lambda _layout: shapes
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            compile_blocks=True,
            _compile_fullgraph=True,
            _compile_dynamic=dynamic,
        )
    )
    return process, runtime


def test_strict_static_arena_fails_before_exhausting_recompile_limit():
    shapes = [ObservedInputShape(16, width, 0) for width in range(2, 18, 2)]
    process, runtime = _process(dynamic=False, limit=8, shapes=shapes)

    with mock.patch(
        "jobs.process.BaseSDTrainProcess.get_memory_runtime",
        return_value=runtime,
    ), mock.patch(
        "jobs.process.BaseSDTrainProcess.unwrap_model",
        side_effect=lambda model: model,
    ), pytest.raises(RuntimeError, match="arena_fullgraph_static_recompile_limit"):
        process._validate_arena_fullgraph_static_recompile_budget()


def test_strict_static_arena_allows_an_explicit_adequate_limit():
    shapes = [ObservedInputShape(16, width, 0) for width in range(2, 18, 2)]
    process, runtime = _process(dynamic=False, limit=16, shapes=shapes)

    with mock.patch(
        "jobs.process.BaseSDTrainProcess.get_memory_runtime",
        return_value=runtime,
    ), mock.patch(
        "jobs.process.BaseSDTrainProcess.unwrap_model",
        side_effect=lambda model: model,
    ):
        process._validate_arena_fullgraph_static_recompile_budget()


def test_strict_dynamic_arena_does_not_enumerate_static_variants():
    process, runtime = _process(dynamic=True, limit=1, shapes=None)
    process._observed_input_shapes = mock.Mock(
        side_effect=AssertionError("dynamic mode must not enumerate static variants")
    )

    with mock.patch(
        "jobs.process.BaseSDTrainProcess.get_memory_runtime",
        return_value=runtime,
    ), mock.patch(
        "jobs.process.BaseSDTrainProcess.unwrap_model",
        side_effect=lambda model: model,
    ):
        process._validate_arena_fullgraph_static_recompile_budget()


def test_strict_static_arena_warns_when_shapes_are_unknown():
    process, runtime = _process(dynamic=False, limit=8, shapes=None)

    with mock.patch(
        "jobs.process.BaseSDTrainProcess.get_memory_runtime",
        return_value=runtime,
    ), mock.patch(
        "jobs.process.BaseSDTrainProcess.unwrap_model",
        side_effect=lambda model: model,
    ), mock.patch("jobs.process.BaseSDTrainProcess.print_acc") as printed:
        process._validate_arena_fullgraph_static_recompile_budget()

    assert "arena_fullgraph_static_shapes_unknown" in printed.call_args.args[0]


def test_explicit_cache_limit_updates_current_and_compatibility_torch_names():
    config = torch._dynamo.config
    saved = {
        name: getattr(config, name)
        for name in ("recompile_limit", "cache_size_limit")
        if hasattr(config, name)
    }
    try:
        _set_dynamo_cache_size_limit(17)
        assert all(getattr(config, name) == 17 for name in saved)
    finally:
        for name, value in saved.items():
            setattr(config, name, value)
