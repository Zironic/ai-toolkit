from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scripts import smoke_runtime


def test_contention_guard_rejects_cuda_above_thirty_percent(monkeypatch):
    monkeypatch.setattr(smoke_runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        smoke_runtime.torch.cuda,
        "mem_get_info",
        lambda device: (6 * smoke_runtime.GIB, 10 * smoke_runtime.GIB),
    )

    with pytest.raises(SystemExit, match=r"40\.0% VRAM.*--ignore-contention"):
        smoke_runtime.fail_if_vram_contended("cuda", ignore_contention=False)


def test_contention_guard_allows_threshold_and_explicit_override(monkeypatch):
    monkeypatch.setattr(smoke_runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        smoke_runtime.torch.cuda,
        "mem_get_info",
        lambda device: (7 * smoke_runtime.GIB, 10 * smoke_runtime.GIB),
    )
    smoke_runtime.fail_if_vram_contended("cuda", ignore_contention=False)

    monkeypatch.setattr(
        smoke_runtime.torch.cuda,
        "mem_get_info",
        lambda device: (1 * smoke_runtime.GIB, 10 * smoke_runtime.GIB),
    )
    smoke_runtime.fail_if_vram_contended("cuda", ignore_contention=True)


def test_contention_guard_is_noop_for_cpu(monkeypatch):
    monkeypatch.setattr(
        smoke_runtime.torch.cuda,
        "mem_get_info",
        lambda device: (_ for _ in ()).throw(AssertionError("unexpected CUDA query")),
    )
    smoke_runtime.fail_if_vram_contended("cpu", ignore_contention=False)


def test_cuda_smoke_disables_windows_inductor_cpu_isa_probe(monkeypatch):
    from torch._inductor import config

    monkeypatch.setattr(smoke_runtime.sys, "platform", "win32")
    monkeypatch.setattr(config.cpp, "vec_isa_ok", None)
    monkeypatch.setattr(smoke_runtime.torch._dynamo.config, "suppress_errors", True)

    smoke_runtime.configure_cuda_smoke_inductor()

    assert config.cpp.vec_isa_ok is False
    assert smoke_runtime.torch._dynamo.config.suppress_errors is False


def test_cuda_smoke_leaves_inductor_isa_policy_unchanged_off_windows(monkeypatch):
    from torch._inductor import config

    monkeypatch.setattr(smoke_runtime.sys, "platform", "linux")
    monkeypatch.setattr(config.cpp, "vec_isa_ok", None)
    monkeypatch.setattr(smoke_runtime.torch._dynamo.config, "suppress_errors", True)

    smoke_runtime.configure_cuda_smoke_inductor()

    assert config.cpp.vec_isa_ok is None
    assert smoke_runtime.torch._dynamo.config.suppress_errors is False


def test_smoke_load_mode_is_explicit_and_self_checking():
    model = SimpleNamespace(_prepared_canonical_build=None)
    smoke_runtime.configure_smoke_load_mode(
        model, smoke_runtime.PRODUCTION_LOAD_MODE
    )
    assert model._smoke_direct_arena_load is False
    smoke_runtime.assert_smoke_load_mode(model, smoke_runtime.PRODUCTION_LOAD_MODE)

    paging_mode = smoke_runtime.PAGING_LOAD_MODE
    smoke_runtime.configure_smoke_load_mode(model, paging_mode)
    assert model._smoke_direct_arena_load is False
    smoke_runtime.assert_smoke_load_mode(model, paging_mode)

    smoke_runtime.configure_smoke_load_mode(
        model, smoke_runtime.SMOKE_DIRECT_LOAD_MODE
    )
    model._prepared_canonical_build = object()
    assert model._smoke_direct_arena_load is True
    smoke_runtime.assert_smoke_load_mode(model, smoke_runtime.SMOKE_DIRECT_LOAD_MODE)

    with pytest.raises(RuntimeError, match="requested smoke load mode"):
        smoke_runtime.assert_smoke_load_mode(model, paging_mode)

    with pytest.raises(ValueError, match="unknown_smoke_load_mode"):
        smoke_runtime.configure_smoke_load_mode(model, "surprise")


def test_production_smoke_mode_uses_production_model_load_session(monkeypatch):
    events = []
    session = object()

    @contextmanager
    def fake_session(model):
        events.append(("enter", model))
        yield session
        events.append(("exit", model))

    monkeypatch.setattr(
        "toolkit.memory_management.arena_offload.model_load_arena_session",
        fake_session,
    )
    model = SimpleNamespace()

    with smoke_runtime.smoke_model_load_session(
        model, smoke_runtime.PRODUCTION_LOAD_MODE
    ) as active:
        assert active is session
        events.append(("body", model))

    assert events == [("enter", model), ("body", model), ("exit", model)]
    with smoke_runtime.smoke_model_load_session(
        model, smoke_runtime.SMOKE_DIRECT_LOAD_MODE
    ) as active:
        assert active is None
    with smoke_runtime.smoke_model_load_session(
        model, smoke_runtime.PAGING_LOAD_MODE
    ) as active:
        assert active is None
