from __future__ import annotations

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

    smoke_runtime.configure_cuda_smoke_inductor()

    assert config.cpp.vec_isa_ok is False


def test_cuda_smoke_leaves_inductor_isa_policy_unchanged_off_windows(monkeypatch):
    from torch._inductor import config

    monkeypatch.setattr(smoke_runtime.sys, "platform", "linux")
    monkeypatch.setattr(config.cpp, "vec_isa_ok", None)

    smoke_runtime.configure_cuda_smoke_inductor()

    assert config.cpp.vec_isa_ok is None


def test_smoke_load_mode_is_explicit_and_self_checking():
    model = SimpleNamespace(_prepared_canonical_build=None)
    smoke_runtime.configure_smoke_load_mode(model, "normal")
    assert model._smoke_direct_arena_load is False
    smoke_runtime.assert_smoke_load_mode(model, "normal")

    smoke_runtime.configure_smoke_load_mode(model, "direct-arena")
    model._prepared_canonical_build = object()
    assert model._smoke_direct_arena_load is True
    smoke_runtime.assert_smoke_load_mode(model, "direct-arena")

    with pytest.raises(RuntimeError, match="requested smoke load mode"):
        smoke_runtime.assert_smoke_load_mode(model, "normal")

    with pytest.raises(ValueError, match="unknown_smoke_load_mode"):
        smoke_runtime.configure_smoke_load_mode(model, "surprise")
