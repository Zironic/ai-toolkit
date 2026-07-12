from __future__ import annotations

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
