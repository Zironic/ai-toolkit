import os
from types import SimpleNamespace

import pytest
import torch

from toolkit.compile_cache import (
    _recover_windows_triton_bundle_replace,
    compile_cache_artifact_counts,
    load_compile_cache,
    load_compile_cache_artifact,
    save_compile_cache,
    save_compile_cache_artifact,
)


def test_compile_cache_artifact_round_trip_preserves_inventory(tmp_path, monkeypatch):
    saved_info = SimpleNamespace(
        artifacts={"inductor": ("fx-a", "fx-b"), "aot_autograd": ("aot-a",)}
    )
    loaded_info = SimpleNamespace(artifacts=saved_info.artifacts)
    blob = b"serialized-compiler-cache"
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (blob, saved_info)
    )
    monkeypatch.setattr(
        torch.compiler,
        "load_cache_artifacts",
        lambda value: loaded_info if value == blob else None,
    )

    path = tmp_path / "cache.bin"
    saved = save_compile_cache_artifact(path)
    assert saved is not None
    assert saved.byte_count == len(blob)
    assert saved.artifact_counts == {"aot_autograd": 1, "inductor": 2}
    assert path.read_bytes() == blob

    loaded = load_compile_cache_artifact(path)
    assert loaded is not None
    assert loaded.byte_count == len(blob)
    assert loaded.artifact_counts == saved.artifact_counts


def test_compile_cache_legacy_boolean_wrappers(tmp_path, monkeypatch):
    info = SimpleNamespace(artifacts={"inductor": ("fx",)})
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (b"blob", info)
    )
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", lambda value: info)

    assert save_compile_cache(str(tmp_path), "model/key") is True
    assert load_compile_cache(str(tmp_path), "model/key") is True
    assert load_compile_cache(str(tmp_path), "absent") is False


def test_compile_cache_artifact_counts_handles_no_inventory():
    assert compile_cache_artifact_counts(None) == {}


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific Triton fallback")
def test_windows_triton_bundle_rename_fallback_copies_materialized_files(tmp_path):
    parent = tmp_path / "triton" / "0"
    source = parent / "tmp.probe"
    destination = parent / "HASH"
    source.mkdir(parents=True)
    (source / "kernel.cubin").write_bytes(b"kernel")
    error = SimpleNamespace(
        winerror=5,
        filename=str(source),
        filename2=str(destination),
    )

    assert _recover_windows_triton_bundle_replace(error) is True
    assert (destination / "kernel.cubin").read_bytes() == b"kernel"
    assert not source.exists()


def test_windows_triton_bundle_fallback_rejects_unrelated_permission_error(tmp_path):
    error = SimpleNamespace(
        winerror=13,
        filename=str(tmp_path / "tmp.probe"),
        filename2=str(tmp_path / "HASH"),
    )
    assert _recover_windows_triton_bundle_replace(error) is False
