import os
from types import SimpleNamespace

import pytest
import torch
import toolkit.compile_cache as compile_cache_module

from toolkit.compile_cache import (
    CompileCacheArtifact,
    CompileCacheSession,
    _recover_windows_triton_bundle_replace,
    compile_cache_artifact_counts,
    load_compile_cache,
    load_compile_cache_artifact,
    model_compile_cache_key,
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


def _model_config(**overrides):
    values = {
        "name_or_path": "checkpoint.safetensors",
        "arch": "synthetic",
        "qtype": "float8",
        "compile": True,
        "compile_sample": False,
        "compile_cache": True,
        "compile_cache_dir": None,
        "compile_mode": "default",
        "compile_fullgraph": True,
        "compile_dynamic": True,
        "compile_dynamic_hints": (),
        "compile_coordinate_descent": None,
        "layer_offloading_fp8_forward": True,
        "layer_offloading_fp8_grad_input": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_model_compile_cache_key_partitions_policy_but_not_residency():
    config = _model_config()
    model = SimpleNamespace(model_config=config)
    baseline = model_compile_cache_key(model)

    config.layer_offloading_simulated_vram_gb = 10
    config.layer_offloading_transformer_percent = 90
    assert model_compile_cache_key(model) == baseline

    config.compile_dynamic = False
    assert model_compile_cache_key(model) != baseline


def test_compile_cache_session_is_default_on_and_saves_only_after_new_frames(
    tmp_path, monkeypatch
):
    frames = {"count": 0}
    saved = []
    monkeypatch.setattr(
        compile_cache_module, "_dynamo_frame_count", lambda: frames["count"]
    )
    monkeypatch.setattr(
        compile_cache_module, "load_compile_cache_artifact", lambda _path: None
    )
    monkeypatch.setattr(
        compile_cache_module,
        "save_compile_cache_artifact",
        lambda path: saved.append(path)
        or CompileCacheArtifact(path=path, byte_count=4, info=None),
    )

    config = _model_config()
    model = SimpleNamespace(model_config=config)
    session = CompileCacheSession.for_model(
        model, default_cache_dir=tmp_path
    )
    assert session.enabled
    assert session.load() is None
    assert session.save() is None

    frames["count"] = 1
    assert session.save() is not None
    assert len(saved) == 1
    assert session.save() is None


def test_compile_cache_session_opt_out_and_failures_are_nonfatal(
    tmp_path, monkeypatch
):
    config = _model_config(compile_cache=False)
    disabled = CompileCacheSession.for_model(
        SimpleNamespace(model_config=config), default_cache_dir=tmp_path
    )
    assert not disabled.enabled

    messages = []
    session = CompileCacheSession(tmp_path, "broken", logger=messages.append)
    monkeypatch.setattr(
        compile_cache_module,
        "load_compile_cache_artifact",
        lambda _path: (_ for _ in ()).throw(RuntimeError("bad blob")),
    )
    monkeypatch.setattr(
        compile_cache_module,
        "save_compile_cache_artifact",
        lambda _path: (_ for _ in ()).throw(OSError("read only")),
    )

    assert session.load() is None
    assert session.save(force=True) is None
    assert any("cold compile" in message for message in messages)
    assert any("without persistence" in message for message in messages)
