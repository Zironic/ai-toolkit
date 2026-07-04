"""Tests for toolkit.async_save -- the off-thread, crash-atomic checkpoint writer.

Pure CPU/IO; no GPU required.
"""

import os
import threading
import time

import pytest
import torch

from toolkit.async_save import (
    AsyncSaver,
    PinnedStager,
    atomic_save_file,
    atomic_torch_save,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_pinned_stager_matches_per_param_loop():
    torch.manual_seed(0)
    sd = {}
    for i in range(200):
        sd[f"down_{i}"] = torch.randn(16, 512, device="cuda")
        sd[f"up_{i}"] = torch.randn(512, 16, device="cuda")
    ref = {k: v.detach().clone().to("cpu").to(torch.float16) for k, v in sd.items()}
    st = PinnedStager(cap_bytes=1 * 1024 * 1024)  # small cap -> many chunks
    got = st.snapshot(list(sd.items()), out_dtype=torch.float16)
    st.close()
    assert set(got) == set(ref)
    for k in ref:
        assert got[k].dtype == torch.float16 and got[k].shape == ref[k].shape
        assert torch.equal(got[k], ref[k])


@cuda
def test_pinned_stager_handles_tensor_larger_than_buffer():
    # A single tensor exceeding the whole buffer must still snapshot correctly.
    sd = {"big": torch.randn(4096, 4096, device="cuda")}  # 64 MB fp32
    st = PinnedStager(cap_bytes=1 * 1024 * 1024)  # 1 MB buffer
    got = st.snapshot(list(sd.items()), out_dtype=torch.float16)
    st.close()
    assert torch.equal(got["big"], sd["big"].to("cpu").to(torch.float16))


def test_atomic_torch_save_leaves_no_tmp_and_roundtrips(tmp_path):
    path = str(tmp_path / "opt.pt")
    obj = {"a": torch.arange(10), "step": 5}
    atomic_torch_save(obj, path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")
    loaded = torch.load(path)
    assert loaded["step"] == 5
    assert torch.equal(loaded["a"], torch.arange(10))


def test_atomic_save_file_roundtrips(tmp_path):
    path = str(tmp_path / "lora.safetensors")
    sd = {"w": torch.ones(4, 4), "b": torch.zeros(4)}
    atomic_save_file(sd, path, metadata={"step": "7"})
    assert not os.path.exists(path + ".tmp")
    from safetensors.torch import load_file, safe_open

    loaded = load_file(path)
    assert torch.equal(loaded["w"], torch.ones(4, 4))
    with safe_open(path, framework="pt") as f:
        assert f.metadata()["step"] == "7"


def test_atomic_replace_overwrites_existing(tmp_path):
    path = str(tmp_path / "lora.safetensors")
    atomic_save_file({"w": torch.ones(2)}, path)
    atomic_save_file({"w": torch.full((2,), 3.0)}, path)
    from safetensors.torch import load_file

    assert torch.equal(load_file(path)["w"], torch.full((2,), 3.0))


def test_async_saver_writes_all_distinct_jobs(tmp_path):
    saver = AsyncSaver()
    try:
        for i in range(8):
            p = str(tmp_path / f"ckpt_{i}.pt")
            saver.submit(lambda p=p, i=i: atomic_torch_save({"i": i}, p),
                         description=f"ckpt_{i}")
        assert saver.wait_idle(timeout=10)
    finally:
        saver.close(timeout=10)
    for i in range(8):
        assert torch.load(str(tmp_path / f"ckpt_{i}.pt"))["i"] == i


def test_async_saver_coalesces_latest_wins(tmp_path):
    # Gate the worker so several coalesced jobs pile up before any runs; only
    # the newest should actually hit disk.
    release = threading.Event()
    ran = []
    saver = AsyncSaver()
    try:
        saver.submit(lambda: release.wait(5), description="gate")
        for i in range(5):
            saver.submit(
                lambda i=i: ran.append(i),
                coalesce_key="recovery",
                description=f"snap_{i}",
            )
        release.set()
        assert saver.wait_idle(timeout=10)
    finally:
        saver.close(timeout=10)
    # Earlier coalesced snapshots were superseded; only the last ran.
    assert ran == [4]


def test_async_saver_surfaces_write_error(tmp_path):
    saver = AsyncSaver()

    def boom():
        raise IOError("disk full")

    try:
        saver.submit(boom, description="boom")
        with pytest.raises(RuntimeError, match="disk full"):
            saver.wait_idle(timeout=10)
    finally:
        saver.close(timeout=10)


def test_async_saver_snapshot_is_isolated_from_mutation(tmp_path):
    # Mimic the training-thread contract: snapshot (clone) before handing off,
    # then keep mutating the "live" tensor. The written file must hold the
    # snapshot, not the later mutation.
    live = torch.ones(4)
    snapshot = live.clone()
    path = str(tmp_path / "snap.pt")
    saver = AsyncSaver()
    try:
        saver.submit(lambda: atomic_torch_save({"w": snapshot}, path))
        live.add_(100.0)  # training moves on
        assert saver.wait_idle(timeout=10)
    finally:
        saver.close(timeout=10)
    assert torch.equal(torch.load(path)["w"], torch.ones(4))
