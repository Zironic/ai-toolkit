import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from toolkit.memory_management import ingraph_stream


def _reset(depth=2):
    ingraph_stream.configure_fetch_runtime(depth=depth)
    ingraph_stream.fetch_stats(reset=True)
    torch._dynamo.reset()


def test_fetch_wait_shape_and_stats():
    _reset(depth=2)
    host = torch.arange(1024, dtype=torch.uint8).pin_memory()
    token = torch.ops.mm.fetch_start(host)
    flat = torch.ops.mm.fetch_wait(token, host.numel())
    torch.ops.mm.fetch_free(token)
    torch.cuda.synchronize()

    assert flat.device.type == "cuda"
    assert flat.dtype == torch.uint8
    assert flat.numel() == host.numel()
    assert torch.equal(flat.cpu(), host)
    stats = ingraph_stream.fetch_stats(reset=True)
    assert stats["fetches"] == 1
    assert stats["bytes"] == host.numel()


def test_fetch_start_rejects_pageable_source():
    _reset(depth=2)
    host = torch.empty(128, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="pinned"):
        torch.ops.mm.fetch_start(host)


def test_fetch_ops_compile_fullgraph():
    _reset(depth=2)
    host = torch.arange(2048, dtype=torch.uint8).pin_memory()

    def fn(h):
        token = torch.ops.mm.fetch_start(h)
        flat = torch.ops.mm.fetch_wait(token, h.numel())
        out = flat.to(torch.float32).sum()
        torch.ops.mm.fetch_free(token)
        return out

    compiled = torch.compile(fn, fullgraph=True, dynamic=False)
    out = compiled(host)
    torch.cuda.synchronize()
    assert out.item() == host.to(torch.float32).sum().item()
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0



def test_guarded_fetch_ops_order_depth_one_compile():
    _reset(depth=1)
    host_a = torch.arange(1024, dtype=torch.uint8).pin_memory()
    host_b = torch.arange(1024, dtype=torch.uint8).pin_memory()

    def fn(x, a, b):
        token = torch.ops.mm.fetch_start_after(a, x)
        flat = torch.ops.mm.fetch_wait(token, a.numel())
        y = x + flat[: x.numel()].to(device=x.device, dtype=x.dtype).reshape_as(x)
        torch.ops.mm.fetch_free_after(token, y)
        token = torch.ops.mm.fetch_start_after(b, y)
        flat = torch.ops.mm.fetch_wait(token, b.numel())
        y = y + flat[: y.numel()].to(device=y.device, dtype=y.dtype).reshape_as(y)
        torch.ops.mm.fetch_free_after(token, y)
        return y

    compiled = torch.compile(fn, fullgraph=True, dynamic=False)
    x = torch.zeros(16, device="cuda")
    out = compiled(x, host_a, host_b)
    torch.cuda.synchronize()

    expected = torch.arange(16, device="cuda", dtype=torch.float32) * 2
    torch.testing.assert_close(out, expected)
    stats = ingraph_stream.fetch_stats(reset=True)
    assert stats["fetches"] == 2
    assert stats["depth_waits"] == 0
def test_depth_guard_requires_free_before_overflow():
    _reset(depth=1)
    host = torch.empty(128, dtype=torch.uint8).pin_memory()
    token = torch.ops.mm.fetch_start(host)
    with pytest.raises(RuntimeError, match="depth exceeded"):
        torch.ops.mm.fetch_start(host)
    flat = torch.ops.mm.fetch_wait(token, host.numel())
    torch.ops.mm.fetch_free(token)
    torch.cuda.synchronize()
    assert flat.numel() == host.numel()
