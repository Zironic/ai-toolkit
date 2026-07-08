"""Phase 3 S7 stability proofs (INGRAPH_PHASE3_SAMPLER_PLAN.md).

DoD 7: a repack (new host tensors, same values) and same-shape repeat calls
cause zero recompiles -- weights-as-inputs stays guard-free.
DoD 10: buffer-reuse hammer at depth=2 -- the ticket ring's free events must
prevent a device buffer from being recycled while the compute stream still
reads it (silent numeric corruption if violated; hit in the Phase 0 spike).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from toolkit.memory_management import ingraph_stream


def _reset(depth=2):
    ingraph_stream.configure_fetch_runtime(depth=depth)
    ingraph_stream.fetch_stats(reset=True)
    torch._dynamo.reset()


def _unique_graphs():
    return torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)


def _streamed_matmul_chain(x, hosts, n, k):
    """Fetch each host buffer, view it as a (k, k) float32 weight, matmul."""
    y = x
    for host in hosts:
        token = torch.ops.mm.fetch_start_after(host, y)
        flat = torch.ops.mm.fetch_wait(token, n)
        w = flat.view(torch.float32).reshape(k, k)
        y = y @ w
        torch.ops.mm.fetch_free_after(token, y)
    return y


def _heterogeneous_pack():
    torch.manual_seed(2)
    entries = (
        ("attn.wq", torch.randn(32, 32) * 0.05, None),
        ("attn.wk", torch.randn(16, 32) * 0.05, None),
        ("attn.wv", torch.randn(16, 32) * 0.05, None),
        ("attn.gate", torch.randn(32, 32) * 0.05, None),
        ("attn.wo", torch.randn(32, 32) * 0.05, None),
        ("mlp.gate", torch.randn(64, 32) * 0.05, None),
        ("mlp.up", torch.randn(64, 32) * 0.05, None),
        ("mlp.down", torch.randn(32, 64) * 0.05, None),
    )
    return ingraph_stream.pack_block_host("heterogeneous", entries, repoint=False, pin=True)


def _heterogeneous_streamed_block(x, host, nbytes, pack):
    token = torch.ops.mm.fetch_start_after(host, x)
    flat = torch.ops.mm.fetch_wait(token, nbytes)
    views = ingraph_stream.block_tensor_views(flat, pack)
    q = ingraph_stream.streamed_linear_tensors(
        x, views[0][0], views[0][1], views[0][2], fp8_qualifies=pack.fp8_flags[0]
    )
    gate = ingraph_stream.streamed_linear_tensors(
        x, views[5][0], views[5][1], views[5][2], fp8_qualifies=pack.fp8_flags[5]
    )
    up = ingraph_stream.streamed_linear_tensors(
        x, views[6][0], views[6][1], views[6][2], fp8_qualifies=pack.fp8_flags[6]
    )
    down = ingraph_stream.streamed_linear_tensors(
        torch.nn.functional.silu(gate) * up,
        views[7][0],
        views[7][1],
        views[7][2],
        fp8_qualifies=pack.fp8_flags[7],
    )
    y = q + down
    torch.ops.mm.fetch_free_after(token, y)
    return y


def test_heterogeneous_pack_views_cause_zero_recompiles():
    _reset(depth=2)
    pack = _heterogeneous_pack()
    compiled = torch.compile(_heterogeneous_streamed_block, fullgraph=True, dynamic=False)
    x = torch.randn(4, 32, device="cuda")

    first = compiled(x, pack.host_flat, int(pack.required_pin_bytes), pack)
    torch.cuda.synchronize()
    graphs = _unique_graphs()
    assert graphs >= 1
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0

    again = compiled(x, pack.host_flat, int(pack.required_pin_bytes), pack)
    torch.cuda.synchronize()
    torch.testing.assert_close(again, first)
    assert _unique_graphs() == graphs

    repacked_host = pack.host_flat.clone().pin_memory()
    third = compiled(x, repacked_host, int(pack.required_pin_bytes), pack)
    torch.cuda.synchronize()
    torch.testing.assert_close(third, first)
    assert _unique_graphs() == graphs
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0

def test_repack_and_repeat_cause_zero_recompiles():
    _reset(depth=2)
    k = 32
    n = k * k * 4  # float32 bytes
    torch.manual_seed(0)
    weights = [torch.randn(k, k) * 0.1 for _ in range(4)]
    hosts = [w.reshape(-1).view(torch.uint8).clone().pin_memory() for w in weights]

    compiled = torch.compile(_streamed_matmul_chain, fullgraph=True, dynamic=False)
    x = torch.randn(8, k, device="cuda")

    first = compiled(x, hosts, n, k)
    torch.cuda.synchronize()
    graphs = _unique_graphs()
    assert graphs >= 1
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0

    # Same-shape repeat: zero recompiles.
    again = compiled(x, hosts, n, k)
    torch.cuda.synchronize()
    torch.testing.assert_close(again, first)
    assert _unique_graphs() == graphs

    # Repack: brand-new host tensors (new storages), same values.
    repacked = [w.reshape(-1).view(torch.uint8).clone().pin_memory() for w in weights]
    third = compiled(x, repacked, n, k)
    torch.cuda.synchronize()
    torch.testing.assert_close(third, first)
    assert _unique_graphs() == graphs
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0


def test_buffer_reuse_hammer_depth_2():
    """Many passes over more blocks than the ring depth, with compute long
    enough that fetches for later blocks race the reads of earlier buffers.
    Any premature buffer recycle corrupts the integer checksums exactly."""
    _reset(depth=2)
    n_blocks = 8
    k = 256
    n = k * k * 4
    torch.manual_seed(1)
    # Signed permutation matrices: every intermediate is an exact signed
    # shuffle of the input, so results stay bitwise-exact through the whole
    # chain and corruption cannot hide inside float tolerance.
    weights = []
    gen = torch.Generator().manual_seed(1)
    for _ in range(n_blocks):
        perm = torch.randperm(k, generator=gen)
        sign = torch.randint(0, 2, (k,), generator=gen).to(torch.float32) * 2 - 1
        w = torch.zeros(k, k)
        w[torch.arange(k), perm] = sign
        weights.append(w)
    hosts = [w.reshape(-1).view(torch.uint8).clone().pin_memory() for w in weights]

    compiled = torch.compile(_streamed_matmul_chain, fullgraph=True, dynamic=False)
    x = torch.arange(4 * k, dtype=torch.float32, device="cuda").reshape(4, k) + 1

    with torch.no_grad():
        expected = x.cpu()
        for w in weights:
            expected = expected @ w
        expected = expected.to("cuda")

    passes = 50
    for _ in range(passes):
        out = compiled(x, hosts, n, k)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    stats = ingraph_stream.fetch_stats(reset=True)
    assert stats["fetches"] == n_blocks * passes
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
