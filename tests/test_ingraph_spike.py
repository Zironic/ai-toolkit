"""Phase 0 spike for tasks/done/INGRAPH_STREAM_PLAN.md (git-bug 3ca8a7b).

Proves, on the real GPU, the machinery the in-graph streaming design rests on:

1. Split-phase custom fetch ops (fetch_start -> token -> fetch_wait) trace
   under torch.compile(fullgraph=True) with zero graph breaks, with the
   flat-buffer -> view slicing done in-graph.
2. SAC (selective activation checkpoint) with MUST_RECOMPUTE on the fetch
   ops makes the compiled backward RE-FETCH weights instead of saving them.
3. Numerics parity vs the same model with resident weights, fwd and bwd.
4. H2D of block i+1 overlaps block i's GEMM with source-level K-ahead
   pipelining (event-timed).

Throwaway op namespace mm_spike:: -- the production ops live in
toolkit/memory_management/ingraph_stream.py once Phase 2 lands.
"""

import functools
import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEVICE = torch.device("cuda")
DIM = 2048
N_BLOCKS = 6
NBYTES = DIM * DIM * 2  # bf16

# ---------------------------------------------------------------- runtime --

_TICKETS: dict = {}
_STATE = {
    "next_id": 0,
    "transfer_stream": None,
    "real_fetches": 0,
    # optional event instrumentation for the overlap test
    "trace_events": None,  # list of (kind, start_ev, end_ev) or None
}


def _transfer_stream():
    if _STATE["transfer_stream"] is None:
        _STATE["transfer_stream"] = torch.cuda.Stream()
    return _STATE["transfer_stream"]


@torch.library.custom_op("mm_spike::fetch_start", mutates_args=())
def fetch_start(host_flat: torch.Tensor) -> torch.Tensor:
    ts = _transfer_stream()
    _STATE["real_fetches"] += 1
    with torch.cuda.stream(ts):
        if _STATE["trace_events"] is not None:
            s = torch.cuda.Event(enable_timing=True)
            s.record(ts)
        buf = host_flat.to(DEVICE, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(ts)
        if _STATE["trace_events"] is not None:
            e = torch.cuda.Event(enable_timing=True)
            e.record(ts)
            _STATE["trace_events"].append(("h2d", s, e))
    tid = _STATE["next_id"]
    _STATE["next_id"] += 1
    _TICKETS[tid] = (buf, ev)
    return torch.tensor([tid], dtype=torch.int64)


@fetch_start.register_fake
def _(host_flat):
    return torch.empty(1, dtype=torch.int64, device="cpu")


@torch.library.custom_op("mm_spike::fetch_wait", mutates_args=())
def fetch_wait(token: torch.Tensor, nbytes: int) -> torch.Tensor:
    tid = int(token[0].item())
    buf, ev = _TICKETS.pop(tid)
    cs = torch.cuda.current_stream()
    cs.wait_event(ev)
    # buf was allocated on the transfer stream; without this, the allocator
    # may recycle it for a later fetch's H2D while the compute stream is
    # still reading it (use-after-free race across streams). The production
    # runtime bounds reuse with explicit ring free-events instead.
    buf.record_stream(cs)
    return buf


@fetch_wait.register_fake
def _(token, nbytes):
    return torch.empty(nbytes, dtype=torch.uint8, device=DEVICE)


# ------------------------------------------------------------------ model --


def make_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    weights = [
        (torch.randn(DIM, DIM, generator=g, dtype=torch.float32) * 0.02).to(
            torch.bfloat16
        )
        for _ in range(N_BLOCKS)
    ]
    flats = [
        w.contiguous().view(torch.uint8).flatten().pin_memory() for w in weights
    ]
    resident = [w.to(DEVICE) for w in weights]
    return flats, resident


def block_streamed(x, host_flat):
    tok = torch.ops.mm_spike.fetch_start(host_flat)
    flat = torch.ops.mm_spike.fetch_wait(tok, NBYTES)
    w = flat.view(torch.bfloat16).reshape(DIM, DIM)  # in-graph view/reshape
    return F.silu(x @ w.t())


def model_streamed(x, flats):
    for hf in flats:
        x = block_streamed(x, hf)
    return x


def model_resident(x, ws):
    for w in ws:
        x = F.silu(x @ w.t())
    return x


def model_streamed_pipelined(x, flats, k=2):
    """Source-level K-ahead prefetch: fetch_start(i+k) issued while block i
    computes. Dynamo unrolls the loop so the schedule is trace-visible."""
    tokens = [torch.ops.mm_spike.fetch_start(flats[i]) for i in range(min(k, len(flats)))]
    for i in range(len(flats)):
        flat = torch.ops.mm_spike.fetch_wait(tokens[i], NBYTES)
        if i + k < len(flats):
            tokens.append(torch.ops.mm_spike.fetch_start(flats[i + k]))
        w = flat.view(torch.bfloat16).reshape(DIM, DIM)
        x = F.silu(x @ w.t())
    return x


def _sac_context_fn():
    from torch.utils.checkpoint import (
        CheckpointPolicy,
        create_selective_checkpoint_contexts,
    )

    def policy(ctx, op, *args, **kwargs):
        if op in (
            torch.ops.mm_spike.fetch_start.default,
            torch.ops.mm_spike.fetch_wait.default,
        ):
            return CheckpointPolicy.MUST_RECOMPUTE
        return CheckpointPolicy.PREFER_SAVE

    return create_selective_checkpoint_contexts(policy)


def model_streamed_sac(x, flats):
    from torch.utils.checkpoint import checkpoint

    for hf in flats:
        x = checkpoint(
            block_streamed, x, hf, use_reentrant=False, context_fn=_sac_context_fn
        )
    return x


def _reset():
    _TICKETS.clear()
    _STATE["real_fetches"] = 0
    _STATE["trace_events"] = None
    torch._dynamo.reset()


# ------------------------------------------------------------------ tests --


def test_fullgraph_zero_breaks_and_parity():
    _reset()
    flats, resident = make_weights()
    x = torch.randn(64, DIM, device=DEVICE, dtype=torch.bfloat16)

    compiled = torch.compile(model_streamed, fullgraph=True, dynamic=False)
    compiled_ref = torch.compile(model_resident, fullgraph=True, dynamic=False)
    with torch.no_grad():
        out = compiled(x, flats)
        ref = compiled_ref(x, resident)
    torch.cuda.synchronize()
    # Same compiled kernels either side; only the weight transport differs.
    assert torch.equal(out, ref), "streamed fetch must be bitwise vs compiled resident"
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    assert not _TICKETS, "all tickets consumed"


def test_sac_backward_refetches_and_grad_parity():
    _reset()
    flats, resident = make_weights()
    x = torch.randn(64, DIM, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    compiled = torch.compile(model_streamed_sac, fullgraph=True, dynamic=False)
    out = compiled(x, flats)
    _STATE["real_fetches"] = 0  # count one steady-state step cleanly
    out2 = compiled(x, flats)
    out2.sum().backward()
    torch.cuda.synchronize()
    fetches = _STATE["real_fetches"]

    compiled_ref = torch.compile(model_resident, fullgraph=True, dynamic=False)
    ref = compiled_ref(x_ref, resident)
    ref.sum().backward()

    # Compiled reference: same kernels, so parity is tight; SAC recompute can
    # still reorder epilogues, so allow bf16-noise tolerance on the grad.
    torch.testing.assert_close(out2, ref, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-2, atol=1e-3)
    # forward fetch + backward re-fetch per block => 2 * N_BLOCKS real fetches
    assert fetches == 2 * N_BLOCKS, (
        f"expected backward to re-fetch (2x{N_BLOCKS}), saw {fetches} -- "
        "partitioner did not honor MUST_RECOMPUTE on the fetch ops"
    )
    assert sum(torch._dynamo.utils.counters["graph_break"].values()) == 0
    assert not _TICKETS


def test_sac_does_not_save_weights_vram():
    """With MUST_RECOMPUTE fetches, forward must not retain all N weight
    buffers for backward. Allow resident set ~= ring depth, not N blocks."""
    _reset()
    flats, _ = make_weights()
    x = torch.randn(64, DIM, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(model_streamed_sac, fullgraph=True, dynamic=False)
    compiled(x, flats).sum().backward()  # warmup/compile
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    out = compiled(x, flats)
    held = torch.cuda.memory_allocated() - base  # what forward keeps alive
    out.sum().backward()
    torch.cuda.synchronize()
    weight_bytes = NBYTES * N_BLOCKS
    assert held < weight_bytes * 0.7, (
        f"forward holds {held/2**20:.0f} MiB between fwd and bwd; "
        f"all-weights-saved would be ~{weight_bytes/2**20:.0f} MiB"
    )


def _time_ms(fn, iters=10):
    fn()  # warmup
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def test_streaming_overlaps_transfer_with_compute():
    """Overlap proof: streamed wall time must be well under the sum of
    compute-only and transfer-only time. Note the plain (non-pipelined)
    streamed model already overlaps via host run-ahead (the plan's Tier 0):
    the host enqueues fetch i+1 while the GPU still runs block i's GEMM.
    K-ahead pipelining (Tier 1) is measured alongside for the record."""
    _reset()
    flats, resident = make_weights()
    x = torch.randn(2048, DIM, device=DEVICE, dtype=torch.bfloat16)

    compiled = torch.compile(model_streamed, fullgraph=True, dynamic=False)
    compiled_pipe = torch.compile(
        functools.partial(model_streamed_pipelined, k=2),
        fullgraph=True,
        dynamic=False,
    )
    compiled_ref = torch.compile(model_resident, fullgraph=True, dynamic=False)

    with torch.no_grad():
        # parity for the pipelined variant (plain variant covered above)
        assert torch.equal(compiled_pipe(x, flats), compiled_ref(x, resident))
        torch.cuda.synchronize()

        def transfer_only():
            toks = [torch.ops.mm_spike.fetch_start(hf) for hf in flats]
            for t in toks:
                torch.ops.mm_spike.fetch_wait(t, NBYTES)

        compute_ms = _time_ms(lambda: compiled_ref(x, resident))
        transfer_ms = _time_ms(transfer_only)
        streamed_ms = _time_ms(lambda: compiled(x, flats))
        pipelined_ms = _time_ms(lambda: compiled_pipe(x, flats))

    parts = compute_ms + transfer_ms
    best = min(streamed_ms, pipelined_ms)
    print(
        f"\n[spike overlap] compute={compute_ms:.2f}ms transfer={transfer_ms:.2f}ms "
        f"sum={parts:.2f}ms streamed={streamed_ms:.2f}ms "
        f"pipelined(k=2)={pipelined_ms:.2f}ms blocks={N_BLOCKS}"
    )
    assert best < parts * 0.9, (
        f"no overlap: streamed {best:.2f}ms vs compute+transfer {parts:.2f}ms"
    )
