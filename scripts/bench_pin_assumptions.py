"""Pin-management assumption benchmarks (evidence for the canonical-arena plan).

Measures the platform facts that tasks/open/IMMUTABLE_TRANSFER_ARENA_PLAN.md
is built on. Re-run after torch/CUDA/driver upgrades; test 1 is the canary for
torch's is_pinned() semantics, tests 13-15 for Dynamo guard behavior.

Run through the project venv on an idle GPU (no training job):

    venv/Scripts/python.exe scripts/bench_pin_assumptions.py
    venv/Scripts/python.exe scripts/bench_pin_assumptions.py --tests 1,5,13,14
    venv/Scripts/python.exe scripts/bench_pin_assumptions.py --list

Reference results (2026-07-10, RTX 4070, torch 2.12.0+cu132, Win11 WDDM,
32 GiB RAM) are recorded in the plan document.

Note: test 12 loads cudart via ctypes, which can segfault at interpreter exit
(after all output) when combined with torch's runtime. Harmless for a bench
process; it is why test 12 runs last and is excluded from --tests default on
CI-like usage.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.cuda import _pin_memory_utils as pmu

MB = 1024 ** 2
GIB = 1024 ** 3
PAGE = 4096


def dxgi_usage():
    try:
        from toolkit.memory_management import dxgi_meminfo
    except Exception:
        return None
    info = dxgi_meminfo.query_non_local_video_memory_info(
        cuda_device_index=0, min_interval_s=0.0
    )
    return None if info is None else int(info.current_usage_bytes)


def aligned_range(nbytes):
    """Pageable buffer with an exclusive page-aligned interior (pin_register
    style: a full slack page on each side so no neighbor shares our pages)."""
    padded = (nbytes + PAGE - 1) // PAGE * PAGE
    base = torch.empty(padded + 3 * PAGE, dtype=torch.uint8)
    ptr = base.data_ptr()
    start = ((ptr + PAGE + PAGE - 1) // PAGE) * PAGE
    off = start - ptr
    return base, base[off:off + padded], padded


def timed(label, fn, n=3):
    vals = [fn() for _ in range(n)]
    print(f"  {label:<58} best={min(vals) * 1000:9.1f} ms  "
          f"({[f'{v * 1000:.0f}' for v in vals]})")
    return min(vals)


def header(num, title):
    print("=" * 78)
    print(f"TEST {num}: {title}")
    print("=" * 78)


# ---------------------------------------------------------------------------

def test_1():
    header(1, "is_pinned() semantics for cudaHostRegister'd memory")
    _base, view, padded = aligned_range(64 * MB)  # _base kept alive: owns the pages
    pmu.pin_memory(view.data_ptr(), padded)
    print(f"  register-pinned view .is_pinned()      = {view.is_pinned()}")
    print(f"  interior sub-view .is_pinned()         = {view[1024:2048].is_pinned()}")
    alloc_pinned = torch.empty(1 * MB, dtype=torch.uint8, pin_memory=True)
    print(f"  cudaHostAlloc tensor .is_pinned()      = {alloc_pinned.is_pinned()}")
    pmu.unpin_memory(view.data_ptr())


def test_2():
    header(2, "pin/registration speed, 600 MiB (order & mechanism matrix)")
    N = 600 * MB

    def t_hostalloc():
        torch._C._host_emptyCache()
        t0 = time.perf_counter()
        t = torch.empty(N, dtype=torch.uint8, pin_memory=True)
        dt = time.perf_counter() - t0
        del t
        return dt

    timed("cudaHostAlloc (pin_memory=True, cache emptied first)", t_hostalloc)
    torch._C._host_emptyCache()

    def t_register(populate_first):
        _base, view, padded = aligned_range(N)
        if populate_first:
            view.fill_(7)
        t0 = time.perf_counter()
        pmu.pin_memory(view.data_ptr(), padded)
        dt = time.perf_counter() - t0
        pmu.unpin_memory(view.data_ptr())
        return dt

    timed("cudaHostRegister, pages UNTOUCHED (demand-zero)",
          lambda: t_register(False))
    timed("cudaHostRegister, pages PRE-POPULATED",
          lambda: t_register(True))

    def t_populate():
        _base, view, _ = aligned_range(N)
        t0 = time.perf_counter()
        view.fill_(7)
        return time.perf_counter() - t0

    timed("populate only (fill_ on fresh pageable buffer)", t_populate)

    def t_unpin():
        _base, view, padded = aligned_range(N)
        pmu.pin_memory(view.data_ptr(), padded)
        t0 = time.perf_counter()
        pmu.unpin_memory(view.data_ptr())
        return time.perf_counter() - t0

    timed("cudaHostUnregister 600 MiB", t_unpin)


def test_3():
    header(3, "registration granularity + thread parallelism (1.2 GiB total)")
    TOT = 1200 * MB

    def t_chunks(nchunks, threads):
        bufs = []
        for _ in range(nchunks):
            b, v, p = aligned_range(TOT // nchunks)
            v.fill_(1)
            bufs.append((b, v, p))
        if threads:
            workers = [
                threading.Thread(target=pmu.pin_memory, args=(v.data_ptr(), p))
                for _, v, p in bufs
            ]
            t0 = time.perf_counter()
            for w in workers:
                w.start()
            for w in workers:
                w.join()
            dt = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            for _, v, p in bufs:
                pmu.pin_memory(v.data_ptr(), p)
            dt = time.perf_counter() - t0
        for _, v, _ in bufs:
            pmu.unpin_memory(v.data_ptr())
        return dt

    timed("register 1 x 1200 MiB serial (populated)", lambda: t_chunks(1, False))
    timed("register 8 x 150 MiB serial (populated)", lambda: t_chunks(8, False))
    timed("register 2 threads x 600 MiB (populated)", lambda: t_chunks(2, True))
    timed("register 4 threads x 300 MiB (populated)", lambda: t_chunks(4, True))
    timed("register 8 threads x 150 MiB (populated)", lambda: t_chunks(8, True))


def test_4():
    header(4, "H2D transfer semantics by source kind (600 MiB)")
    N = 600 * MB
    dev = torch.device("cuda")

    def profile(src, label):
        g = src.to(dev, non_blocking=True)
        torch.cuda.synchronize()
        del g
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g = src.to(dev, non_blocking=True)
        submit = time.perf_counter() - t0
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        bw = (src.numel() / GIB) / total
        print(f"  {label:<44} submit={submit * 1000:8.2f} ms  "
              f"total={total * 1000:8.2f} ms  bw={bw:5.2f} GiB/s")
        del g
        torch.cuda.synchronize()

    src = torch.empty(N, dtype=torch.uint8, pin_memory=True)
    src.fill_(9)
    profile(src, "cudaHostAlloc pinned")
    del src
    torch._C._host_emptyCache()

    base, view, padded = aligned_range(N)
    view.fill_(9)
    pmu.pin_memory(view.data_ptr(), padded)
    profile(view, f"cudaHostRegister pinned (is_pinned()={view.is_pinned()})")
    profile(view[PAGE:PAGE + 256 * MB], "  interior 256 MiB view of registered flat")
    pmu.unpin_memory(view.data_ptr())
    del base, view

    src = torch.empty(N, dtype=torch.uint8)
    src.fill_(9)
    profile(src, "pageable")
    del src


def test_5():
    header(5, "does Dynamo guard/specialize on pinnedness?")
    import torch._dynamo as dynamo

    dynamo.reset()
    count = {"n": 0}

    def backend(gm, example_inputs):
        count["n"] += 1
        return gm.forward

    dev = torch.device("cuda")

    @torch.compile(backend=backend)
    def fn(x):
        return (x.to(dev, non_blocking=True).float() * 2).sum()

    pinned = torch.ones(4 * MB, dtype=torch.uint8, pin_memory=True)
    pageable = torch.ones(4 * MB, dtype=torch.uint8)
    _base, view, padded = aligned_range(4 * MB)  # _base kept alive: owns the pages
    view.fill_(1)
    pmu.pin_memory(view.data_ptr(), padded)

    fn(pinned)
    a = count["n"]
    fn(pageable)
    b = count["n"]
    fn(view[:4 * MB])
    c = count["n"]
    print(f"  compiles: pinned={a} pageable={b} register-pinned={c} "
          f"(recompiled: {b > a or c > b})")
    pmu.unpin_memory(view.data_ptr())
    dynamo.reset()


def test_6():
    header(6, "DXGI cost: pin_alloc bucket rounding vs pin_register exact")
    if dxgi_usage() is None:
        print("  DXGI probe unavailable; skipped")
        return
    torch._C._host_emptyCache()
    time.sleep(0.3)
    ODD = 300 * MB
    u0 = dxgi_usage()
    t = torch.empty(ODD, dtype=torch.uint8, pin_memory=True)
    time.sleep(0.2)
    print(f"  cudaHostAlloc 300 MiB: DXGI delta = {(dxgi_usage() - u0) / MB:7.1f} MiB")
    del t
    time.sleep(0.3)
    print(f"  after del (NO emptyCache): retained = {(dxgi_usage() - u0) / MB:7.1f} MiB")
    torch._C._host_emptyCache()
    time.sleep(0.3)
    print(f"  after emptyCache: retained = {(dxgi_usage() - u0) / MB:7.1f} MiB")

    u0 = dxgi_usage()
    t = torch.empty(512 * MB, dtype=torch.uint8, pin_memory=True)
    time.sleep(0.2)
    print(f"  cudaHostAlloc exactly 512 MiB (pow2): delta = "
          f"{(dxgi_usage() - u0) / MB:7.1f} MiB")
    del t
    torch._C._host_emptyCache()

    _base, view, padded = aligned_range(ODD)  # _base kept alive: owns the pages
    view.fill_(1)
    u0 = dxgi_usage()
    pmu.pin_memory(view.data_ptr(), padded)
    time.sleep(0.2)
    print(f"  cudaHostRegister 300 MiB: delta = {(dxgi_usage() - u0) / MB:7.1f} MiB")
    pmu.unpin_memory(view.data_ptr())
    time.sleep(0.2)
    print(f"  after unregister: residual = {(dxgi_usage() - u0) / MB:7.1f} MiB")


def test_7():
    header(7, "per-call overhead: many small registrations (per-tensor pins)")
    for count, size in ((1500, 256 * 1024), (1500, 2 * MB)):
        bufs = []
        for _ in range(count):
            b, v, p = aligned_range(size)
            v.fill_(1)
            bufs.append((b, v, p))
        t0 = time.perf_counter()
        for _, v, p in bufs:
            pmu.pin_memory(v.data_ptr(), p)
        reg = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _, v, _ in bufs:
            pmu.unpin_memory(v.data_ptr())
        unreg = time.perf_counter() - t0
        print(f"  {count} x {size // 1024:6d} KiB: register={reg * 1000:8.1f} ms "
              f"({reg * 1e6 / count:6.1f} us/call)  unregister={unreg * 1000:8.1f} ms")
        del bufs


def test_8():
    header(8, "4 GiB single registration (populated) + unregister")
    _b, v, p = aligned_range(4 * 1024 * MB)  # _b kept alive: owns the pages
    v.fill_(1)
    t0 = time.perf_counter()
    pmu.pin_memory(v.data_ptr(), p)
    print(f"  register 4 GiB populated: {(time.perf_counter() - t0) * 1000:.1f} ms")
    t0 = time.perf_counter()
    pmu.unpin_memory(v.data_ptr())
    print(f"  unregister 4 GiB:         {(time.perf_counter() - t0) * 1000:.1f} ms")


def test_9():
    header(9, "registration speed with DXGI budget heavily committed (~8 GiB)")
    held = []
    committed = 0
    while committed < 8 * GIB:
        b, v, p = aligned_range(1024 * MB)
        v.fill_(1)
        pmu.pin_memory(v.data_ptr(), p)
        held.append((b, v))
        committed += p
    usage = dxgi_usage()
    _b2, v2, p2 = aligned_range(600 * MB)  # _b2 kept alive: owns the pages
    v2.fill_(1)
    t0 = time.perf_counter()
    pmu.pin_memory(v2.data_ptr(), p2)
    dt = time.perf_counter() - t0
    usage_txt = "n/a" if usage is None else f"{usage / GIB:.1f}"
    print(f"  register 600 MiB at ~{usage_txt} GiB usage: {dt * 1000:.1f} ms")
    pmu.unpin_memory(v2.data_ptr())
    t0 = time.perf_counter()
    for _, v in held:
        pmu.unpin_memory(v.data_ptr())
    print(f"  unregister the ~8 GiB held set: {(time.perf_counter() - t0) * 1000:.1f} ms")


def test_10():
    header(10, "page-collision 'resource already mapped' (763bb75)")
    base = torch.empty(3 * PAGE + 1024, dtype=torch.uint8)
    ptr = base.data_ptr()
    start = ((ptr + PAGE - 1) // PAGE) * PAGE
    r1 = base[start - ptr:start - ptr + PAGE]
    pmu.pin_memory(r1.data_ptr(), PAGE)
    try:
        pmu.pin_memory(r1.data_ptr(), PAGE)
        print("  overlapping-page second registration: SUCCEEDED (unexpected)")
        pmu.unpin_memory(r1.data_ptr())
    except Exception as e:
        print(f"  overlapping-page second registration raised: "
              f"{type(e).__name__} (expect cudaError 712): {e}")
    pmu.unpin_memory(r1.data_ptr())


def _fetch_ops_setup():
    from toolkit.memory_management import ingraph_stream, pin_manager

    ingraph_stream.configure_fetch_runtime(depth=2)
    return pin_manager


def test_13():
    header(13, "compiled mm.fetch ops: host_flat swap as ARGUMENT (same shape)")
    pin_manager = _fetch_ops_setup()
    N = 8 * MB
    count = {"n": 0}

    def backend(gm, example_inputs):
        count["n"] += 1
        return gm.forward

    def fetch_fn(host, x):
        token = torch.ops.mm.fetch_start_after(host, x)
        flat = torch.ops.mm.fetch_wait(token, N)
        out = x + flat.view(torch.uint8)[:4].to(x.dtype).sum()
        torch.ops.mm.fetch_free_after(token, out)
        return out

    compiled = torch.compile(fetch_fn, backend=backend, fullgraph=True, dynamic=False)
    x = torch.ones(4, device="cuda")
    hA = pin_manager.pin_alloc(N, "bench", required=True)
    hB = pin_manager.pin_alloc(N, "bench", required=True)
    hA.tensor.fill_(1)
    hB.tensor.fill_(1)
    compiled(hA.tensor, x)
    torch.cuda.synchronize()
    a = count["n"]
    compiled(hB.tensor, x)
    torch.cuda.synchronize()
    b = count["n"]
    print(f"  compiles: host A={a}, host B (new tensor, same shape)={b} "
          f"(recompiled: {b > a})")
    pin_manager.release(hA)
    pin_manager.release(hB)


def test_14():
    header(14, "compiled mm.fetch ops: fresh CLOSURE per 'boundary'")
    pin_manager = _fetch_ops_setup()
    import torch._dynamo as dynamo

    dynamo.reset()
    N = 8 * MB
    count = {"n": 0}

    def backend(gm, example_inputs):
        count["n"] += 1
        return gm.forward

    def make_block_fn(host):
        def fn(x):
            token = torch.ops.mm.fetch_start_after(host, x)
            flat = torch.ops.mm.fetch_wait(token, N)
            out = x + flat.view(torch.uint8)[:4].to(x.dtype).sum()
            torch.ops.mm.fetch_free_after(token, out)
            return out
        return fn

    x = torch.ones(4, device="cuda")
    hA = pin_manager.pin_alloc(N, "bench", required=True)
    hB = pin_manager.pin_alloc(N, "bench", required=True)
    hA.tensor.fill_(1)
    hB.tensor.fill_(1)
    fnA = torch.compile(make_block_fn(hA.tensor), backend=backend,
                        fullgraph=True, dynamic=False)
    fnA(x)
    torch.cuda.synchronize()
    a = count["n"]
    fnB = torch.compile(make_block_fn(hB.tensor), backend=backend,
                        fullgraph=True, dynamic=False)
    fnB(x)
    torch.cuda.synchronize()
    b = count["n"]
    fnA(x)
    torch.cuda.synchronize()
    c = count["n"]
    print(f"  compiles: closure A={a}, fresh closure B={b} "
          f"(recompiled: {b > a}), re-run A={c}")
    pin_manager.release(hA)
    pin_manager.release(hB)
    dynamo.reset()


def test_15():
    header(15, "fetch_start with a PAGEABLE host flat (policy assert)")
    _fetch_ops_setup()
    pageable = torch.empty(8 * MB, dtype=torch.uint8)
    try:
        torch.ops.mm.fetch_start(pageable)
        print("  pageable fetch_start WORKED (assert missing?)")
    except Exception as e:
        print(f"  raised: {type(e).__name__}: {e}")


def test_16():
    header(16, "multi-range H2D submission cost (Krea2-block-sized flat)")
    BLOCK = 384 * MB
    dev = torch.device("cuda")
    host = torch.empty(BLOCK, dtype=torch.uint8, pin_memory=True)
    host.fill_(7)
    dst = torch.empty(BLOCK, dtype=torch.uint8, device=dev)
    stream = torch.cuda.Stream()
    print(f"  {'ranges':>7} {'submit_ms':>10} {'gpu_ms':>8} {'total_ms':>9} "
          f"{'eff GiB/s':>9}")
    for nranges in (1, 4, 8, 16, 24, 32, 64):
        step = BLOCK // nranges
        with torch.cuda.stream(stream):
            for i in range(nranges):
                dst[i * step:(i + 1) * step].copy_(
                    host[i * step:(i + 1) * step], non_blocking=True)
        torch.cuda.synchronize()
        submits, totals, gpus = [], [], []
        for _ in range(5):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                e0.record(stream)
                for i in range(nranges):
                    dst[i * step:(i + 1) * step].copy_(
                        host[i * step:(i + 1) * step], non_blocking=True)
                e1.record(stream)
            submits.append(time.perf_counter() - t0)
            torch.cuda.synchronize()
            totals.append(time.perf_counter() - t0)
            gpus.append(e0.elapsed_time(e1))
        print(f"  {nranges:>7} {min(submits) * 1000:>10.3f} {min(gpus):>8.2f} "
              f"{min(totals) * 1000:>9.2f} "
              f"{(BLOCK / GIB) / min(totals):>9.2f}")
    small = 64 * 1024
    nsmall = 256
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        for i in range(nsmall):
            dst[i * small:(i + 1) * small].copy_(
                host[i * small:(i + 1) * small], non_blocking=True)
    per = (time.perf_counter() - t0) / nsmall
    torch.cuda.synchronize()
    print(f"  marginal submit cost, 64 KiB slices: {per * 1e6:.1f} us/copy")


def test_17():
    header(17, "I3: repin cost under RAM pressure (pagefile-protection case)")
    print("  Justifies whole-run persistence as pagefile protection (plan")
    print("  Explicitly-Rejected #4: phase-owned arenas). Pins a populated")
    print("  buffer, unpins it, forces the freed pages under enough RAM")
    print("  pressure that Windows is likely to have paged them out, then")
    print("  re-registers and measures -- against a clean-RAM baseline.")
    N = 600 * MB

    def repin_once(pressure_tensors):
        _base, view, padded = aligned_range(N)
        view.fill_(7)
        pmu.pin_memory(view.data_ptr(), padded)
        pmu.unpin_memory(view.data_ptr())
        # Under memory pressure the OS is free to evict/reuse these pages
        # once unregistered; touch other large buffers to create that
        # pressure before re-registering the SAME view.
        for b in pressure_tensors:
            b.fill_(3)
        t0 = time.perf_counter()
        pmu.pin_memory(view.data_ptr(), padded)
        dt = time.perf_counter() - t0
        pmu.unpin_memory(view.data_ptr())
        return dt

    baseline = timed("repin, no pressure (0 extra buffers touched)",
                      lambda: repin_once([]))

    import psutil
    avail = psutil.virtual_memory().available
    # Touch enough pageable memory between unpin and repin to plausibly
    # evict the just-freed pages: ~60% of currently AVAILABLE RAM, capped
    # so the bench doesn't itself trigger a system-wide thrash.
    pressure_bytes = min(int(avail * 0.6), 12 * GIB)
    n_pressure_bufs = max(1, pressure_bytes // (256 * MB))
    print(f"  available RAM: {avail / GIB:.1f} GiB; pressure buffers: "
          f"{n_pressure_bufs} x 256 MiB = {n_pressure_bufs * 256 / 1024:.1f} GiB")
    pressure_bufs = []
    for _ in range(n_pressure_bufs):
        pressure_bufs.append(torch.empty(256 * MB, dtype=torch.uint8))
    under_pressure = timed(
        "repin, AFTER touching pressure buffers",
        lambda: repin_once(pressure_bufs),
    )
    pressure_bufs.clear()

    ratio = under_pressure / baseline if baseline > 0 else float("inf")
    print(f"  ratio (pressure/baseline) = {ratio:.2f}x")
    if ratio > 3.0:
        print("  -> repin is materially slower under pressure: persistence "
              "stays the default (pagefile protection), matches plan I3 "
              "expectation.")
    else:
        print("  -> repin stayed cheap even under pressure on this box; "
              "phase-scoped registration would be a legal future "
              "simplification (not exercised by this plan either way).")


def test_12():
    header(12, "cudaPointerGetAttributes as a registry-free pinnedness oracle")
    print("  NOTE: loading cudart via ctypes can segfault at interpreter exit")
    import ctypes

    name = None
    for candidate in ("cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll"):
        try:
            cudart = ctypes.CDLL(candidate)
            name = candidate
            break
        except OSError:
            continue
    if name is None:
        print("  no cudart DLL found; skipped")
        return

    class _Attrs(ctypes.Structure):
        _fields_ = [
            ("type", ctypes.c_int),
            ("device", ctypes.c_int),
            ("devicePointer", ctypes.c_void_p),
            ("hostPointer", ctypes.c_void_p),
        ]

    def ptr_type(ptr):
        a = _Attrs()
        err = cudart.cudaPointerGetAttributes(ctypes.byref(a), ctypes.c_void_p(ptr))
        return err, a.type  # 0=unregistered 1=host 2=device 3=managed

    _b, v, p = aligned_range(64 * MB)  # _b kept alive: owns the pages
    v.fill_(1)
    print(f"  pageable tensor:        err/type = {ptr_type(v.data_ptr())}")
    pmu.pin_memory(v.data_ptr(), p)
    print(f"  register-pinned tensor: err/type = {ptr_type(v.data_ptr())}")
    print(f"  interior pointer:       err/type = {ptr_type(v.data_ptr() + 5 * PAGE)}")
    n = 10000
    t0 = time.perf_counter()
    for _ in range(n):
        ptr_type(v.data_ptr())
    print(f"  latency: {(time.perf_counter() - t0) / n * 1e6:.2f} us/call")
    pmu.unpin_memory(v.data_ptr())


TESTS = {
    1: test_1, 2: test_2, 3: test_3, 4: test_4, 5: test_5, 6: test_6,
    7: test_7, 8: test_8, 9: test_9, 10: test_10, 13: test_13, 14: test_14,
    15: test_15, 16: test_16, 17: test_17,
    12: test_12,  # 12 last: ctypes/cudart exit hazard
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tests", default=None,
                        help="comma-separated test numbers (default: all)")
    parser.add_argument("--list", action="store_true", help="list tests and exit")
    args = parser.parse_args()
    if args.list:
        for num in TESTS:
            print(f"{num:3d}  {TESTS[num].__doc__ or TESTS[num].__name__}")
        return
    if not torch.cuda.is_available():
        print("CUDA unavailable; these benchmarks need the real GPU.")
        sys.exit(1)
    torch.cuda.init()
    selected = (
        list(TESTS) if args.tests is None
        else [int(s) for s in args.tests.split(",")]
    )
    for num in selected:
        TESTS[num]()
        print()
    print("done.")


if __name__ == "__main__":
    # Serialize GPU scripts: two full-model runs on a 12 GB card do not just
    # measure badly, the second OOMs. See scripts/smoke_runtime.py.
    from smoke_runtime import run_locked

    sys.exit(run_locked("bench_pin_assumptions", main))
