"""Probe: gc_threshold target is measured against the WHOLE torch footprint,
so live allocations eat the idle-cache allowance 1:1.

Setup: hold LIVE GiB permanently, cap the allocator, then churn fresh
allocations. If live > threshold*cap (allowance <= 0), the GC condition is
permanently true: expect a full idle-cache sweep on (nearly) every fresh
malloc, visible as num_device_free climbing continuously.
"""
import os
import time
import torch

# Serialize GPU scripts against each other (see scripts/smoke_runtime.py):
# this bench allocates hard against the cap and a co-running smoke would OOM.
from smoke_runtime import acquire_gpu_lock

acquire_gpu_lock("bench_gc_threshold_allowance")

assert torch.cuda.is_available()
dev = 0
GIB = 1024 ** 3
total = torch.cuda.get_device_properties(dev).total_memory
thr = float(os.environ["PROBE_THR"])

CAP_GIB = 4.0
LIVE_GIB = 2.0
torch.cuda.set_per_process_memory_fraction(CAP_GIB * GIB / total, dev)
live = torch.empty(int(LIVE_GIB * GIB), dtype=torch.uint8, device="cuda")

target = thr * CAP_GIB
allowance = target - LIVE_GIB
print(
    f"threshold={thr} cap={CAP_GIB} target={target:.2f} GiB "
    f"live={LIVE_GIB} allowance={allowance:+.2f} GiB"
)


def counters():
    s = torch.cuda.memory_stats(dev)
    return s.get("num_device_free", 0), s.get("num_alloc_retries", 0)


# Churn: cycle three sizes so frees leave idle cache behind and the next
# different-size alloc takes the fresh-malloc path.
sizes = [int(0.30 * GIB), int(0.42 * GIB), int(0.55 * GIB)]
f0, r0 = counters()
t0 = time.perf_counter()
per_iter_frees = []
for i in range(30):
    fa, _ = counters()
    z = torch.empty(sizes[i % 3], dtype=torch.uint8, device="cuda")
    del z
    fb, _ = counters()
    per_iter_frees.append(fb - fa)
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) * 1000
f1, r1 = counters()
iters_with_sweep = sum(1 for n in per_iter_frees if n > 0)
print(
    f"30 churn iters: {dt:.1f} ms total; cudaFree +{f1 - f0} "
    f"across {iters_with_sweep}/30 iters; retries +{r1 - r0}; "
    f"reserved now {torch.cuda.memory_reserved(dev) / GIB:.2f} GiB "
    f"(alloc {torch.cuda.memory_allocated(dev) / GIB:.2f})"
)
