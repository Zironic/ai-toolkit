"""Probe: does a set_per_process_memory_fraction cap make the caching
allocator GC (free cached segments) instead of OOM, and which
memory_stats counters observe it?

Pattern: build up idle cache (allocate then free a few GiB in varied
sizes), then tighten the cap below current reserved and allocate again.
Watch reserved, num_alloc_retries, num_device_free.
"""
import time
import torch

# Serialize GPU scripts against each other (see scripts/smoke_runtime.py):
# this bench allocates hard against the cap and a co-running smoke would OOM.
from smoke_runtime import acquire_gpu_lock

acquire_gpu_lock("bench_allocator_cap_gc")

assert torch.cuda.is_available()
dev = 0
GIB = 1024 ** 3


def stats():
    s = torch.cuda.memory_stats(dev)
    return {
        "alloc": torch.cuda.memory_allocated(dev) / GIB,
        "reserved": torch.cuda.memory_reserved(dev) / GIB,
        "retries": s.get("num_alloc_retries", 0),
        "dev_free_cnt": s.get("num_device_free", 0),
        "dev_alloc_cnt": s.get("num_device_alloc", 0),
    }


def show(tag):
    st = stats()
    free_b, total_b = torch.cuda.mem_get_info(dev)
    print(
        f"{tag:36s} alloc={st['alloc']:.2f} reserved={st['reserved']:.2f} "
        f"driver_free={free_b / GIB:.2f} retries={st['retries']} "
        f"cudaFree={st['dev_free_cnt']} cudaMalloc={st['dev_alloc_cnt']}"
    )
    return st


total = torch.cuda.get_device_properties(dev).total_memory
print(f"total = {total / GIB:.2f} GiB")

# 1) Build idle cache: varied-size allocs, then free them all.
junk = []
for mb in (37, 128, 256, 64, 512, 96, 384, 256, 128, 512, 200, 300, 450):
    junk.append(torch.empty(mb * 1024 * 1024, dtype=torch.uint8, device="cuda"))
show("after alloc ~3.3 GiB varied")
junk.clear()
base = show("after free (idle cache built)")

# 2) Tighten cap to ~1.5 GiB below current reserved.
cap_gib = max(0.5, base["reserved"] - 1.5)
frac = cap_gib * GIB / total
torch.cuda.set_per_process_memory_fraction(frac, dev)
print(f"cap set to {cap_gib:.2f} GiB (fraction {frac:.3f})")

# 3) Allocate under the tightened cap -- should force GC, not OOM.
t0 = time.perf_counter()
try:
    x = torch.empty(int(0.75 * GIB), dtype=torch.uint8, device="cuda")
    ok = True
except torch.cuda.OutOfMemoryError as e:
    ok = False
    print(f"OOM: {e}")
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) * 1000
after = show(f"after 0.75 GiB alloc under cap")
print(f"alloc under cap: ok={ok} took {dt:.1f} ms")
print(
    f"delta: retries +{after['retries'] - base['retries']}, "
    f"cudaFree +{after['dev_free_cnt'] - base['dev_free_cnt']}, "
    f"reserved {base['reserved']:.2f} -> {after['reserved']:.2f} GiB"
)

# 4) Steady-state churn cost when the cap binds every iteration:
#    alloc/free cycle sized so each round trips the GC.
torch.cuda.synchronize()
per_iter = []
b2 = stats()
for i in range(20):
    t0 = time.perf_counter()
    y = torch.empty(int(0.6 * GIB), dtype=torch.uint8, device="cuda")
    del y
    torch.cuda.synchronize()
    per_iter.append((time.perf_counter() - t0) * 1000)
a2 = stats()
print(
    f"churn under binding cap: {sum(per_iter) / len(per_iter):.2f} ms/iter avg "
    f"(min {min(per_iter):.2f}, max {max(per_iter):.2f}); "
    f"retries +{a2['retries'] - b2['retries']}, cudaFree +{a2['dev_free_cnt'] - b2['dev_free_cnt']}"
)

# 5) Same churn with a loose cap for comparison.
del x
torch.cuda.set_per_process_memory_fraction(0.95, dev)
torch.cuda.synchronize()
per_iter2 = []
b3 = stats()
for i in range(20):
    t0 = time.perf_counter()
    y = torch.empty(int(0.6 * GIB), dtype=torch.uint8, device="cuda")
    del y
    torch.cuda.synchronize()
    per_iter2.append((time.perf_counter() - t0) * 1000)
a3 = stats()
print(
    f"churn with loose cap:    {sum(per_iter2) / len(per_iter2):.2f} ms/iter avg "
    f"(min {min(per_iter2):.2f}, max {max(per_iter2):.2f}); "
    f"retries +{a3['retries'] - b3['retries']}, cudaFree +{a3['dev_free_cnt'] - b3['dev_free_cnt']}"
)
