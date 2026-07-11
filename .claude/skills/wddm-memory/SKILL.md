---
name: wddm-memory
description: Two-cliff WDDM/DXGI memory model, pinned-memory economics, and allocator-cap rationale for the offload subsystem. Load BEFORE editing toolkit/memory_management/ or diagnosing OOM, raw cudaErrorMemoryAllocation crashes, sudden 10-30x step slowdowns near full VRAM, pinning/budget sizing, or headroom controller behavior on this 12 GB RTX 4070 Windows box.
---

# WDDM/DXGI memory model (hard-earned invariants)

These are platform physics for this box (Windows 11, WDDM, RTX 4070 12 GB).
They were expensive to learn and do not change with refactors. For current
implementation detail, trust the code and git-bug tickets over this file.

## The two cliffs (different failure modes -- never conflate them)

1. **Dedicated VRAM ceiling** (~11.5 GiB usable of 11.99): crossing it makes
   WDDM silently page GPU memory to system RAM. Catastrophic slowdown, **no
   error**. Governed by `torch.cuda.mem_get_info`.
2. **Shared (DXGI NON_LOCAL) budget** (~15.1-15.7 GiB on this box, dynamic):
   a slice of system RAM that **pinned host memory commits against**.
   Exhausting it is a hard `cudaErrorMemoryAllocation` crash. CUDA cannot see
   this budget; only DXGI can. Per-process `CurrentUsage` moves ~1:1 with pins.

Coupling: pinned weights spend the shared budget, which is *also* the
dedicated cliff's overflow valve. **Over-pinning converts a would-be slowdown
into a crash.** Size pinned memory against both cliffs independently. The
real DXGI headroom probe governs the pin cap; RAM proxies (the old RAM*0.25,
~7.94 GiB) are fallback only.

## Allocator hard cap: make the silent cliff loud

`torch.cuda.set_per_process_memory_fraction(1 - hard_gib/total)` makes the
caching allocator raise a real OOM at ~(total - 1 GiB) instead of letting
WDDM page past the ceiling (observed: torch_allocated=12.23 GiB on an
11.99 GiB card, then a crash in an unrelated bystander op). Applied at
`attach_smart_training` / `inference_resident` via
`MemoryManager._apply_wddm_hard_allocator_cap` (Windows-only, idempotent).
Debugging value: a capped allocator OOMs at the *true culprit's* allocation
line -- it once exposed a stray `model.to(cuda)` hauling a whole quantized
model onto the card, where the uncapped run crashed somewhere downstream.

### The cap is also the GC mechanism

Before raising OOM, a capped allocator **frees its idle cached segments
and retries** -- torch `reserved` GCs down toward the cap on demand, so
the reserved-minus-allocated gap (typically 1-3 GiB) is **reclaimable**
(evidence: `scripts/bench_allocator_cap_gc.py`).

- The GC is **all-or-nothing**: one binding allocation dumps *every* idle
  segment, on both the OOM-retry path and the gc_threshold path.
- Reclaim costs tens of ms (~80 ms / 3.25 GiB); seconds under external
  VRAM pressure. Bind the cap at phase boundaries, never per-step.
- **Cache hits bypass both the cap and gc_threshold** -- they act only on
  the fresh-cudaMalloc path; a cache-served allocation ignores the cap
  even when reserved already exceeds it.
- `memory_stats()['num_alloc_retries']` ticks once per OOM-retry reclaim;
  the gc_threshold sweep does NOT tick it (watch `num_device_free`).
  Steady state must hold retries/step at ~0; sustained retries mean the
  cap or residency growth has bitten into fragmentation slack -- back off.
- Minimum viable cap for a phase = within-step peak allocated +
  fragmentation slack (workload-dependent; find it as the knee where a
  cap sweep starts producing per-step retries).

### garbage_collection_threshold: fixed at 0.95; steer with the cap

`run.py` sets `garbage_collection_threshold:0.95` on Windows and it stays
fixed -- the threshold is process-start env config, while the fraction cap
is a runtime per-device API the manager already moves at phase boundaries.
**Control policy: GC target = 0.95 * cap; steer by moving the cap, never
the threshold.**

The GC check runs on fresh cudaMallocs only (never cache hits): when total
reserved exceeds the target it frees idle blocks toward it, *before* the
allocation would fail -- avoiding the OOM-retry path's
`synchronize_and_free_events` device sync, which stalls in-flight
transfer-stream work mid-step.

The target is measured against the WHOLE torch footprint; live allocations
(residents + ring + activations) count against it but cannot be freed. The
**idle-cache allowance is `0.95 * cap - live`**, so the planner can grant a
cache budget via `cap = (planned_live + cache_budget) / 0.95`, clamped by
the WDDM cliff bound. The allowance MUST stay positive at the live peak:
if live alone exceeds the target, thrash self-sustains -- every sweep
dumps ALL idle cache, every would-be reuse becomes a fresh cudaMalloc,
which sweeps again (measured 10x per-alloc cost even in a mild synthetic).
Residency growth eats the allowance 1:1: the layout controller's ceiling
and the cap are coupled. Evidence: `scripts/bench_gc_threshold_allowance.py`.

## Pinned host memory economics

- MEASURED 2026-07-10 (scripts/bench_pin_assumptions.py): `cudaHostRegister`
  on **RAM-resident (populated) pages is ms-scale, ~150 GiB/s** (3 ms/600 MiB,
  27 ms/4 GiB); untouched demand-zero pages ~9 GiB/s; unregister ~33 ms/600
  MiB; per-call overhead ~100-145 us; threading does not help. The old
  "0.6-2 GB/s, pin/unpin is seconds" figure was an artifact of per-tensor
  churn, multi-GiB copies, settle waits, and (plausibly) pagefile faults
  after unpinning under RAM pressure -- the ONE regime where repin is slow.
  Persistent pinned arenas remain the right default as **pagefile
  protection**, not because registration is expensive. Always populate a
  buffer BEFORE registering it (20x cheaper than register-then-populate).
- Also measured: Dynamo/compile is completely indifferent to pinnedness and
  to host-flat identity (same-shape swaps, even via fresh closures, cause
  zero recompiles). Pinnedness gates in compile paths are our policy code.
- Multi-range H2D submission from Python costs ~11 us/copy (24 ranges on a
  384 MiB block = 0.39 ms submit, GPU bandwidth unaffected) -- no native
  transfer runtime is justified.
- Anything pinned through torch's caching host allocator (`pin_memory=True`,
  every `non_blocking=True` D2H staging buffer) is **retained page-locked for
  process lifetime** on free -- it keeps committing against the DXGI budget
  until `torch._C._host_emptyCache()`.
- `cudaHostRegister` (used for weight pins) is the one variant whose unpin
  actually returns budget. That is why the pin manager's eviction rung
  empties the torch host cache first, then unpins weights.
- `pin_manager.py` is the **single authority** for pinned host memory
  (priority: bounce > reserve > weights; explicit-release accounting only,
  no finalizers). Do not pin around it.

## Controllers: training vs sampling are different regimes

- **Training** headroom must be conservative: there is a backward pass to
  reserve for, and paging mid-step is unrecoverable. Govern on the
  cohabitation high-watermark `(total - free) - reserved` vs the WDDM cliff,
  NOT on allocated-side working_headroom_used (it misleads).
- **Sampling** is forward-only, uses a separate manager, and trusts the
  learned peak + 0.5 GiB pad instead of a flat 2 GiB floor. The
  conservative-vs-paging caution is a training concern; do not port it back.
- Every ring/reserve resize destroys prefetch state. Aim for a no-resize
  stable band, not continuous adaptation.

## Forbidden knobs

Never set `max_split_size_mb` (~30x slowdown near full VRAM on this box).
`expandable_segments` is a warn-and-ignore no-op on Windows (torch 2.12).
`garbage_collection_threshold` stays fixed at 0.95; steer the GC by moving
the fraction cap (see above) -- lower thresholds thrash. Allocator config
belongs in `run.py`'s `_configure_windows_torch_allocator`, not in user
env vars.

## Pointers (source of truth for current behavior)

- `toolkit/memory_management/manager.py` -- planner + live controllers
- `toolkit/memory_management/manager_modules.py` -- per-Linear streaming
- `toolkit/memory_management/bounce_pool.py` -- pinned bounce pool + ledger
- `toolkit/memory_management/pin_manager.py` -- pin authority
- `tests/` -- CPU/sim coverage for the controllers (GPU CI does not exist)
- Mutable status: git-bug tickets (see docs/TICKETS.md)
