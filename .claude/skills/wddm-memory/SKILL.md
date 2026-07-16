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
   error**. Governed by `vram_budget.device_free_bytes` (NVML-backed) -- **NOT**
   by `torch.cuda.mem_get_info`, see below.
2. **Shared (DXGI NON_LOCAL) budget** (~15.1-15.7 GiB on this box, dynamic):
   a slice of system RAM that **pinned host memory commits against**.
   Exhausting it is a hard `cudaErrorMemoryAllocation` crash. CUDA cannot see
   this budget; only DXGI can. Per-process `CurrentUsage` moves ~1:1 with pins.

Coupling: pinned weights spend the shared budget, which is *also* the
dedicated cliff's overflow valve. **Over-pinning converts a would-be slowdown
into a crash.** Size pinned memory against both cliffs independently. The
real DXGI headroom probe governs the pin cap; RAM proxies (the old RAM*0.25,
~7.94 GiB) are fallback only.

## The free signal: only NVML tells the truth

**`torch.cuda.mem_get_info` free is NOT physical availability.** It is what the
driver will promise *this* process, and on WDDM that promise is backed by paging
other processes out. **DXGI LOCAL `Budget` is no better** for this purpose: it is
a *permission*, not an availability -- it shrinks under contention but still
grants memory the OS intends to obtain by evicting the other tenant. That
permission IS the silent-paging mechanism.

Measured with a second process holding 9 GiB of the 12 GB card:

```text
NVML physical free              :  0.39 GiB   <- the truth (== nvidia-smi)
torch.cuda.mem_get_info free    : 10.85 GiB   <- promises nearly the whole card
DXGI LOCAL Budget - CurrentUsage:  4.58 GiB   <- also over-reports
```

So **govern on `vram_budget.device_free_bytes`** (NVML-backed, `min` with the
driver number), never raw `mem_get_info`. NVML sees every process on the GPU, is
cross-platform (`nvml.dll` / `libnvidia-ml.so.1`, ctypes, no dependency), and at
~3 us/call is ~25x *cheaper* than `mem_get_info` (~80 us) -- it is safe on hot
paths. Sensor: `nvml_meminfo.py`. DXGI keeps its own job: the *shared* NON_LOCAL
pin budget, not the dedicated cliff.

Consequence: **any other GPU tenant silently wrecks a run** if you trust the
driver number -- a game, a ComfyUI server, or an orphaned training job. Observed
in production: a 73 s/it job ran at **304 s/it** because a duplicate of itself
was still alive holding VRAM; the planner saw "4.33 GiB free" (physical: 0.4)
and sized residency into memory that did not exist. No error, no allocator
retry -- `num_alloc_retries` does not catch this, because the allocation
genuinely succeeds. Only throughput tells you.

Delta-probe loops that measure this process's *own* allocation deltas may keep
using raw `mem_get_info` -- they measure differences, not availability.

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
  Cheap registration enables **residency-aware arena pinning**: the runtime
  keeps only streamed (non-resident) blocks plus a small demotion reserve
  (`DEFAULT_DEMOTION_PIN_RESERVE_BLOCKS = 2`, deterministic demotion order)
  registered, and unpins resident blocks after their promotion copies settle
  (`residency.pin_requirements_for_plan`,
  `immutable_runtime.reconcile_pin_policy`), returning DXGI shared budget.
  The canonical host flats stay allocated and populated while unregistered,
  so the slow pagefile-repin regime only bites under real RAM contention.
  Always populate a buffer BEFORE registering it (20x cheaper than
  register-then-populate).
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
- **Sampling residency budgets on the allocated side**: the planner free
  input is max(driver-free, `0.95*cap - allocated + hard`) via
  `vram_budget.sampling_allocator_budget_free_bytes` -- driver-free counts
  torch's reclaimable idle cache as used and would under-promote. GC health
  per training window is in the perf log (`alloc_retries_delta`,
  `cuda_free_count_delta`); the digest prints an "Allocator GC" summary and
  per-window `reclaimable_at_peak`. Exercise a phase's footprint empirically
  with the current smoke harnesses (`scripts/smoke_krea2_train_cuda.py`,
  `scripts/smoke_krea2_inference_cuda.py`,
  `scripts/smoke_transformer_train_cuda.py`).
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

- `toolkit/memory_management/arena_offload/` -- generic arena dispatcher
  (primary offload runtime), backed by `immutable_runtime.py` for shared
  training/sampling source and residency transitions
- `toolkit/memory_management/manager.py` -- legacy planner + live controllers
- `toolkit/memory_management/manager_modules.py` -- per-Linear streaming
- `toolkit/memory_management/bounce_pool.py` -- pinned bounce pool + ledger
- `toolkit/memory_management/pin_manager.py` -- pin authority
- `toolkit/memory_management/vram_budget.py` -- NVML-backed free/budget sensors
- `toolkit/memory_management/allocator_cap.py` -- WDDM hard allocator cap
- `tests/` -- CPU/sim coverage for the controllers (GPU CI does not exist)
- Mutable status: git-bug tickets (see docs/TICKETS.md)
