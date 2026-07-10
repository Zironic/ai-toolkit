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

## Pinned host memory economics

- Page-locking is per-page kernel work: **0.6-2 GB/s** on consumer Windows.
  Large pin/unpin is seconds, never free. Design accordingly: persistent
  pinned arenas that get repointed beat phase-boundary repinning.
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

Never set `PYTORCH_CUDA_ALLOC_CONF` (max_split_size_mb, gc_threshold):
~30x slowdown near full VRAM on this box. `expandable_segments` is
unsupported on Windows. A pre-bash hook denies these; do not work around it.

## Pointers (source of truth for current behavior)

- `toolkit/memory_management/manager.py` -- planner + live controllers
- `toolkit/memory_management/manager_modules.py` -- per-Linear streaming
- `toolkit/memory_management/bounce_pool.py` -- pinned bounce pool + ledger
- `toolkit/memory_management/pin_manager.py` -- pin authority
- `tests/` -- CPU/sim coverage for the controllers (GPU CI does not exist)
- Mutable status: git-bug tickets (see docs/TICKETS.md)
