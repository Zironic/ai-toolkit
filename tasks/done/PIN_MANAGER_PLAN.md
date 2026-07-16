# Pin Manager — single authority for pinned host memory

> **Completed 2026-07-16:** The single pin authority shipped and passed its
> focused GPU validation under ticket `da1aae5`.
>
> Durable plan. Status lives in a git-bug ticket, not here.
> Related: `INGRAPH_STREAM_PLAN.md` Phase 6 (planner coordination of
> shared-budget consumers — this plan is the general mechanism Phase 6
> needs), the DXGI pin-for-speed effort (probe + two-cliff model),
> `docs/decisions/UPSTREAM_PR_PLAN.md` (WDDM gating stays local).

## Problem

Pinned (page-locked) host memory on Windows/WDDM commits against the DXGI
NON_LOCAL shared budget (~15 GiB on the reference box, **dynamic** — it
shrinks under system RAM pressure; observed at 9.8 GiB). Exhausting it is a
hard `cudaErrorMemoryAllocation`, not a slowdown. Today at least six
independent code paths pin against that one budget, each deciding alone:

| # | Consumer | Governance today |
|---|----------|------------------|
| 1 | Attach per-tensor weight pinning (`_pin_tensor_in_place`) | ledger + `_cap_auto_pin_budget` (DXGI probe) |
| 2 | Bounce pool buffers | ledger + own auto-cap over leftovers |
| 3 | Ingraph block packs (`ingraph_stream._empty_host_flat`) | none; private `_host_emptyCache` retry |
| 4 | Legacy block-stream host pack (`manager_modules.py` ~1015) | none |
| 5 | Async-save `PinnedStager` (`toolkit/async_save.py`) | none |
| 6 | torch caching host allocator (every `non_blocking=True` D2H: grad staging, sampler-restore snapshots) | none, invisible, and it CACHES — freed buffers stay committed until `torch._C._host_emptyCache()` |

Observed failure modes, all from this uncoordination:

- Auto-pin spent the entire DXGI headroom at attach; first backward's grad
  D2H staging (consumer 6) then hard-crashed (train smoke, 2026-07-04).
- Sampler restore left ~12 GiB cached in the host allocator (consumer 6),
  starving ingraph pack pinning until an ad-hoc `_host_emptyCache` retry
  was bolted into consumer 3.
- Bounce pool live buffers on top of 14 GiB of packs crossed the budget
  during the all-28 sampling run (fixed that day by hand-disabling the
  pool).
- Uncapped auto-pin at startup crashed with raw
  `cudaErrorMemoryAllocation` (pre-DXGI-probe era).

Each fix so far has been a point patch inside one consumer. The decision
"what is pinned, and who yields when the budget is tight" must be made in
ONE place.

## Design

New module `toolkit/memory_management/pin_manager.py`. Everything that
page-locks host memory goes through it; nothing else calls
`.pin_memory()` / `pin_memory=True` directly (enforced by test, see
below). The existing `bounce_pool` ledger + DXGI probe move behind it.

### Core model

- **One ledger, keyed by consumer class** (`weights`, `bounce`,
  `ingraph_pack`, `save_stager`, `host_cache_reserve`, ...), replacing the
  single `_pinned_bytes_total` counter. Per-class totals make the budget
  explainable in logs and let policy reason about who holds what.
- **Budget source of truth:** the DXGI NON_LOCAL probe
  (`dxgi_pinned_headroom`), re-queried at decision time because the budget
  is dynamic; the RAM-fraction proxy remains only as fallback when the
  probe is unavailable. Spill reserve stays as today.
- **Standing reserves for implicit consumers.** The torch caching host
  allocator cannot ask permission, so the manager holds back a configured
  reserve for it (grad-staging + save-stager + working slack; sized from
  measurements, config-overridable). Explicit consumers are only granted
  `headroom - reserves`.
- **Grant API (synchronous, fail-fast):**
  - `pin_alloc(nbytes, kind) -> pinned uint8 tensor` (or raises
    `PinBudgetExceeded` with a per-class ledger dump in the message);
  - `pin_tensor_in_place(t, kind) -> bool` (False = stay pageable —
    degradable consumers like attach weight pinning);
  - `release(handle)` / weakref-based auto-release, keeping ledger truth.
  - `reconcile()` — escalation path used before failing a must-pin
    request: `torch._C._host_emptyCache()` first (reclaims consumer 6's
    cache), then ask *evictable* consumers to shrink (bounce pool can
    drop buffers; attach-pinned weights can unpin per layer), then raise.
    This replaces consumer 3's private retry.
- **Policy hooks live here, not in consumers.** The rules we learned by
  crashing become explicit, testable policy:
  - full-model pin (attach or packs) => bounce pool budget 0 (the
    pinned-source bypass makes it redundant);
  - ingraph packs requested => attach per-tensor pinning 0 (repoint=False
    packs would double-commit);
  - training mode => grad-staging reserve is mandatory; sampling mode =>
    it can be released to packs.
- **Diagnostics:** one log line per mode transition:
  `pin ledger: budget=... probe=... weights=... bounce=... packs=...
  reserve=... free=...`; snapshot API for the smokes/perf digest.

### Allocation strategy — who gets pin, in what order

The grant policy is not first-come-first-served; it is a fixed priority
scheme evaluated at attach/mode-transition time (the decision point where
the full demand set is known):

**Step 0 — the full-pin check.** First ask: does the DXGI headroom (minus
the mandatory reserves in step 2) cover pinning **all offloaded weights**?
If yes, that is the whole plan: pin everything, set the bounce pool budget
to 0. A fully pinned source set makes the pool redundant (pinned-source
bypass) and every byte it would take competes with nothing useful. This is
the fast path and the preferred end state (the ~2x pin-for-speed regime).

**Otherwise, priority order for a partial-pin world:**

1. **Bounce pool first.** When weights are not all pinned, the pool is
   what keeps streaming fast for the pageable remainder — it is the
   highest-leverage bytes-per-GiB consumer and gets its budget before any
   weight is pinned.
2. **Torch host-allocator reserve second.** The standing reserve for the
   implicit consumer (grad D2H staging in training, save-stager bursts,
   sampler-restore snapshots). Mandatory in training mode; sized from
   measurement, config-overridable. This is carved out *before* weights so
   backward can never be starved by a greedy weight pin (the 2026-07-04
   train-smoke crash).
3. **Weights last, as budget remains.** Whatever headroom is left after 1
   and 2 goes to pinning offloaded weights, highest-traffic layers first
   (same ordering the attach path uses today). Weights are the degradable
   consumer: an unpinned weight still works, just slower through the pool.

The priority is strict at the boundary: a box that can *almost* full-pin
does NOT get to pin 95% and starve the pool. If the full-pin check fails,
the pool and the reserve are funded in full first, even when that means
evicting a significant share of the model back to pageable — there is no
intermediate "pin nearly everything, tiny pool" regime. DECIDED
2026-07-04.

OPEN (measure after S4): A/B the two partial regimes on a real config —
"bounce pool + fewer pinned weights" vs "no pool + more pinned weights"
(pool-less streaming straight from per-tensor pins). Same manager code
serves both; the winner just changes the default policy constant. The
smokes are the harness: same step loop, flip the policy, compare
steady step time.

**Eviction is the reverse of priority.** When headroom shrinks (dynamic
DXGI budget) or a must-pin request cannot be granted, `reconcile()` takes
from the lowest-priority holder first:

1. empty the torch host-allocator cache (free bytes it is merely caching —
   costs nothing but a later re-alloc);
2. unpin weights, lowest-traffic first (degrades speed, never correctness);
3. shrink the bounce pool only below its functional floor as a last resort
   (it is top priority precisely because taking from it hurts most);
4. the standing reserve is never granted away in training mode — a request
   that would need it is refused instead (the alternative is a backward
   crash).

Ingraph packs slot into this scheme as a *weights-tier* consumer with an
all-or-nothing granularity twist: a block pack is only useful fully
pinned, so packs are granted per whole block (demote the block to the
legacy path if it does not fit), and — per the repoint=False
double-commit rule — pack pinning and per-tensor attach pinning for the
same weights are mutually exclusive.

### What changes in each consumer

1. `_pin_tensor_in_place` + `_cap_auto_pin_budget`: cap logic moves into
   the manager (`kind="weights"`); manager decides, module obeys.
2. Bounce pool: allocates via `pin_alloc(kind="bounce")`; its auto-cap
   collapses into manager policy; implements the `shrink()` evictable
   protocol.
3. Ingraph packs: `_empty_host_flat` becomes a thin call to
   `pin_alloc(kind="ingraph_pack")`; drops its private retry (reconcile
   covers it). Fixes the "packs pin unguarded" Phase 6 debt.
4. Legacy block-stream pack: same, `kind="block_stream"`.
5. Async-save stager: `pin_alloc(kind="save_stager")` — small, but it
   must be visible in the ledger during checkpoint saves.
6. Host allocator: cannot be routed, so it is *modeled* — the standing
   reserve plus `reconcile()`'s cache-empty. Optionally sample
   `torch.cuda.host_memory_stats()` (if available in torch 2.12) to track
   its actual footprint in the ledger snapshot.

### Non-goals

- No async/waiting grant queue — decisions stay synchronous and
  fail-fast per repo convention.
- No cross-process coordination (one training process owns the GPU).
- No change to *device* VRAM planning; this is host-pin-side only. The
  smart planner keeps deciding residency; it consults the pin manager for
  "how much can I pin" instead of calling `_cap_auto_pin_budget` itself.

## Slices

1. **S1 — Module + ledger + probe migration.** `pin_manager.py` with
   per-class ledger, DXGI-first headroom, reserves, `pin_alloc` /
   `pin_tensor_in_place` / `release` / snapshot. `bounce_pool`'s ledger
   functions become delegating shims (keep API for external callers).
   CPU tests: ledger accounting, reserve arithmetic, fallback proxy,
   grant/deny boundaries (mock probe).
2. **S2 — Route the explicit consumers** (1-5 above), one commit each,
   behavior-preserving at default settings. Grep-based conformance test:
   no `pin_memory=True` / `.pin_memory()` outside `pin_manager.py` and
   tests.
3. **S3 — Policy + reconcile.** Mode-aware reserves, the
   full-pin=>no-bounce and packs=>no-attach-pin rules, `reconcile()`
   escalation with evictable-consumer protocol. Sim tests for the three
   historical crash scenarios (each must now be refused or reconciled,
   not crash).
4. **S4 — Wire into planner + smokes.** `attach_smart_training` and
   ingraph enablement consult the manager; both smoke scripts print the
   ledger snapshot per phase; perf-log window gains `pinned_by_class`.

## Acceptance

- The three historical crashes, reproduced as tests/sims against a mocked
  probe, are prevented by policy (deny/reconcile), not by luck.
- Full `tests/` suite green; flags-off behavior unchanged (the manager
  with default policy must reproduce today's numbers on the roomy path).
- Train + sampler smokes pass end-to-end with auto budgets (no manual
  `--pinned-weight-gib` / `AI_TOOLKIT_BOUNCE_POOL_GIB` knobs needed for
  the all-28 sampling recipe — the recipe becomes policy).
- One grep finds every pin site: `pin_manager.` call sites only.
