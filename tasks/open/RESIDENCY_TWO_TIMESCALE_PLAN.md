# Residency control via a two-timescale cap/demote loop -- implementation plan

> **git-bug:** `0c577ef` -- "Make the two-timescale residency FSM live
> (arena-native, block-granular)" owns the remaining wiring. Refactor context:
> `553ffec`. Closely related: `5fa0e3d` (autotune working_reserve + keep_last),
> `68d3565` (manual WDDM cliff guard), `628b0cb` (immutable arena). This file is
> the durable design; current state lives in the ticket.

> **STATE (verified 2026-07-12).** The **policy is built and green**: all five
> pure helpers (`allocator_allowance_bytes`, `cap_bytes_for_live`,
> `cap_can_host_promotion`, `residency_promote_ok`, `residency_fsm_step`) live in
> `vram_budget.py`, with `tests/test_residency_two_timescale.py` passing (20).
>
> The **wiring (step 2 of the sketch below) was never done**: grep finds no
> production caller of the FSM or either gate. The single live caller anywhere is
> `cap_bytes_for_live` in `MemoryManager.inference_resident` (`manager.py:6024`)
> -- the sampling-side cap reclaim. The FSM has never run in a training step.
>
> That wiring is now folded into `arena_offload/policy.py` (see
> `UPSTREAM_ARENA_EXTRACTION_PLAN.md`, Phase 2) rather than being retrofitted
> into `manager.py`: `vram_budget.py` is in the host-memory layer that both
> backends import, so the FSM is callable from the arena policy as-is. Build it
> once, there. This plan's open questions (training slack-pad sizing, `Kclean` /
> `Kverify` / `N`) carry over unchanged.
>
> **Caveat on the throughput claim (added 2026-07-12).** The 2.50 GiB reclaimable
> above is real, but the device-side fetch ring made Krea2 training
> **compute-bound, not PCIe-bound** (occupancy 87 -> 96%). If streaming is already
> hidden behind compute, a promoted block saves ~nothing on step time and the
> banked VRAM buys *headroom* (resolution, batch, fewer OOM demotes) rather than
> *throughput*. The FSM's guaranteed win is safety - banking VRAM without crossing
> the thrash/paging cliff. Measure the marginal resident block (`0c577ef` S0)
> before building the climb; the promote gate's aggressiveness depends on the
> answer.

Goal: turn the measured cap-descent result into a live residency policy that
banks the reclaimable VRAM **safely**, by separating a cheap, reversible
allocator-cap lever from the expensive, prefetch-destroying residency lever, and
by always approaching residency **from below**.

## What already exists (do not rebuild)

- **Historical `--cap-descent` measurement.** This lived on the retired
  `scripts/smoke_krea2_ingraph_cuda.py`. The current inference entry point is
  `scripts/smoke_krea2_inference_cuda.py`, which does not expose the deleted
  legacy cap-descent control; add an arena-native probe there only if this
  measurement must be repeated. The historical probe notch-descended
  the allocator cap (settle + measure pass per notch), reports the phase's true
  footprint floor and the dirty knee. Restores the manager cap afterwards.
- **Per-window GC telemetry** -- `manager.py training_runtime_diagnostics` logs
  `alloc_retries_delta`, `cuda_malloc_count_delta`, `cuda_free_count_delta` via
  the pure `_gc_counter_deltas` helper.
- **Digest** -- `digest_perf_log.py` prints a run-level `Allocator GC:` summary
  and per-window `alloc_gc:` line with
  `reclaimable_at_peak = peak_reserved - peak_alloc`.
- **Sampling allocated-side budget** -- `vram_budget.sampling_allocator_budget_free_bytes`
  = `max(driver_free, 0.95*cap - allocated + hard)`, wired into
  `inference_resident` so torch idle cache can't suppress residency.

This plan consumes those; it does not duplicate them.

## Measured ground truth (Krea2, 512px, fp8, compiled, gc_threshold 0.95)

| Quantity | Value |
| --- | --- |
| True footprint floor (clean floor cap) | **7.35 GiB** |
| Peak live allocations | **6.77 GiB** |
| Fragmentation slack at the knee | **0.21 GiB** (one 0.25 notch) |
| Reclaimable vs today's base cap (9.85) | **2.50 GiB** |
| Streamed block size (Krea2) | **~0.375 GiB** (per-model; measure it) |

The allowance model validated to within one notch: `0.95*7.35 = 6.98 > 6.77`
(clean, +0.21) vs `0.95*7.10 = 6.75 < 6.77` (dirty, -0.02). So the knee is real
and the model predicts it.

**Key arithmetic:** a block (~0.375) is *larger* than the frag slack (0.21). So
one resident block cannot be absorbed by the slack -- adding/removing a block is
a coarse move that must be paired with cap headroom, not hidden in the pad.

## Governing principle 1: asymmetric loss -> approach from below

The two residency errors cost very differently:

- **One block too few:** one extra streamed block/step. Linear, gentle, fully
  recoverable, never touches thrash.
- **One block too many:** pushes live past `0.95*cap` -> self-sustaining thrash
  (sampling) or silent WDDM paging (training), and the correction is a deferred,
  prefetch-destroying resize.

Therefore **start at N-1 (one block below the theoretical fit) and climb**, never
start above and fall back. Undershoot is the cheap-error side; bias into it. This
is the same shape as the existing keep_last speed-climb and the AIMD reserve.

At N-1 the allowance is `0.21 + 0.375 ~= 0.585 GiB` -- comfortably clear of the
0.21 knee and robust to the fact that the 0.21 was measured on one config whose
fragmentation shifts when residency changes. Staying below the fit line keeps the
allowance positive **by construction**, so no per-block cap-lift is required just
to hold N-1.

## Governing principle 2: two-timescale control (the cap vs residency levers)

Two levers raise the GC allowance `0.95*cap - live`. They differ only in cost, so
split them by timescale:

- **Inner lever -- the allocator cap** (`set_per_process_memory_fraction`).
  Cheap, **reversible**, **no prefetch loss**. Buys allowance from reclaimed
  dedicated VRAM / cliff headroom. Use it for the fine, frequent adjustments.
- **Outer lever -- residency** (ring/reserve resize). Expensive, sticky,
  **destroys prefetch**. Use it only when the inner lever saturates.

Control law: absorb normal fluctuation in the cap; escalate to a resident demote
only when the cap cannot help. This directly serves the "no-resize stable band"
goal -- residency stays put because the cheap lever soaks up the wobble.

### Escalation is prechecked, not trial-and-error

The cap is clamped above by the cliff bound (`vram_budget.cap_fraction`'s
`total - hard - non_torch`). **The arithmetic is in cap-space, but the allowance
we care about is in target-space, and the two differ by the `gc_threshold`
(0.95) factor** -- a cap raise of `d` only adds `0.95*d` of GC target / allowance
(`allowance = 0.95*cap - live`). So the precheck must divide by 0.95, not compare
raw byte-for-byte:

```
gc = 0.95
# Promoting a block raises live by block_size; to keep the pad afterward the GC
# target (gc*cap) must cover live + block + slack. Solve for the cap it needs.
need_cap = (live + block_size + slack_pad) / gc

if cliff_bound >= need_cap:  raise cap to need_cap    # tier 1 -- no prefetch cost
else:                        demote 1 resident        # tier 2 -- cap pinned at cliff
```

A naive `cliff_bound - current_cap >= block_size` test under-reserves by the
0.95 factor and would license a promotion that immediately binds.

Then **verify**: after a cap raise, confirm `num_alloc_retries -> 0` over a
window (the `non_torch` / fragmentation estimate can be wrong); escalate to a
demote only if it does not hold. Sequence: precheck -> act -> verify -> escalate.

### When the GC actually fires -- the one-step lag

Changing the cap does **not** reclaim or rebind anything at the moment of the
`set_per_process_memory_fraction` call. The caching allocator only acts on a
**fresh cudaMalloc**: it sweeps idle segments toward the new target, or (if that
is not enough) OOM-retries, *when the next allocation asks for memory that the
cache cannot serve*. The next such allocation is in the next `forward()` -- i.e.
**the next training/sampling step**. Cache-served allocations don't touch it at
all.

Consequences the controller must respect:

- **Act at boundary K, observe at boundary K+1.** A cap change made at a phase
  boundary produces its `num_alloc_retries` / `num_device_free` signal only in
  the *next* window. This is a unit-delay feedback loop -- do not read the
  counters in the same window you moved the cap and conclude "it worked."
- **Do not stack two cap moves in consecutive boundaries** expecting the first to
  have settled. Move, wait one window, then read. (Feeds the hysteresis in Open
  Questions.)
- **Reclaim is lazy.** Lowering the cap to hand dedicated VRAM to a co-tenant
  (VAE/TE decode) frees nothing until the next torch cudaMalloc forces the sweep;
  a co-tenant allocating on a path that bypasses torch's allocator may run before
  the reserved pool has actually shrunk. Where a reclaim must be realized *before*
  a specific co-tenant runs, force it explicitly (a throwaway alloc or
  `empty_cache`) rather than assuming the fraction change did it.

This lag is compatible with the existing per-step measurement cadence; it just
means every decision reads state that is one step stale, which the from-below
bias already tolerates (undershoot is safe to sit in for a step).

### The hierarchy self-adjusts to the regime (no branch needed)

- **Sampling:** live peak 6.77 vs cliff ~11.5 -> ~4.7 GiB of cap range. Tier-1 is
  deep; demotes are rare. Fluctuation absorbed entirely in the cheap lever.
- **Training:** cap already pinned near the cliff (~11.0/11.5) -> `cap_headroom
  ~= 0`. Tier-1 is shallow, so escalation to demote happens quickly -- correct,
  because at the cliff the only free allowance comes from lowering live.

Same law, opposite behavior, driven purely by measured `cap_headroom`.

### The cap is an invariant, not a relief rung

The existing pre-step demote guard (`MemoryManager.prepare_training_memory_for_shape`,
`manager.py:3660`) is structured as a **relief ladder**: rung 1 `empty_cache()`,
rung 2 demote canonical sidecars. That shape is wrong under this design, in two
separate ways.

**1. Its predicate already assumes a perfect cap, so it already asks the right
question.** Both pressure signals predict on *peak allocated* (live), not peak
reserved, and say so:

- `_predict_dxgi_local_peak_bytes` (`manager.py:2417`) returns
  `non_torch + peak_allocated` -- "Historical allocator reservation is cache
  appetite, not required live memory."
- `training_cliff_predicted_peak_free_gib` (`vram_budget.py:439`) --
  "ask whether peak allocated memory itself clears the WDDM hard floor."

So "pressure" already means *even with zero idle cache, live at peak breaches the
floor*. That is exactly the regime in which the cap lever **cannot help** -- you
cannot cap your way out of live memory. When this predicate fires, demotion
really is the only lever left, and the guard is right to reach for it.

**2. Which makes the `empty_cache` rung structurally dead** (`manager.py:3755-3789`).
It reclaims idle cache and re-evaluates a predicate that does not look at idle
cache. In the DXGI path `non_torch = usage - current_reserved`, and `empty_cache`
drops both by the reclaimed amount, so `non_torch` is invariant; `peak_allocated`
is untouched. In the physical path NVML free rises by the reclaimed bytes while
`torch_reserved` falls by the same, so `non_torch_gib` is invariant. The re-check
can only flip if another process moved memory between the two samples -- i.e.
noise. What it reliably costs is an all-or-nothing GC (~80 ms typical, seconds
under VRAM pressure) on every pressure event, before falling through to demote
anyway. Delete the rung. (Reasoned from the arithmetic, not yet measured --
confirm with an instrumented pressure event before removing.)

**The correction.** The cap is not something you reach for when pressure hits. It
is a standing invariant, bound at phase boundaries to `live_at_peak + slack`, and
its job is to *make the guard's live-based prediction true* rather than merely
hoped-for. Nothing enforces that today: the allocator is free to hoard reserved
above live for a whole step, and the predicate simply assumes it will not.

With the cap bound, the guard becomes the last-resort floor guard it should be:

- Pre-step pressure fires only on **genuine growth** -- a new larger shape bucket,
  or a foreign VRAM tenant arriving. Demotion is then correct.
- The **other** demote trigger is discovered by the cap lever, not by this
  predicate: if the cap you would need sits below the thrash threshold
  (`allowance = 0.95*cap - live <= 0`), the cheap lever is not viable at that
  setting and residency must come down. That is the `CAP_VERIFY ->
  DEMOTE_REQUIRED` edge in the state machine below -- measured cache churn at the
  allotment we actually need is what forces the sticky lever.

Demotes should therefore be **rare**, not because the controller tries the cheap
lever first and it usually works, but because a correctly bound cap keeps
"live alone breaches the floor" a rare event.

## The climb (promote) gate -- built on the shipped telemetry

At a phase boundary, promote N-1 -> N (convert one streamed block to resident)
when, sustained over a window:

- `num_alloc_retries == 0` (nothing cap-binding), **and**
- `reclaimable_at_peak (= peak_reserved - peak_alloc) > block_size + slack_pad`
  -- more than a block of idle cache still sits reclaimable *at the peak*, so the
  promotion still leaves the pad.

No hardcoded floor: `--cap-descent` gives the ceiling per config, the live
counters gate the last block. Re-run the descent as a **per-config
recalibration** whenever the layout changes (resolution, pinned-arena on/off,
model) -- fragmentation, hence the knee, moves with the layout.

### Signal per regime

- **Sampling** gates on `num_alloc_retries` (+ `reclaimable_at_peak`): overshoot
  is loud (the cap ticks a retry) and recoverable.
- **Training** gates the promote on the **cohabitation high-watermark**
  `(total - free) - reserved` clearing the cliff across *several* steps -- its
  dangerous overshoot is silent paging, which the retry counter does NOT catch,
  and its peak is noisy so one clean step is not enough evidence.

## Control state machine (hysteresis)

The two-tier levers plus the one-step GC lag demand an explicit FSM, not a
per-window if-ladder -- otherwise the loop hunts (promote/demote oscillation) or
reads stale counters as if fresh. One transition per phase boundary; each verify
phase spends `Kverify` windows because a cap or residency move only shows its
`num_alloc_retries` signal on the *next* step (see the one-step lag above).

```
COLD
    Measurements invalid (post-compile / retrace / layout change).
    Wait for compile- and retrace-free windows before trusting any counter.
    -> STABLE once measurements are clean and stable.

STABLE
    No retries for Kclean windows; no layout or graph-generation changes.
    Eligible for promotion (the from-below climb).
    promote gate fires + cap needs a raise -> CAP_VERIFY
    promote gate fires + cap already covers -> PROMOTION_VERIFY
    binding/pressure                        -> (cap can relieve) CAP_VERIFY
                                               (cap pinned)      DEMOTE_REQUIRED

CAP_VERIFY
    Cap was raised (to relieve pressure, or to pre-fund a promotion).
    Verify Kverify windows:
      retry / pressure persists -> DEMOTE_REQUIRED
      clean                     -> STABLE   (a pre-funded promotion then
                                             re-enters via STABLE's promote gate,
                                             now on the "cap already covers" edge)

PROMOTION_VERIFY
    Exactly one block promoted to resident.
    Ignore the first (cold) window -- the new layout's first step re-primes.
    Verify Kverify stable windows:
      dirty -> rollback the block, COOLDOWN
      clean -> STABLE

DEMOTE_REQUIRED
    Demote one canonical block as a transaction (ring resize).
    Invalidate old-layout measurements (fragmentation/peak changed).
    -> COLD

COOLDOWN
    Prohibit re-promotion for N windows (so a rolled-back block can't
    immediately re-promote and re-thrash). Demote still allowed.
    -> STABLE after N clean windows.
```

Why the phases are split rather than one "adjust" step:

- **CAP_VERIFY vs PROMOTION_VERIFY are separate** because the cap raise and the
  residency change are two different-cost, differently-reversible acts, and the
  GC lag means the cap's effect isn't observable until the step after it's set.
  Serializing them (raise cap -> verify clean -> then promote -> verify clean)
  is the only ordering where each act is confirmed before the next is committed.
- **DEMOTE_REQUIRED -> COLD, but PROMOTION_VERIFY-dirty -> COOLDOWN.** A demote
  under external pressure changes the whole layout/fragmentation profile, so all
  measurements are stale (COLD). A failed promotion just rolls back to the prior
  known-good layout, which is still characterized -- so it skips COLD but is
  barred from immediately retrying (COOLDOWN).
- **COLD gates on compile/retrace**, tying into the compile-offload seam: a
  recompile or graph-generation change moves the memory profile, so any counter
  read across it is meaningless until fresh windows accrue.

Implement as a **pure** `residency_fsm_step(state, signals) -> (state, action)`
(no CUDA), so the whole hysteresis is unit-testable against synthetic window
sequences -- the controller convention in this repo (GPU CI does not exist).

## Honesty notes (do not oversell)

- The cap lever is cheaper than a demote but **not free**: lowering it triggers
  an all-or-nothing idle-cache sweep (~80 ms typical, seconds under external VRAM
  pressure). "Cheap" means *no prefetch destruction and reversible*, not zero.
- **Both levers are phase-boundary-bound.** The reclaim sync stalls in-flight
  transfer-stream work, so neither runs per-step. The inner loop is finer and
  reversible, not continuous -- keep both off the per-step path.
- Bias high in *residency* is wrong (previous discussion corrected): we bias the
  cap high/reversible and residency **low/sticky**.

## Implementation sketch (where the code goes)

1. **Pure policy helpers in `vram_budget.py`** (unit-testable, no CUDA):
   - `allocator_allowance_bytes(cap, live)` = `0.95*cap - live`, and its inverse
     `cap_bytes_for_live(planned_live, cache_budget, cliff_cap, ...)` =
     `(live + cache_budget)/0.95` clamped to the cliff -- the two carry the 0.95
     factor so no call site open-codes it.
   - `cap_can_host_promotion(live, block, slack_pad, cliff_cap)` -> bool (the
     precheck, with the `/0.95` divisor).
   - `residency_promote_ok(num_alloc_retries, reclaimable_at_peak, block, slack_pad)`
     -> bool (sampling climb gate).
   - `residency_fsm_step(state, signals) -> (state, action)` -- the hysteresis
     FSM above, pure. `signals` carries the per-window counters + gate verdicts;
     `action` in {hold, raise_cap, promote, demote, rollback}.
   - Training variant keyed on the cohabitation watermark (reuse
     `sampling_guard_predicted_peak_free` / `training_cliff_predicted_peak_free_gib`).
2. **Controller wiring in `manager.py`** at the existing phase-boundary hook
   where the live controllers already run (same seam as keep_last/working_reserve
   in `BaseSDTrainProcess`). Start residency at N-1; run the two-tier loop.
3. **Slack pad + block size** as measured inputs: `block_bytes` from the layout
   (per-model), `slack_pad` from the last `--cap-descent` (default 0.25 GiB, one
   notch, until a descent refines it). No env vars for the live policy (config
   only, per the training-config rule); a debug env override is fine for tests.
4. **Descent as recalibration:** expose the cap-descent floor + knee so the
   controller can seed `cliff_bound`/`slack_pad` from the most recent run instead
   of a constant.

## Testing

- **CPU/sim** (no GPU CI): unit-test the pure helpers against the measured
  numbers -- feed 6.77 live / 7.35 floor / 0.21 slack / 0.375 block and assert
  the gate/precheck decisions match the validated knee. Add a small state-machine
  test that drives promote -> saturate -> escalate -> settle and asserts at most
  one demote (no oscillation).
- **GPU** (short synthetic, `venv/Scripts/python.exe`, per the cuda-testing
  methodology): re-run `--cap-descent` after a residency bump to confirm the
  floor holds under the higher residency, and watch `num_alloc_retries/step ~= 0`
  as the live guardrail. No full training run required for the sampling half.

## Open questions

- **Slack pad sizing for training.** Sampling's 0.21 is rock-steady; training's
  noisy peak needs a wider pad. Derive it from the per-step variance in the
  descent, or start at a conservative multiple of a block and let the from-below
  climb find the ceiling.
- **FSM constants.** `Kclean`, `Kverify`, `N` (cooldown), and the
  ignore-first-window rule (see the state machine) need tuning against real
  window traces -- start conservative (Kverify >= 2 so the lagged signal is seen)
  and relax if convergence is too slow.
- **Interaction with pinned-arena.** The descent was run without `--pinned-arena`
  (fresh pins per sample -> DXGI yoyo, harmless to the dedicated-side
  measurement). Confirm the floor/knee under a persistent arena before trusting
  the numbers for arena runs.
