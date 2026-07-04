# Prefetch 2.0 Implementation Plan

> **git-bug:** `ee6fa64` (open) — tune thresholds from real trace captures.
> Core landed under `4bfb44c` (closed) — durable execution trace implemented.
> Status lives in the tickets; this file is the plan.

## Goal

Stop treating trace invalidation as the default recovery path. The memory
manager should be conservative about VRAM safety, but opportunistic about
prefetch hints.

Core invariant:

```text
Execution trace is durable and stores ordered semantic accesses.
Transfer plan is disposable.
Memory budget is worst-case.
```

## Implementation Status

Implemented in the current slice:

- Semantic trace schedules now use `(layer_key, operation, occurrence)` entries.
- Trace lifecycle APIs are split into `mark_transfer_plan_dirty` and `invalidate_execution_trace`.
- Layout/residency changes mark transfer planning dirty without clearing frozen execution traces.
- Resolution changes can reuse compatible traces when the execution-order policy matches.
- Compatible fallback is blocked per shape after a bad provisional trace, without poisoning other shapes.
- `PinnedBouncePool` performs local semantic resync for both transfer and resident-skip paths.
- Source registration can be refreshed after promotion/demotion without clearing schedules.
- Prefetch diagnostics report confidence, resync/mismatch rates, duplicate-key blocks, and invalid-trace thresholds.
- Resident growth is gated on validated exact/observed prefetch health; compatible/cold traces may prefetch but cannot authorize promotion.
- Autotune repairs missing or invalid prefetch traces before waiting on memory-layout cadence, except for urgent demotion.
- Offline replay harness added at `scripts/replay_prefetch_trace.py` for real access-stream threshold tuning.
- Real runs can emit replay JSONL through job config: `model.layer_offloading_prefetch_trace_capture: path/to/capture.jsonl`. `AI_TOOLKIT_BOUNCE_TRACE_CAPTURE` remains a low-level fallback.

Remaining follow-up:

- Capture real training streams with `model.layer_offloading_prefetch_trace_capture`, then feed them into `scripts/replay_prefetch_trace.py`.
- Tune production thresholds from those replay results.
- Consider reducing lookahead for compatible traces if pinned CPU pressure appears.


## Current Hot Spots

- `manager_modules.py`: `_OffloadTrace`
- `manager_modules.py`: `mark_transfer_plan_dirty` / `invalidate_execution_trace`
- `bounce_pool.py`: positional schedule state
- `bounce_pool.py`: `PinnedBouncePool.acquire`
- `bounce_pool.py`: `PinnedBouncePool.consume_without_transfer`
- `manager.py`: `MemoryManager.offload_step_begin`

## Design Rules

1. Residency changes must not clear execution traces.

   Promotion and demotion change whether a layer needs transfer, not whether the
   layer is executed.

2. Resolution changes should prefer compatible stale traces over cold
   no-prefetch.

   A stale trace is a hint. It may cause wasted copies, but it should avoid
   catastrophic hard-miss storms.

3. Trace health must not grant memory safety.

   Provisional or stale traces can drive prefetch, but cannot authorize resident
   growth, lower `working_reserve`, or reduce `wddm_margin`.

4. Cursor mismatch should resync locally.

   A single positional mismatch should not poison the rest of the macro-step.

5. Resync must not match by raw layer key alone.

   Layer keys repeat within a macro-step: forward, recompute, backward, DOP, and preservation paths may touch the same module more than once. A resync that jumps from a forward occurrence to a later backward occurrence silently corrupts the cursor. Resync must use a semantic access key.

6. Transfer state should be rebuilt from:

   ```text
   execution_trace + current resident_set + gpu_ring_state + pool_budget
   ```

## Phase 1: Stop Destructive Trace Resets

Split trace lifecycle APIs so transfer-plan dirtiness is not named like trace invalidation.

Current behavior:

```text
layout change
-> clear schedule_by_shape_key
-> bump trace version
-> bounce pool loses schedule
-> cold miss storm
```

New behavior:

```text
layout change
-> keep execution traces
-> bump transfer-plan epoch only
-> mark current schedule provisional/dirty
-> remove stale ready slots narrowly, if necessary
-> continue using trace order
```

Implementation direction:

- Implemented API split:

  ```python
  mark_transfer_plan_dirty(reason: str = "")
  invalidate_execution_trace(reason: str = "")
  ```

- Layout promote/demote calls `mark_transfer_plan_dirty`, not full trace
  invalidation.
- Full execution invalidation should be reserved for checkpoint policy changes,
  DOP path changes, model layout changes, or repeated severe divergence.

## Phase 2: Split Execution Order Key From Memory Profile Key

Shape keys are useful, but prefetch order and memory safety should not be forced
to use the same cache key.

Introduce:

```python
execution_order_key = (
    mode,
    model_layout_version,
    checkpoint_signature,
    precision_signature,
    preservation_path_signature,
    effective_microbatch,
    dop_policy_signature,
)

memory_profile_key = (
    execution_order_key,
    latent_shape,
    dop_shape_or_none,
    batch_size,
)
```

Policy:

```text
Exact execution_order_key trace:
  use normally

Compatible execution_order_key trace from another memory_profile_key:
  use provisionally

No compatible trace:
  use observed/cold registration fallback
```

Memory estimates remain per `memory_key`. Resident growth is gated by the worst
validated active or configured `memory_key`.


## Phase 2A: Store Semantic Access Keys

Trace schedules must not be raw layer keys. Use an access key that distinguishes repeated visits to the same layer within a macro-step.

Preferred representation:

```python
access_key = (
    layer_key,
    operation,
    occurrence_index,
)
```

At minimum:

```python
access_key = (
    layer_key,
    operation,
)
```

`occurrence_index` is the monotonically increasing occurrence for this `layer_key` within the recorded macro-step. It prevents a forward mismatch from resyncing to a later backward or recompute occurrence of the same layer.

For the first implementation slice, use exact semantic access-key matching for resync. Same-layer-only matching should be counted as a weak candidate but should not move the cursor until diagnostics prove it is safe.

## Phase 3: Make BouncePool Resync Instead Of Cascade-Missing

Add local semantic alignment to both paths:

- `PinnedBouncePool.acquire`
- `PinnedBouncePool.consume_without_transfer`

Conceptual helper:

```python
def _align_locked(access_key, window):
    pos = self._consume_pos

    if pos < len(self._scheduled) and self._scheduled[pos].access_key == access_key:
        return pos, "aligned"

    end = min(len(self._scheduled), pos + 1 + window)
    for j in range(pos + 1, end):
        if self._scheduled[j].access_key == access_key:
            return j, "resynced"

    return pos, "mismatch"
```

On resync:

```text
mark skipped positions stale
discard ready slots for skipped positions
set consume_pos to aligned position + 1
increment resync counter
do not clear schedule
```

On mismatch:

```text
demand-load this access
record mismatch
continue
```

Do not resync by `layer_key` alone. If the same layer appears nearby but the semantic access key does not match, count `duplicate_key_resync_blocked` and keep the schedule provisional or mismatched.

Only re-record if mismatch rate is high across a step or repeated steps.

## Phase 4: Treat Resident Skips As Validated Trace Consumption

`consume_without_transfer(layer_key)` should not blindly advance by one. It
should use the same alignment logic as `acquire`.

Reason: after promotion, resident layers may skip transfer. If skip-path cursor
movement is wrong, it silently corrupts the whole positional schedule.

Expected behavior:

```text
resident layer appears where expected:
  consume position

resident layer appears nearby:
  resync and consume

resident layer not found nearby:
  count mismatch, consume conservatively
```

Also remove ready/filling work for positions that become resident if it is still
safe to discard.

## Phase 5: Add Provisional Trace Health

Track schedule confidence in the `BouncePool` and trace handoff:

```python
schedule_confidence = "exact" | "compatible" | "observed" | "cold"
```

Diagnostics should include:

```text
schedule_confidence
resyncs
mismatches
stale_positions
provisional_hit_rate
exact_shape_key / source_shape_key
```

Control policy:

```text
exact:
  normal lookahead

compatible/provisional:
  allow prefetch
  maybe smaller lookahead initially
  block resident growth

observed:
  allow prefetch
  validate against new recording

cold:
  minimal fallback
```

## Phase 6: Safety Gates

Resident promotion should require:

```python
can_promote = (
    current_memory_profile_key_has_valid_estimate
    and all_active_memory_profile_keys_have_valid_estimate
    and predicted_free_for_worst_key_after_promotion >= wddm_margin
    and not in_wddm_emergency_retreat
    and trace_health_not_catastrophic_for_current_execution_order
)
```

But prefetch should require only:

```python
can_prefetch = (
    pool_budget_available
    and host_ram_floor_ok
    and not in_wddm_emergency_retreat
)
```

This asymmetry is intentional: a bad prefetch hint can waste CPU/pinned work,
but a bad memory assumption can spill WDDM and wreck the run.

## Tests To Add

1. `test_resident_promotion_does_not_clear_execution_trace`

   Freeze a trace, simulate promote/demote, assert `schedule_by_shape_key`
   survives, and assert the pool schedule can still be handed off.

2. `test_resolution_change_uses_compatible_trace_provisionally`

   Freeze a 512 trace, begin a 768 step with no exact trace, and assert the
   schedule is reused with provisional/compatible confidence.

3. `test_acquire_resyncs_near_future_match`

   Schedule `a,b,c,d`, access `b`, and assert the cursor realigns instead of
   hard-missing the rest.

4. `test_consume_without_transfer_resyncs_like_acquire`

   Same as above, but through the resident skip path.

5. `test_resync_does_not_jump_to_later_duplicate_layer`

   Schedule semantic accesses like `forward:a`, `forward:b`, `backward:a`, `backward:b`. Observing `forward:b` should resync to `forward:b`, not `backward:b`. Observing `backward:a` should resync only when the observed access key includes the backward context.

6. `test_layout_change_rebuilds_transfer_plan_not_trace`

   Add ready slots, mark transfer dirty, and assert the trace remains while
   stale ready slots are discarded safely.

7. `test_provisional_trace_does_not_authorize_resident_growth`

   Compatible trace exists, memory key is unknown, and prefetch is allowed while
   promotion is denied.

## Implementation Order

1. Rename or split the destructive trace reset API.
2. Preserve traces across promote/demote.
3. Add schedule confidence metadata.
4. Add a resync helper in `BouncePool`.
5. Apply resync to `acquire`.
6. Apply resync to `consume_without_transfer`.
7. Add diagnostics.
8. Add safety gate tests.
9. Tune thresholds: resync window, mismatch threshold, provisional lookahead.

## Shippable Default

```text
Unknown order:
  use best compatible trace as prefetch hint

Unknown memory:
  assume worst-case reserve and block resident growth

Layout changed:
  keep execution trace, rebuild transfer plan

Trace mismatch:
  resync locally, demand-load this access, re-record only if mismatch rate stays high
```

This should address the current performance cliff directly: resolution changes
and resident-set changes no longer collapse the prefetcher into a cold-start
storm.
