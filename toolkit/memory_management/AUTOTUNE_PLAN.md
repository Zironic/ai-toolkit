# Dynamic headroom + keep_last autotune — implementation plan

Goal: stop hand-picking `layer_offloading_smart_headroom_gb` and
`layer_offloading_checkpoint_keep_last`. Tune the split between
**resident weights**, **streaming**, and **activations** automatically, per
resolution, on any card — while never triggering the WDDM shared-memory spill.

## The governing principle: asymmetric loss

The failure modes are lopsided, so the controller must be lopsided too:

- **Headroom too small → physical VRAM fills → WDDM spills to shared system
  memory → ~30x slowdown (the "explosion").** Catastrophic.
- **Headroom too big → we waste ~1-2 GB of VRAM** but throughput is unaffected
  (still ~60% power, more streaming than necessary). Cheap.

**We would rather permanently waste 1 GB than risk one spill.** Every design
choice below favors safety over reclaiming the last GB. AIMD discipline:
additive (one smallest layer) increase toward resident, multiplicative
(several layers) decrease on any breach.

## The real ceiling is measured, not 12 GiB

Windows never lets a display-attached GPU's physical VRAM exceed ~total minus a
~500 MB system reserve (DWM desktop composition + WDDM driver overhead). That
reserve is **not fixed** — opening a browser / adding a monitor grows the
desktop footprint and lowers our ceiling mid-run.

So **never target a hardcoded total.** Target driver-level free memory, which
already accounts for everyone (us + DWM + other processes).

`_cuda_memory` ([manager.py](manager.py)) already reads `mem_get_info`, exposing
driver-level `device_used_gb` / `device_free_gb` in each perf-log record. The
spill cliff is exactly when driver-level **free → 0**.

**Invariant:** keep peak `device_free` ≥ `BUFFER`, i.e.
`total − (peak_torch_reserved + system_reserve) ≥ BUFFER`, where
`system_reserve = device_used − torch_reserved` is measured live.

- `system_reserve` tracked as a **running max + its own cushion** (it is
  outside our control and volatile).
- If `system_reserve` jumps mid-run (desktop activity), peak `device_free`
  shrinks → controller treats it exactly like a spill breach → hard retreat to
  restore the buffer.

Targets:
- Hard floor: never let peak `device_free` cross `BUFFER_HARD` (≈ 1.0 GB).
- Soft stop-line: settle while peak `device_free` ≥ `BUFFER_STOP` (≈ 1.5 GB),
  leaving a full cushion between the resting point and the cliff.

## Architecture: two staged controllers, not one concurrent loop

keep_last↑ and headroom↓ both raise reserved toward the same ceiling. Run
concurrently they oscillate. Solution: **sequence them**, exploiting that one
self-limits and the other is greedy.

1. **Stage A — keep_last first (self-limiting).** Its benefit (less recompute)
   flattens then reverses as the backward-fetch tax grows, so climbing it *on
   step time* stops on its own at a speed optimum, leaving predictable slack.
2. **Stage B — headroom second (greedy).** Pulling layers resident removes
   streaming and keeps paying until everything is resident; it will eat VRAM to
   the floor, so it must run last and soak up only the remainder.

Headroom-first would make everything resident and starve keep_last. KL-first is
correct.

## Stage A — keep_last retarget (speed-based)

Retarget the existing `CheckpointKeepLastAutotuner`
([checkpoint_autotuner.py](checkpoint_autotuner.py)) from "largest keep_last
that fits in reserved" to **"keep_last that minimizes windowed step time."**

- Per resolution bucket (already structured that way).
- Climb keep_last while windowed-average step (or backward) time improves;
  stop / step back when it flattens or regresses (the compute_wait + backward
  ring-reuse-miss tax we observed: 19-21 misses, compute_wait up to 1.5s).
- Keep the reserved spill-guard as a **hard ceiling**, not the objective.
- Hill-climb on a **windowed average with hysteresis** — single-step noise
  (512 steps bounced 5.8-7.0s) must not trigger a move.
- Mark the bucket `settled` when at optimum; Stage B only starts for a bucket
  after Stage A has settled it.

## Stage B — conservative headroom controller (AIMD)

Only runs after Stage A has settled. Walks the offload boundary (resident set)
down while the driver-free invariant holds.

1. **Look before you leap.** We know each offloaded layer's byte size, so
   *predict* the reserved bump before committing. Refuse any promotion where
   `peak_torch_reserved + layer_bytes + cache_pad + system_reserve` would land
   inside the stop-line. Observed `device_free` only *validates*; we never
   intentionally step over the edge.
2. **Smallest layer, slow cadence.** Promote one *smallest* eligible offloaded
   layer at a time (finest control), then wait several windows to confirm
   `device_free` held before the next.
3. **Worst-case resolution governs.** Track `max(peak_torch_reserved)` /
   `min(device_free)` across **all** buckets, never the current one — a 256
   step must not authorize a headroom that explodes on the next 512.
4. **Hard retreat + danger memory.** If observed peak `device_free` ever crosses
   `BUFFER_HARD`, immediately re-offload *more* than was promoted (undo 2-3
   layers) and record that point as a ceiling — never promote within a cushion
   of it again for the rest of the run. One near-miss locks out the danger zone.
5. **Stop when it stops paying.** If promoting a layer doesn't improve windowed
   step time, **stop and leave the VRAM free.** Don't spend memory or risk
   reclaiming slack that buys no throughput — may mean it barely moves on a
   given card. That's the desired conservative outcome.

## Coupling to exploit (second cut, not first)

keep_last's backward-fetch tax exists *because the kept blocks' weights are
offloaded* (backward must fetch them → the ring-reuse misses). If Stage B
preferentially makes the **last-k blocks resident**, that tax vanishes (no
recompute *and* no backward fetch), and keep_last's optimum shifts up. Turns the
two controllers from competitors into reinforcers. First cut can ignore this; it
is where the real win is.

## The one genuinely new mechanic: live re-plan

`smart_training_plan` ([manager.py L574](manager.py)) runs **once at attach** and
never moves the offload boundary again. Stage B needs to **promote/demote
individual layers mid-run** (move a few weights GPU↔CPU+pinned, flip the
`_layer_memory_manager` offload flag). `attach` does this en masse; a bounded
per-layer version is the new, riskiest code.

- Only mutate at a **step boundary**, never mid-prefetch.
- Move **incrementally** (a few layers per adjustment) so the spill-guard
  catches overshoot.
- Coordinate with the bounce pool / ring state when a layer changes class.

Precedent for the measure→derive→re-plan loop already exists for **sampling**:
`_sampling_peak_headroom_bytes` learned and fed as `max(default, learned)` into
the next plan ([manager.py L1026-1147](manager.py)). Stage B is the training,
within-run equivalent.

## Wiring / config

- `layer_offloading_smart_headroom_gb = -1` → auto (Stage B active); a positive
  value remains a manual override / seed.
- `layer_offloading_checkpoint_keep_last = -1` → auto (Stage A active) — already
  wired.
- Constants (env-overridable, conservative defaults):
  `BUFFER_HARD ≈ 1.0 GB`, `BUFFER_STOP ≈ 1.5 GB`, `cache_pad`, system-reserve
  cushion, window size, settle thresholds, retreat layer count.
- Seed Stage B from a **generous** headroom so Stage A climbs safely first.

## Files

- `toolkit/memory_management/checkpoint_autotuner.py` — retarget Stage A to
  windowed step-time objective + settle flag.
- `toolkit/memory_management/headroom_autotuner.py` (new) — Stage B controller:
  driver-free invariant, predict-before-commit, AIMD, danger memory, worst-case
  bucket tracking, stop-when-no-speedup.
- `toolkit/memory_management/manager.py` — per-layer promote/demote (live
  re-plan); expose offloaded-layer sizes; surface `device_used`/`device_free`
  peaks to the tuner.
- `extensions_built_in/sd_trainer/SDTrainer.py` — drive both tuners at step
  boundaries (mirror existing `_checkpoint_autotuner` wiring); feed windowed
  step time + per-bucket reserved/device_free peaks.
- `toolkit/config_modules.py` — accept `-1` for headroom.
- UI (`SimpleJob.tsx`, `types.ts`, `utils.ts`, `docs.tsx`) — allow `-1` (auto)
  on the headroom field, like keep_last.

## Validation

CPU-controller unit test first (as with `checkpoint_autotuner`): feed synthetic
per-window reserved/device_free + step-time series, assert it (a) never crosses
BUFFER_HARD, (b) retreats hard on a simulated system-reserve spike, (c) stops
when step time flattens. Then GPU-validate via the venv methodology
(venv\Scripts\python.exe) on the Krea offload run.

## Status / observed data motivating this

From `output/LA Jinx Krea DOP phrasal` (run digested with
`scripts/digest_perf_log.py`): `headroom=0.90/7.00` used while
`peak_reserved` 9.24→10.03 on a 12 GiB card — ~2 GB idle behind ~6 GB of unused
headroom budget. Largest payoff case seen so far. Also flagged separately:
BouncePool hit-rate regressed to ~45% (was 100%) with copy throughput halved —
investigate independently of this plan.
