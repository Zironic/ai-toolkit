#!/usr/bin/env python3
"""Offline simulator for the auto-working_reserve training memory controller.

Tuning the live controller by running real training jobs is slow. This models
the *control dynamics* — how the resident set, peak_reserved, and available VRAM
evolve as the controller promotes/demotes layers — without a GPU, so the band,
safety margin, cadence, and retreat size can be tuned in seconds.

Fidelity: the simulator calls the **real** decision functions
``MemoryManager._available_vram_gib`` (measurement) and
``MemoryManager._training_layout_action`` (deadband), so the logic under test is
exactly production. Only the *physics* are modelled here:

    peak_reserved(bucket) = working_floor[bucket] + resident_layers * layer_gib
    other                 = context_gib + external_gib(step)      # not our allocator
    reserved              = peak_reserved        (steady: current ≈ peak)
    device_used           = reserved + other
    device_free           = total - device_used
    spill (OOM)           = peak_reserved + other > total

``working_floor`` is everything resident that the autotuner does NOT control
(activations, ring/bounce buffers, always-resident weights, and the roughly
constant transient footprint of streamed layers). Promoting a layer moves it
from briefly-streamed to permanently-resident, so peak_reserved rises by ~its
size — matching the real run (reserved climbed 6.4→9.1 GiB as resident grew).

These numbers are knobs: the defaults are calibrated loosely to the Krea run
(16 GiB card, ~1.9 GiB context, res512 floor a bit above res256). Adjust them
to match your card, then read the trace / assertions.

Usage:
    python scripts/sim_working_reserve_controller.py                 # default scenario, trace
    python scripts/sim_working_reserve_controller.py --scenario pressure
    python scripts/sim_working_reserve_controller.py --list
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolkit.memory_management import MemoryManager


@dataclass
class SimConfig:
    total_gib: float = 16.0
    layer_gib: float = 0.08            # per autotuner-controlled layer (~96 MiB fp8)
    n_layers: int = 142               # offloadable layers the autotuner can promote
    always_resident_gib: float = 0.5  # non-offloadable weights + must-resident
    context_gib: float = 1.9          # our CUDA context / cuDNN / compiled-graph overhead
    # Per-bucket working floor EXCLUDING autotuner-promoted layers.
    working_floor_gib: dict = field(
        default_factory=lambda: {"res512": 6.6, "res256": 5.9}
    )
    # Controller knobs (mirror the AI_TOOLKIT_TRAINING_* env defaults).
    safety_gib: float = 0.5
    wddm_hard_gib: float = 1.0
    wddm_hold_high_gib: float = 2.0
    retreat_layers: int = 3
    promote_interval: int = 4
    start_resident: int = 0
    # "single": promote one layer per cadence (current live behavior — slow).
    # "batch": promote enough layers to jump to mid-band in one move (proposed:
    #   one layout change = one trace reset, converges in ~1-2 steps).
    promote_strategy: str = "single"
    # Per-step memory noise (calibrated loosely to the Krea logs: within-bucket
    # peak_alloc jitter was ~±0.1-0.15 GiB). All default to 0 -> deterministic.
    act_noise_gib: float = 0.0       # std of activation/tensor-shape jitter
    act_spike_prob: float = 0.0      # chance of a large-aspect-ratio batch
    act_spike_gib: float = 0.0       # extra activation on a spike step
    frag_gib: float = 0.0            # typical allocator fragmentation slack (reserved-allocated)
    frag_noise_gib: float = 0.0      # std of fragmentation jitter
    other_noise_gib: float = 0.0     # std of OS/other-process random walk
    seed: int = 0
    # EMA smoothing on the governing free signal (1.0 = off/instantaneous). Lower
    # = steadier under noise but slower to react to real sustained pressure. The
    # spill/OOM path always uses the instantaneous reading, so safety is immediate.
    smooth_alpha: float = 1.0
    # --- Manual working_reserve + cliff-guard scenario (run_manual_sim) ---
    # Manual mode does NOT auto-tune the budget; the layout is fixed. Instead we
    # model the caching allocator hoarding idle blocks: under multi-resolution
    # bouncing freed blocks do not coalesce, so `reserved` ratchets to its
    # high-water mark and never falls back — exactly the observed climb to 11.48
    # GiB / device_free=0 on the 12 GiB Krea run. The guard reclaims it.
    frag_ratchet_gib: float = 0.0     # per-step fragmentation creep toward the hoard ceiling
    retreat_layers_manual: int = 3    # layers the guard demotes per round if empty_cache is not enough
    manual_residual_gib: Optional[float] = None  # step-end trough after activations free
    # --- Working_reserve SIZING controller (run_reserve_sim) ---
    # This is the *other* auto controller: it sizes the activation budget
    # (working_reserve) via _training_working_reserve_decision / _signal, rather
    # than moving layers. The planner packs resident weights to fill whatever the
    # reserve leaves, so the spill margin follows directly:
    #     device_free(bucket) = working_reserve + wddm_margin - activation_peak(bucket)
    start_reserve_gib: float = 7.0    # generous seed (auto starts high, climbs down)
    # Spare free NOT tied to the reserve, modelling resident weights sitting at the
    # floor (cold-start / prefetch-locked): the card has lots of headroom the
    # planner has not yet packed, so a reserve set below the activation peak does
    # NOT spill -- and therefore never triggers a retreat to climb it. This is the
    # live failure mode; the grow branch must climb the reserve on the MEASUREMENT
    # alone. 0.0 = the original "planner packs residents to fill" physics.
    free_offset_gib: float = 0.0
    wddm_margin_gib: float = 1.5
    wddm_stop_gib: float = 1.5
    pad_gib: float = 0.5
    step_gib: float = 0.5
    retreat_gib: float = 1.0
    stable_windows: int = 2
    min_working_reserve_gib: float = 1.5
    # Per-bucket TRUE within-step activation peak (what must fit under the reserve).
    act_peak_gib: dict = field(
        default_factory=lambda: {"res768": 5.2, "res512": 4.46, "res256": 3.3}
    )
    # Step-end residual (the misleading trough the old code measured instead).
    act_residual_gib: float = 1.1


@dataclass
class StepRecord:
    step: int
    bucket: str
    resident: int
    peak_reserved: float
    other: float
    available: float
    governing: float
    move: str
    spilled: bool


def run_sim(
    cfg: SimConfig,
    sequence,
    steps: int,
    external_gib: Optional[Callable[[int], float]] = None,
):
    """Drive the real controller decisions over a modelled card.

    ``sequence`` is the list of resolution buckets to cycle through (e.g.
    ["res512", "res256"]). ``external_gib(step)`` returns extra VRAM held by
    other processes at that step. Returns the list of StepRecord.
    """
    import random

    external_gib = external_gib or (lambda _step: 0.0)
    rng = random.Random(cfg.seed)
    resident = int(cfg.start_resident)
    last_available: dict = {}
    last_promote_step = -10_000
    other_walk = 0.0
    history = []

    for step in range(steps):
        bucket = sequence[step % len(sequence)]
        # Activations: per-bucket floor plus tensor-shape jitter and the rare
        # large-aspect-ratio batch.
        activations = cfg.working_floor_gib[bucket]
        if cfg.act_noise_gib:
            activations += rng.gauss(0.0, cfg.act_noise_gib)
        if cfg.act_spike_prob and rng.random() < cfg.act_spike_prob:
            activations += cfg.act_spike_gib
        activations = max(0.0, activations)
        allocated_peak = (
            cfg.always_resident_gib + resident * cfg.layer_gib + activations
        )
        # Caching-allocator fragmentation: reserved sits above allocated, and that
        # slack itself jitters as block reuse changes.
        frag = cfg.frag_gib
        if cfg.frag_noise_gib:
            frag += rng.gauss(0.0, cfg.frag_noise_gib)
        reserved = allocated_peak + max(0.0, frag)
        peak_reserved = reserved          # reset each step -> this step's peak
        # OS / other-process pressure as a mean-reverting random walk so a stray
        # compositor/browser allocation nudges `other` without drifting forever.
        if cfg.other_noise_gib:
            other_walk = 0.85 * other_walk + rng.gauss(0.0, cfg.other_noise_gib)
        other = max(0.0, cfg.context_gib + float(external_gib(step)) + other_walk)
        device_used = reserved + other
        spilled = device_used > cfg.total_gib

        available = MemoryManager._available_vram_gib(
            cfg.total_gib,
            device_used,
            reserved,
            peak_reserved,
            safety_gib=cfg.safety_gib,
        )
        # Optionally smooth each bucket's free signal so a single noisy spike does
        # not trip a (trace-resetting) layout move. Spill detection above stays on
        # the raw reading, so real over-commit is still caught immediately.
        if cfg.smooth_alpha < 1.0:
            prev = last_available.get(bucket, available)
            available = cfg.smooth_alpha * available + (1.0 - cfg.smooth_alpha) * prev
        last_available[bucket] = available
        # Govern by the worst-case recent free across buckets (high-res binds).
        governing = min(last_available.values())

        move = MemoryManager._training_layout_action(
            governing,
            wddm_hard_gib=cfg.wddm_hard_gib,
            wddm_hold_high_gib=cfg.wddm_hold_high_gib,
            did_oom=spilled,
        )

        if move == "down":
            resident = max(0, resident - cfg.retreat_layers)
        elif move == "up":
            cadence_ready = step - last_promote_step >= cfg.promote_interval
            if not (cadence_ready and resident < cfg.n_layers):
                move = "wait_cadence" if resident < cfg.n_layers else "maxed"
            elif cfg.promote_strategy == "batch":
                # Predict-before-commit: promote enough layers to land near the
                # middle of the hold band in a single move (one trace reset).
                mid_band = (cfg.wddm_hard_gib + cfg.wddm_hold_high_gib) / 2.0
                want = int(max(1, (governing - mid_band) / cfg.layer_gib))
                want = min(want, cfg.n_layers - resident)
                resident += want
                last_promote_step = step
                move = f"up x{want}"
            else:
                resident += 1
                last_promote_step = step
        # "hold": no move, no trace reset — the steady state we want.

        history.append(StepRecord(
            step, bucket, resident, peak_reserved, other,
            available, governing, move, spilled,
        ))

    return history


def run_timing_spill_sim(
    cfg: SimConfig,
    sequence,
    steps: int,
    *,
    hidden_cliff_gib: float,
    slowdown_ratio: float = 5.0,
):
    """Layout sim where WDDM reveals itself as a timing cliff, not an OOM."""
    resident = int(cfg.start_resident)
    last_available: dict = {}
    best_time = None
    learned_hard = None
    history = []

    for step in range(steps):
        bucket = sequence[step % len(sequence)]
        activations = cfg.working_floor_gib[bucket]
        peak_reserved = cfg.always_resident_gib + resident * cfg.layer_gib + activations
        other = cfg.context_gib
        device_used = peak_reserved + other
        available = MemoryManager._available_vram_gib(
            cfg.total_gib,
            device_used,
            peak_reserved,
            peak_reserved,
            safety_gib=cfg.safety_gib,
        )
        last_available[bucket] = available
        governing = min(last_available.values())

        base_time = 10.0
        step_time = base_time * slowdown_ratio if available < hidden_cliff_gib else base_time
        learned = MemoryManager._training_timing_spill_floor(
            step_time,
            best_time,
            available,
            steps=step + 1,
            warmup_steps=4,
            slowdown_ratio=3.0,
            max_signal_free_gib=cfg.wddm_hold_high_gib,
            pad_gib=0.25,
        )
        if learned is not None:
            learned_hard = max(learned_hard or 0.0, learned)
        effective_hard = max(cfg.wddm_hard_gib, learned_hard or 0.0)

        move = MemoryManager._training_layout_action(
            governing,
            wddm_hard_gib=effective_hard,
            wddm_hold_high_gib=max(cfg.wddm_hold_high_gib, effective_hard + cfg.layer_gib),
            did_oom=False,
        )
        if move == "down":
            resident = max(0, resident - cfg.retreat_layers)
        elif move == "up":
            resident = min(cfg.n_layers, resident + 1)
        if best_time is None or step_time < best_time * 0.98:
            best_time = step_time

        history.append(StepRecord(
            step,
            bucket,
            resident,
            peak_reserved,
            other,
            available,
            learned_hard or cfg.wddm_hard_gib,
            "timing_spill" if learned is not None else move,
            False,
        ))

    return history


def run_manual_sim(
    cfg: SimConfig,
    sequence,
    steps: int,
    *,
    guard: bool = True,
    external_gib: Optional[Callable[[int], float]] = None,
):
    """Manual working_reserve: fixed layout, fragmentation ratchets `reserved` up,
    and the real cliff guard reclaims idle cache / demotes when driver-free
    crosses the hard floor.

    Unlike :func:`run_sim` there is no band-driven promote/demote — manual mode
    keeps the budget the user picked. The only physics that move are:

        allocated_peak  = always_resident + resident*layer + activations(bucket)
        reserved_held   = max(reserved_held, allocated_peak + frag)   # never falls back
        reserved_held  += frag_ratchet_gib                            # multi-res hoard creep
        device_free     = total - (reserved_held + other)

    With ``guard=False`` this reproduces the observed blow-up: ``reserved`` climbs
    to the ceiling and every step spills. With ``guard=True`` the step boundary
    runs ``MemoryManager._training_cliff_guard_action`` — the *same* predicate the
    live safety net uses — and on "reclaim" first drops ``reserved_held`` back to
    the live footprint (``empty_cache``) and then, only if free is still under the
    floor, demotes the largest resident layers.
    """
    import random

    external_gib = external_gib or (lambda _step: 0.0)
    rng = random.Random(cfg.seed)
    resident = int(cfg.start_resident)   # fixed in manual mode (the user's plan)
    reserved_held = 0.0                   # allocator high-water mark (idle-cache hoard)
    history = []
    shape_peaks = {}

    def footprint(res_layers, activations):
        return cfg.always_resident_gib + res_layers * cfg.layer_gib + activations

    for step in range(steps):
        bucket = sequence[step % len(sequence)]
        activations = cfg.working_floor_gib[bucket]
        if cfg.act_noise_gib:
            activations += rng.gauss(0.0, cfg.act_noise_gib)
        if cfg.act_spike_prob and rng.random() < cfg.act_spike_prob:
            activations += cfg.act_spike_gib
        activations = max(0.0, activations)

        allocated_peak = footprint(resident, activations)
        residual = activations if cfg.manual_residual_gib is None else cfg.manual_residual_gib
        allocated_residual = footprint(resident, max(0.0, residual))
        other = max(0.0, cfg.context_gib + float(external_gib(step)))
        proactive_move = None
        if guard:
            known_peak = shape_peaks.get(bucket)
            if known_peak is not None:
                target_peak_free = max(cfg.wddm_hard_gib, cfg.wddm_hard_gib + 0.5)
                predicted_peak_free = cfg.total_gib - (known_peak + other)
                while predicted_peak_free < target_peak_free and resident > 0:
                    before_resident = resident
                    resident = max(0, resident - cfg.retreat_layers_manual)
                    resident_drop = max(0, before_resident - resident) * cfg.layer_gib
                    known_peak = max(0.0, known_peak - resident_drop)
                    shape_peaks[bucket] = known_peak
                    allocated_peak = footprint(resident, activations)
                    allocated_residual = footprint(resident, max(0.0, residual))
                    predicted_peak_free = cfg.total_gib - (known_peak + other)
                    proactive_move = "pre_down"
        frag = cfg.frag_gib
        if cfg.frag_noise_gib:
            frag += rng.gauss(0.0, cfg.frag_noise_gib)
        # The hoard: reserved holds its high-water mark and creeps as new shapes
        # fragment the pool. It does NOT reset each step (that is the whole bug).
        reserved_held = max(reserved_held, allocated_peak + max(0.0, frag))
        reserved_held += cfg.frag_ratchet_gib

        # End-of-step reading, BEFORE the guard acts.
        device_used = reserved_held + other
        spilled = device_used > cfg.total_gib
        device_free = max(0.0, cfg.total_gib - device_used)

        move = proactive_move or "hold"
        if guard:
            action = MemoryManager._training_cliff_guard_action(
                device_free, wddm_hard_gib=cfg.wddm_hard_gib, did_oom=spilled,
            )
            if action == "reclaim":
                # 1. empty_cache: hand the idle hoard back, leaving the step-end trough.
                reserved_held = allocated_residual
                device_used = reserved_held + other
                device_free = max(0.0, cfg.total_gib - device_used)
                predicted_peak_free = cfg.total_gib - (allocated_peak + other)
                target_peak_free = max(cfg.wddm_hard_gib, cfg.wddm_hard_gib + 0.5)
                if move != "pre_down":
                    move = "empty_cache"
                # 2. If the next step peak would still breach the stop target,
                #    demote now rather than trusting the post-trim trough.
                while predicted_peak_free < target_peak_free and resident > 0:
                    resident = max(0, resident - cfg.retreat_layers_manual)
                    allocated_peak = footprint(resident, activations)
                    allocated_residual = footprint(resident, max(0.0, residual))
                    reserved_held = allocated_residual
                    device_used = reserved_held + other
                    device_free = max(0.0, cfg.total_gib - device_used)
                    predicted_peak_free = cfg.total_gib - (allocated_peak + other)
                    move = "down"

        available = MemoryManager._available_vram_gib(
            cfg.total_gib, device_used, reserved_held, reserved_held,
            safety_gib=cfg.safety_gib,
        )
        # NOTE: `governing` carries driver `device_free` here (the signal the guard
        # actually reads), not the cross-bucket min used in the auto sim.
        shape_peaks[bucket] = max(shape_peaks.get(bucket, 0.0), allocated_peak)
        history.append(StepRecord(
            step, bucket, resident, reserved_held, other,
            available, device_free, move, spilled,
        ))

    return history


def run_reserve_sim(
    cfg: SimConfig,
    sequence,
    steps: int,
    *,
    use_peak: bool = True,
    cross_bucket: bool = True,
    external_gib: Optional[Callable[[int], float]] = None,
):
    """Drive the REAL working_reserve sizing controller over modelled physics.

    Exercises the second auto controller — ``_training_working_reserve_signal``
    and ``_training_working_reserve_decision`` — which measures the activation demand
    (working_reserve), as opposed to :func:`run_sim` which models the layer-layout
    deadband. Because the planner packs resident weights to fill whatever the
    reserve leaves, shrinking the reserve eats the spill margin one-for-one:

        device_free(bucket) = working_reserve + wddm_margin - activation_peak(bucket)

    ``use_peak=False`` feeds the controller the step-end RESIDUAL (the measurement
    bug — it thinks the working set is ~1 GiB); ``use_peak=True`` feeds the true
    within-step PEAK (the fix). ``cross_bucket=True`` governs the reserve by the
    worst bucket's peak so a quiet low-res step cannot starve high-res.
    """
    import random

    external_gib = external_gib or (lambda _step: 0.0)
    rng = random.Random(cfg.seed)
    reserve = float(cfg.start_reserve_gib)
    danger = None
    ema: dict = {}
    latest_signal: dict = {}
    last_free: dict = {}
    bsteps: dict = {}
    history = []

    for step in range(steps):
        bucket = sequence[step % len(sequence)]
        act_peak = cfg.act_peak_gib[bucket]
        if cfg.act_noise_gib:
            act_peak += rng.gauss(0.0, cfg.act_noise_gib)
        act_peak = max(0.0, act_peak)

        # What the controller *measures* this step: the true peak (fix) or the
        # step-end residual trough (bug).
        meas = act_peak if use_peak else cfg.act_residual_gib
        bsteps[bucket] = bsteps.get(bucket, 0) + 1
        ema[bucket] = meas if bucket not in ema else 0.8 * ema[bucket] + 0.2 * meas
        signal_b = MemoryManager._training_working_reserve_signal(
            meas, ema[bucket],
            steps=bsteps[bucket],
            stable_windows=cfg.stable_windows,
            min_working_reserve_gib=cfg.min_working_reserve_gib,
            pad_gib=cfg.pad_gib,
        )
        latest_signal[bucket] = signal_b
        signal = max(latest_signal.values()) if cross_bucket else signal_b

        # Physics: free margin at THIS step's activation peak under the current
        # reserve (an external app, if any, eats straight into the margin).
        device_free = (
            reserve + cfg.wddm_margin_gib - act_peak
            + cfg.free_offset_gib - float(external_gib(step))
        )
        spilled = device_free < 0.0
        last_free[bucket] = device_free
        min_free = min(last_free.values())

        new_reserve, danger, action = MemoryManager._training_working_reserve_decision(
            reserve, signal, min_free, danger,
            wddm_hard_gib=cfg.wddm_hard_gib,
            wddm_stop_gib=cfg.wddm_stop_gib,
            pad_gib=cfg.pad_gib,
            step_gib=cfg.step_gib,
            retreat_gib=cfg.retreat_gib,
        )
        reserve = max(new_reserve, cfg.min_working_reserve_gib)

        # StepRecord reuse: peak_reserved<-reserve, available<-device_free,
        # governing<-signal, move<-decision action.
        history.append(StepRecord(
            step, bucket, 0, reserve, 0.0, max(0.0, device_free), signal, action, spilled,
        ))

    return history


def summarize_reserve(history, cfg: SimConfig):
    """Properties of a working_reserve sizing run."""
    grow_actions = {"shrink", "grow", "retreat", "oom_retreat"}
    spills = [r for r in history if r.spilled]
    warmup_end = len(history) // 2
    n_buckets = len({r.bucket for r in history})
    tail = history[-2 * n_buckets:] if len(history) >= 2 * n_buckets else history
    return {
        "steps": len(history),
        "spills": len(spills),
        "steady_spills": sum(1 for r in spills if r.step >= warmup_end),
        "retreats": sum(1 for r in history if r.move in ("retreat", "oom_retreat")),
        "final_reserve": round(history[-1].peak_reserved, 3),
        "min_free": round(min(r.available for r in history), 3),
        "final_free": round(history[-1].available, 3),
        "tail_moves": sum(1 for r in tail if r.move in grow_actions),
    }


def summarize_manual(history, cfg: SimConfig):
    """Properties of a manual-mode (cliff-guard) run."""
    spills = [r for r in history if r.spilled]
    warmup_end = len(history) // 2
    return {
        "steps": len(history),
        "spills": len(spills),
        "steady_spills": sum(1 for r in spills if r.step >= warmup_end),
        "empty_caches": sum(1 for r in history if r.move == "empty_cache"),
        "demotes": sum(1 for r in history if r.move in ("down", "pre_down")),
        "pre_demotes": sum(1 for r in history if r.move == "pre_down"),
        "max_reserved": round(max(r.peak_reserved for r in history), 3),
        "min_free": round(min(r.governing for r in history), 3),
        "final_resident": history[-1].resident,
    }


def summarize(history, cfg: SimConfig):
    """Derive the properties we actually care about from a run."""
    def is_move(r):
        return r.move.startswith("up x") or r.move in ("up", "down")

    moves = [r for r in history if is_move(r)]
    spills = [r for r in history if r.spilled]
    n_buckets = len({r.bucket for r in history})
    # Steady-state churn: layout moves in the back half (post-warmup). Under noise
    # this is the metric that matters — every such move is a real trace reset.
    warmup_end = len(history) // 2
    steady_moves = sum(1 for r in moves if r.step >= warmup_end)
    steady_spills = sum(1 for r in spills if r.step >= warmup_end)
    # Convergence step: first step after which no up/down move ever happens again.
    converged_at = None
    last_move_step = max((r.step for r in moves), default=-1)
    if last_move_step < history[-1].step:
        converged_at = last_move_step + 1
    held_tail = [r for r in history if converged_at is not None and r.step >= converged_at]
    # Judge "settled in band" over the final full bucket-cycle, AFTER the
    # one-step stale-reading transient that follows a promote has cleared.
    final_cycle = history[-2 * n_buckets:] if len(history) >= 2 * n_buckets else history
    settled_in_band = all(
        cfg.wddm_hard_gib <= r.governing <= cfg.wddm_hold_high_gib for r in final_cycle
    )
    return {
        "steps": len(history),
        "moves": len(moves),
        "promotes": sum(1 for r in moves if r.move.startswith("up")),
        "demotes": sum(1 for r in moves if r.move == "down"),
        "spills": len(spills),
        "converged_at": converged_at,
        "final_resident": history[-1].resident,
        "final_governing": round(history[-1].governing, 3),
        "held_steps": len(held_tail),
        "tail_moves": sum(1 for r in held_tail if is_move(r)),
        "steady_moves": steady_moves,
        "steady_spills": steady_spills,
        "settled_in_band": settled_in_band,
    }


def summarize_timing_spill(history, cfg: SimConfig):
    """Properties of a timing-cliff learning run."""
    learned_floor = max(r.governing for r in history)
    first_event = next((r.step for r in history if r.move == "timing_spill"), None)
    final_cycle = history[-4:] if len(history) >= 4 else history
    return {
        "steps": len(history),
        "timing_spills": sum(1 for r in history if r.move == "timing_spill"),
        "first_timing_spill": first_event,
        "learned_wddm_hard": round(learned_floor, 3),
        "configured_wddm_hard": cfg.wddm_hard_gib,
        "demotes_after_learning": sum(
            1 for r in history if first_event is not None and r.step > first_event and r.move == "down"
        ),
        "final_min_available": round(min(r.available for r in final_cycle), 3),
        "final_above_learned_floor": all(r.available >= learned_floor - 1e-6 for r in final_cycle),
    }

def print_trace(history, cfg: SimConfig, every: int = 1):
    print(
        f"{'step':>4} {'bucket':>7} {'res':>4} {'peak_rsv':>9} "
        f"{'other':>6} {'avail':>6} {'govern':>6}  move"
    )
    real_move = lambda r: r.move.startswith("up x") or r.move in ("up", "down")
    for r in history:
        quiet = r.move in ("hold", "wait_cadence", "maxed")
        if r.step % every and quiet and not r.spilled:
            continue
        flag = " SPILL" if r.spilled else ""
        print(
            f"{r.step:>4} {r.bucket:>7} {r.resident:>4} {r.peak_reserved:>9.2f} "
            f"{r.other:>6.2f} {r.available:>6.2f} {r.governing:>6.2f}  {r.move}{flag}"
        )


# --- scenarios ------------------------------------------------------------

def scenario_default(cfg=None):
    """Single-layer cadence from cold start (current live behavior — slow warmup)."""
    cfg = cfg or SimConfig(promote_strategy="single")
    return cfg, run_sim(cfg, ["res512", "res256"], steps=320)


def scenario_batch(cfg=None):
    """Batch promote toward mid-band (proposed) — converges in a couple moves."""
    cfg = cfg or SimConfig(promote_strategy="batch")
    return cfg, run_sim(cfg, ["res512", "res256"], steps=120)


def scenario_pressure(cfg=None):
    """Converge (batch), then a 3 GiB external app appears at step 50 and leaves at 80."""
    cfg = cfg or SimConfig(promote_strategy="batch")

    def external(step):
        return 3.0 if 50 <= step < 80 else 0.0

    return cfg, run_sim(cfg, ["res512", "res256"], steps=140, external_gib=external)


def scenario_small_card(cfg=None):
    """Tight 12 GiB card — less room, should still converge and hold without spill."""
    cfg = cfg or SimConfig(total_gib=12.0, promote_strategy="batch")
    return cfg, run_sim(cfg, ["res512", "res256"], steps=120)


def scenario_timing_spill_learning(cfg=None):
    """Configured floor is too low; timing cliff teaches the controller the real one."""
    cfg = cfg or SimConfig(
        start_resident=50,
        wddm_hard_gib=0.5,
        wddm_hold_high_gib=2.4,
        retreat_layers=3,
    )
    return cfg, run_timing_spill_sim(
        cfg, ["res512", "res256"], steps=80, hidden_cliff_gib=2.4
    )


# Noise calibrated loosely to the Krea logs: ~±0.12 GiB activation jitter, an
# occasional big-aspect-ratio batch, allocator fragmentation slack, and OS noise.
def _noisy_cfg(strategy, **kw):
    return SimConfig(
        promote_strategy=strategy,
        act_noise_gib=0.12,
        act_spike_prob=0.03,
        act_spike_gib=0.5,
        frag_gib=0.2,
        frag_noise_gib=0.1,
        other_noise_gib=0.1,
        **kw,
    )


def scenario_noisy_batch(cfg=None):
    """Realistic per-step noise, batch promote to mid-band — should hold quietly."""
    cfg = cfg or _noisy_cfg("batch")
    return cfg, run_sim(cfg, ["res512", "res256"], steps=200)


def scenario_noisy_single(cfg=None):
    """Realistic noise, single-layer cadence — exposes near-edge churn risk."""
    cfg = cfg or _noisy_cfg("single")
    return cfg, run_sim(cfg, ["res512", "res256"], steps=400)


def scenario_noisy_smoothed(cfg=None):
    """Realistic noise, batch + EMA smoothing — should hold with ~no steady churn."""
    cfg = cfg or _noisy_cfg("batch", smooth_alpha=0.3)
    return cfg, run_sim(cfg, ["res512", "res256"], steps=200)


# --- manual working_reserve (cliff guard) ----------------------------------
# Calibrated to the observed 12 GiB Krea run: other≈0.5 GiB, per-bucket live
# peaks ~7.9/8.4/9.0 GiB (always_resident 1.8 + ~20 demotable layers @0.08 +
# activations), allocator fragmentation that ratchets `reserved` toward 12.
def _manual_cfg(**kw):
    base = dict(
        total_gib=12.0,
        context_gib=0.5,
        always_resident_gib=1.8,
        layer_gib=0.08,
        start_resident=20,
        working_floor_gib={"res768": 5.6, "res512": 5.0, "res256": 4.5},
        frag_gib=0.5,
        frag_ratchet_gib=0.12,
        retreat_layers_manual=3,
    )
    base.update(kw)
    return SimConfig(**base)


def scenario_manual_cliff_guard(cfg=None, *, guard=True):
    """Manual budget + fragmentation hoard: guard reclaims idle cache, never spills."""
    cfg = cfg or _manual_cfg()
    return cfg, run_manual_sim(
        cfg, ["res512", "res256", "res768", "res512"], steps=160, guard=guard
    )


def scenario_manual_no_guard(cfg=None):
    """Same as manual_cliff_guard but guard OFF — reproduces the observed blow-up."""
    return scenario_manual_cliff_guard(cfg, guard=False)


def scenario_manual_overcommit(cfg=None, *, guard=True):
    """Live footprint near the ceiling: empty_cache alone is not enough, must demote."""
    cfg = cfg or _manual_cfg(
        always_resident_gib=2.6,
        start_resident=40,                # 2.6 + 3.2 resident + activations -> tight
        working_floor_gib={"res768": 5.6, "res512": 5.0, "res256": 4.6},
    )
    return cfg, run_manual_sim(
        cfg, ["res512", "res256", "res768", "res512"], steps=160, guard=guard
    )



def scenario_manual_trough_peak(cfg=None, *, guard=True):
    """Post-empty_cache trough looks healthy, but next-step peak requires demotion."""
    cfg = cfg or _manual_cfg(
        always_resident_gib=2.6,
        start_resident=40,
        working_floor_gib={"res768": 5.6, "res512": 5.0, "res256": 4.6},
        manual_residual_gib=0.2,
    )
    return cfg, run_manual_sim(
        cfg, ["res512", "res256", "res768", "res512"], steps=40, guard=guard
    )

# --- working_reserve sizing controller (auto) ------------------------------
def scenario_reserve_peak_fix(cfg=None, *, use_peak=True, cross_bucket=True):
    """Auto reserve sizing on the TRUE peak: climbs down to ~peak+pad and holds."""
    cfg = cfg or SimConfig()
    return cfg, run_reserve_sim(
        cfg, ["res512", "res256", "res768", "res512"], steps=120,
        use_peak=use_peak, cross_bucket=cross_bucket,
    )


def scenario_reserve_trough_bug(cfg=None):
    """Auto reserve sizing on the step-end RESIDUAL (the bug): starves and spills."""
    return scenario_reserve_peak_fix(cfg, use_peak=False)


def scenario_reserve_per_bucket(cfg=None):
    """Peak signal but per-bucket (no cross-bucket governance): low-res starves high-res."""
    return scenario_reserve_peak_fix(cfg, use_peak=True, cross_bucket=False)


def scenario_reserve_low_seed(cfg=None):
    """Auto SEEDED BELOW the real peak (the live bug: seed ~2.5, peak ~5.2).

    The old shrink-only decision could only walk DOWN toward the target, so a
    low seed never reached the activation peak: it under-reserved every step,
    the activations overflowed the margin, and it relied on hard-floor retreats
    (+1.0) to crawl up — spilling along the way. The grow branch makes it climb
    straight to peak+pad and hold, no spills.
    """
    cfg = cfg or SimConfig(start_reserve_gib=2.5, free_offset_gib=4.0)
    return cfg, run_reserve_sim(
        cfg, ["res512", "res256", "res768", "res512"], steps=120,
        use_peak=True, cross_bucket=True,
    )


def scenario_reserve_pressure(cfg=None):
    """Fixed reserve controller, then a 2 GiB external app appears at step 40-70."""
    cfg = cfg or SimConfig()

    def external(step):
        return 2.0 if 40 <= step < 70 else 0.0

    return cfg, run_reserve_sim(
        cfg, ["res512", "res256", "res768", "res512"], steps=120,
        use_peak=True, cross_bucket=True, external_gib=external,
    )


SCENARIOS = {
    "default": scenario_default,
    "batch": scenario_batch,
    "pressure": scenario_pressure,
    "small_card": scenario_small_card,
    "timing_spill_learning": scenario_timing_spill_learning,
    "noisy_batch": scenario_noisy_batch,
    "noisy_single": scenario_noisy_single,
    "noisy_smoothed": scenario_noisy_smoothed,
    "manual_cliff_guard": scenario_manual_cliff_guard,
    "manual_no_guard": scenario_manual_no_guard,
    "manual_overcommit": scenario_manual_overcommit,
    "manual_trough_peak": scenario_manual_trough_peak,
    "reserve_peak_fix": scenario_reserve_peak_fix,
    "reserve_trough_bug": scenario_reserve_trough_bug,
    "reserve_per_bucket": scenario_reserve_per_bucket,
    "reserve_low_seed": scenario_reserve_low_seed,
    "reserve_pressure": scenario_reserve_pressure,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="default", choices=sorted(SCENARIOS))
    ap.add_argument("--every", type=int, default=4,
                    help="print every Nth hold step (moves/spills always shown)")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args()
    if args.list:
        for name, fn in sorted(SCENARIOS.items()):
            print(f"  {name:12} {fn.__doc__.splitlines()[0]}")
        return
    cfg, history = SCENARIOS[args.scenario]()
    print_trace(history, cfg, every=args.every)
    print()
    if args.scenario.startswith("manual"):
        summary = summarize_manual(history, cfg)
    elif args.scenario.startswith("timing"):
        summary = summarize_timing_spill(history, cfg)
    elif args.scenario.startswith("reserve"):
        summary = summarize_reserve(history, cfg)
    else:
        summary = summarize(history, cfg)
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
