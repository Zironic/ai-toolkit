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


SCENARIOS = {
    "default": scenario_default,
    "batch": scenario_batch,
    "pressure": scenario_pressure,
    "small_card": scenario_small_card,
    "noisy_batch": scenario_noisy_batch,
    "noisy_single": scenario_noisy_single,
    "noisy_smoothed": scenario_noisy_smoothed,
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
    for k, v in summarize(history, cfg).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
