"""Dedicated-VRAM (WDDM) budget arithmetic: one source of truth, one name per meaning.

This module owns the *dedicated-cliff* quantities. Windows/WDDM has two distinct
memory cliffs with different failure modes:

* **Dedicated ceiling** (this module): crossing the card's physical VRAM makes
  WDDM silently page GPU memory to system RAM -- catastrophic slowdown, no
  error. Governed by ``torch.cuda.mem_get_info`` (driver-level, sees every
  process). Quantities: ``hard_gib`` (the never-cross device-free floor,
  default 1.0), ``margin_gib`` (the planning headroom, >= hard).
* **Shared / DXGI NON_LOCAL budget** (NOT this module): pinned host memory
  commits against it and exhausting it is a hard cudaErrorMemoryAllocation.
  That reserve lives in ``pin_manager`` / ``bounce_pool``
  (``dxgi_spill_reserve_bytes``); do not conflate the two.

Vocabulary (each quantity has exactly one name):

* ``total`` / ``free`` -- physical card bytes, from ``mem_get_info``.
* ``torch_reserved`` / ``torch_allocated`` -- torch's caching allocator.
* ``non_torch`` -- ``(total - free) - torch_reserved``: CUDA context, VAE/TE,
  the Windows desktop, other processes. Torch is never the card's only tenant.
* ``hard`` -- device-free floor the card must keep (WDDM spill guard).
* ``margin`` -- planning headroom subtracted from budgets; ``>= hard``.

Everything here is pure (CPU-testable) except ``DeviceSnapshot.capture``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch

GIB = 1024 ** 3


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


@dataclass(frozen=True)
class DeviceSnapshot:
    """Point-in-time physical view of a CUDA device (bytes)."""

    total: int
    free: int
    torch_reserved: int
    torch_allocated: int

    @property
    def used(self) -> int:
        return max(0, self.total - self.free)

    @property
    def non_torch(self) -> int:
        """Device bytes held by anyone but torch's caching allocator."""
        return max(0, self.used - self.torch_reserved)

    @staticmethod
    def capture(device) -> Optional["DeviceSnapshot"]:
        if device is None or not torch.cuda.is_available():
            return None
        dev = torch.device(device)
        if dev.type != "cuda":
            return None
        free, total = torch.cuda.mem_get_info(dev)
        return DeviceSnapshot(
            total=int(total),
            free=int(free),
            torch_reserved=int(torch.cuda.memory_reserved(dev)),
            torch_allocated=int(torch.cuda.memory_allocated(dev)),
        )

    def format(self) -> str:
        return (
            f"torch_allocated={self.torch_allocated / GIB:.2f} GiB "
            f"torch_reserved={self.torch_reserved / GIB:.2f} GiB "
            f"device_used={self.used / GIB:.2f}/{self.total / GIB:.2f} GiB "
            f"device_free={self.free / GIB:.2f} GiB "
            f"non_torch={self.non_torch / GIB:.2f} GiB"
        )


@dataclass(frozen=True)
class WddmMargins:
    """Dedicated-cliff margins for one phase (training attach / sampling start).

    Resolve ONCE at the phase boundary and pass by value; do not re-read env
    vars mid-phase (they cannot change mid-run, and re-reads hide which value
    actually governed a decision).
    """

    hard_gib: float
    margin_gib: float
    source: str  # "config" | "env" | "auto"

    @property
    def hard_bytes(self) -> int:
        return int(self.hard_gib * GIB)

    @property
    def margin_bytes(self) -> int:
        return int(self.margin_gib * GIB)

    def format(self) -> str:
        return (
            f"wddm_hard={self.hard_gib:.2f} GiB "
            f"wddm_margin={self.margin_gib:.2f} GiB ({self.source})"
        )


def auto_margin_gib(device, pct: float = 0.10, floor_gib: float = 1.0) -> float:
    """Auto planning margin: max(floor, pct * card size)."""
    try:
        total_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    except Exception:
        try:
            _free, total_bytes = torch.cuda.mem_get_info(device)
        except Exception:
            total_bytes = 0
    total_gib = max(0.0, float(total_bytes) / GIB)
    return max(float(floor_gib), float(pct) * total_gib)


def resolve_margins(
    device,
    margin_value,
    hard_value,
    *,
    margin_env: str,
    hard_env: str,
) -> WddmMargins:
    """Resolve the phase's margins from config value > env > auto.

    ``margin_value`` / ``hard_value`` are the config-supplied values (``None``
    means "consult the env var"; a negative margin or "auto" means auto).
    ``margin`` is clamped to at least ``hard``.
    """
    hard_gib = float(_env(hard_env, "1.0")) if hard_value is None else float(hard_value)
    raw = _env(margin_env, "-1.0") if margin_value is None else margin_value
    source = "env" if margin_value is None else "config"
    try:
        margin_gib = float(raw)
        auto = margin_gib < 0
    except (TypeError, ValueError):
        auto = str(raw).strip().lower() == "auto"
        margin_gib = -1.0
    if auto:
        margin_gib = auto_margin_gib(device)
        source = "auto"
    return WddmMargins(
        hard_gib=hard_gib,
        margin_gib=max(margin_gib, hard_gib or 0.0),
        source=source,
    )


def cap_fraction(total_bytes, free_bytes, reserved_bytes, hard_gib) -> float:
    """Allocator-cap fraction so device_used stays <= total - hard (pure).

    The cap governs torch's own reserved pool, but torch is not the card's
    only tenant (``non_torch``). Capping torch at ``total - hard`` alone lets
    device_used reach ``total - hard + non_torch`` (observed: reserved 10.97 +
    non_torch 1.02 = 11.99/11.99 GiB, device_free 0, silent WDDM paging).
    Subtract the measured non-torch share so the whole device, not just torch,
    keeps the hard margin free.
    """
    total = float(max(1, int(total_bytes)))
    non_torch = max(0.0, (total - float(free_bytes)) - float(reserved_bytes))
    cap_bytes = total - float(hard_gib) * GIB - non_torch
    return max(0.1, min(1.0, cap_bytes / total))


def sampling_allocator_budget_free_bytes(
    total_bytes,
    allocated_bytes,
    cap_fraction,
    hard_bytes,
    *,
    gc_threshold=0.95,
):
    """Allocated-side equivalent of driver-free for the sampling planner (pure).

    Driver-free counts torch's idle cached segments as *used*, so a plan built
    from it refuses residency that the allocator cap's GC would reclaim on
    demand. The allocator-side capacity is governed by the gc target
    (``gc_threshold * cap``): live allocations may safely grow to it, and the
    planner's margin beyond the hard floor (which is already inside the cap)
    stays free below the target as the fragmentation/allowance pad.

    Returned in the same units/meaning as ``mem_get_info`` free so the
    downstream ``usable = free - working_reserve - margin`` keeps its shape:

        usable = threshold*cap - allocated - working_reserve - (margin - hard)

    Returns ``None`` when no cap fraction is known (non-Windows / cap not
    applied); callers should then stay on the driver-free number.
    """
    if cap_fraction is None:
        return None
    cap_bytes = float(cap_fraction) * float(max(1, int(total_bytes)))
    return int(
        float(gc_threshold) * cap_bytes
        - float(max(0, int(allocated_bytes)))
        + float(max(0, int(hard_bytes)))
    )


def sampling_guard_predicted_peak_free(total_b, free_b, reserved_b, peak_reserved_b):
    """Predicted free VRAM at the next forward's peak (pure).

    ``non_torch = (total - free) - reserved`` plus the worst forward's reserved
    high-water is what the next peak will occupy; the prediction is ``total``
    minus that. It shrinks one-for-one as external use grows -- which is the
    cohabitation guard's trigger. Forward-only sampling never OOMs at the cliff
    (it pages silently), so the guard watches this instead of an exception.
    """
    other_b = max(0, (total_b - free_b) - reserved_b)
    return total_b - (peak_reserved_b + other_b)


def training_cliff_predicted_peak_free_gib(
    total_gib, device_free_gib, torch_reserved_gib, peak_allocated_gib
):
    """Driver free expected when the next step rebuilds its live peak (pure).

    ``empty_cache`` can make step-end free look healthy by dropping idle cached
    blocks, but the next forward/backward will recreate the live peak. Keep
    non-allocator residents (``non_torch``) from the current snapshot and ask
    whether peak allocated memory itself clears the WDDM hard floor.
    """
    device_used_gib = max(0.0, total_gib - device_free_gib)
    non_torch_gib = max(0.0, device_used_gib - torch_reserved_gib)
    return total_gib - (max(0.0, peak_allocated_gib) + non_torch_gib)


def sampling_step_should_trim(free_before_b, trim_margin_b) -> bool:
    """Whether realized device-free warrants a per-step cache trim (pure).

    WDDM pages on the committed footprint silently, so the trigger is realized
    free, not an allocated-side or peak signal. Trim (empty_cache) is cheap and
    non-destructive, so the bar is just "free has dropped into the margin."
    """
    return free_before_b < trim_margin_b


def sampling_step_should_demote(free_after_b, hard_floor_b) -> bool:
    """Whether to escalate to a block demote after a trim (pure).

    Only when trimming left free still under the hard floor -- i.e. there was
    no idle cache to reclaim, so the pressure is real (external) and the only
    relief is giving back resident weights.
    """
    return free_after_b < hard_floor_b


def estimate_sampling_working_reserve_bytes(
    image_tokens: int,
    text_tokens: int = 512,
    *,
    batch_cfg: bool = False,
    fp8_native: bool = True,
    base_bytes: int = int(2.2 * GIB),
    per_token_bytes: int = 40 * 1024,
    dequant_pad_bytes: int = int(1.4 * GIB),
    safety: float = 1.15,
    headroom_bytes: int = GIB,
) -> int:
    """Cold-start estimate of the sampling working set (pure, CPU-testable).

    Used before any measured peak exists, so a high-resolution first sample
    plans enough streaming up front instead of discovering the working set
    via mid-denoise OOM demotions (every demote invalidates compiled state
    and, under strict ingraph, changes the pack set).

    Linear-in-tokens model calibrated on Krea2 RTX 4070 smoke runs
    (2026-07-07/08, fp8 + cutlass attention, sequential CFG, partial
    residency; ``sampling_extra`` = peak allocated minus resident weights):

        512x512  -> L = 1024 + 512 = 1536 tokens,  extra ~= 2.2 GiB
        2000x2000-> L = 15625 + 512 = 16137 tokens, extra ~= 2.8 GiB
        => per_token ~= 40 KiB, base ~= 2.2 GiB (streaming/dequant churn +
           fixed workspaces dominate; per-token activations are small)

        512x512 dequant fallback (fp8 sampling off) -> extra ~= 3.6 GiB
        => dequant_pad ~= 1.4 GiB (torchao fp32 dequant transients)

    Batched CFG scales the token-dependent share by 2.5, not 2.0: besides the
    batch-2 doubling, the fp32 intermediates (observed as (2, L, features)
    fp32 allocations, 740 MiB each at 2000px) are underweighted in the
    batch-1-calibrated per-token constant -- the x2.0 estimate ran ~1 GiB
    short at 2000px (two demote rounds). ``safety`` covers model-to-model
    variation, and
    ``headroom_bytes`` (flat +1 GiB) deliberately overestimates: streaming
    one extra block costs a little bandwidth, while underestimating costs a
    mid-denoise demote -- which invalidates compiled state, mutates the
    strict-ingraph pack set, and (observed at 2000px) can cascade into a
    full streamed transition. The learned per-run reserve replaces this
    estimate after the first measured sample.
    """
    tokens = max(0, int(image_tokens)) + max(0, int(text_tokens))
    token_bytes = int(tokens * per_token_bytes * (2.5 if batch_cfg else 1.0))
    estimate = int(base_bytes) + token_bytes
    if not fp8_native:
        estimate += int(dequant_pad_bytes)
    return int(estimate * float(safety)) + int(headroom_bytes)


def training_guard_pressure(dxgi: dict, physical: dict) -> dict:
    """Combine the DXGI LOCAL and physical cliff signals (pure).

    Pressure if EITHER signal predicts the next step's peak crosses its floor.
    The DXGI LOCAL budget is a per-process OS grant and its usage counter
    excludes other processes, so it can bless a layout the physical
    (mem_get_info) view already knows will overfill the card -- and vice versa
    when the OS shrinks the budget early. The merged dict keeps the DXGI
    fields at the top level (``source`` compatibility) and carries the
    physical signal under ``physical_*``.
    """
    merged = dict(dxgi)
    merged["pressure"] = bool(dxgi.get("pressure")) or bool(physical.get("pressure"))
    merged["physical_predicted_peak_free_gib"] = physical.get("predicted_peak_free_gib")
    merged["physical_target_free_gib"] = physical.get("target_free_gib")
    merged["pressure_sources"] = [
        src["source"] for src in (dxgi, physical) if src.get("pressure")
    ]
    return merged


# ---------------------------------------------------------------------------
# Two-timescale residency control (see tasks/open/RESIDENCY_TWO_TIMESCALE_PLAN.md)
#
# Allowance lives in *target-space* (0.95*cap - live); the allocator cap is set
# in *cap-space*. The two differ by the gc_threshold factor: a cap raise of ``d``
# only adds ``gc_threshold * d`` of GC target / allowance. Every conversion below
# carries the ``/ gc_threshold`` so no call site open-codes it (that missing
# divisor silently under-reserves and licenses a promotion that immediately binds).
#
# All functions here are pure/CPU-testable. The cap and residency levers both act
# only at phase boundaries, and a cap change is realized lazily -- on the next
# fresh cudaMalloc, i.e. the next forward()/step -- so the controller reads
# counters that lag its move by one window (the FSM's verify phases absorb this).
# ---------------------------------------------------------------------------

GC_THRESHOLD = 0.95


def allocator_allowance_bytes(cap_bytes, live_bytes, *, gc_threshold=GC_THRESHOLD) -> int:
    """Idle-cache allowance under the cap: ``gc_threshold*cap - live`` (pure).

    The caching allocator sweeps idle segments when reserved would cross the GC
    target ``gc_threshold * cap``; live bytes (residents + ring + activations)
    count against that target but cannot be freed. So the room left for reusable
    idle cache -- the buffer between smooth reuse and a fresh-cudaMalloc sweep --
    is ``gc_threshold*cap - live``. Negative means live alone exceeds the target:
    every sweep dumps all cache and every reuse re-mallocs (self-sustaining
    thrash), so callers must keep this positive at the live peak.
    """
    return int(float(gc_threshold) * float(max(0, int(cap_bytes))) - float(max(0, int(live_bytes))))


def cap_bytes_for_live(
    planned_live_bytes,
    cache_budget_bytes,
    cliff_cap_bytes,
    *,
    floor_cap_bytes=0,
    gc_threshold=GC_THRESHOLD,
) -> int:
    """Cap that hosts ``planned_live`` plus an idle-cache budget (pure).

    Inverse of :func:`allocator_allowance_bytes`: to let live grow to
    ``planned_live`` while keeping ``cache_budget`` of reusable idle cache under
    the GC target, the cap must be ``(planned_live + cache_budget) / gc_threshold``.
    Clamped to the WDDM cliff bound above (never license silent paging; see
    :func:`cap_fraction`) and an optional floor below.
    """
    want = (float(max(0, int(planned_live_bytes))) + float(max(0, int(cache_budget_bytes)))) / float(gc_threshold)
    want = min(want, float(int(cliff_cap_bytes)))
    want = max(want, float(max(0, int(floor_cap_bytes))))
    return int(want)


def cap_can_host_promotion(
    live_bytes,
    block_bytes,
    slack_pad_bytes,
    cliff_cap_bytes,
    *,
    gc_threshold=GC_THRESHOLD,
) -> bool:
    """Can the cheap cap lever (tier 1) absorb one more resident block? (pure).

    Promoting a streamed block to resident raises live by ``block_bytes``. To
    keep ``slack_pad_bytes`` of allowance afterward, the GC target must reach
    ``live + block + slack``, i.e. the cap must reach
    ``(live + block + slack) / gc_threshold``. The cap lever can do this only if
    that target cap is still under the WDDM cliff bound; otherwise the cap is
    pinned at the cliff and the allowance must come from lowering live -- an
    expensive resident demote (tier 2).

    The ``/ gc_threshold`` is load-bearing: a naive ``cliff - cap >= block`` test
    under-reserves by the 0.95 factor.
    """
    need_cap = (
        float(max(0, int(live_bytes)))
        + float(max(0, int(block_bytes)))
        + float(max(0, int(slack_pad_bytes)))
    ) / float(gc_threshold)
    return need_cap <= float(int(cliff_cap_bytes))


def residency_promote_ok(
    num_alloc_retries,
    reclaimable_at_peak_bytes,
    block_bytes,
    slack_pad_bytes,
) -> bool:
    """Sampling climb gate: convert one streamed block to resident? (pure).

    Approach residency from below (undershoot-and-climb): only add a block when
    the telemetry proves the room is really there --

      * ``num_alloc_retries == 0`` over the window (nothing cap-binding), AND
      * ``reclaimable_at_peak`` (= peak_reserved - peak_alloc, idle cache still
        held AT the allocation peak) exceeds one block plus the pad, so the
        promotion still leaves ``slack_pad`` of allowance.

    Both must hold: retries can be zero simply because residency is too low, so
    the reclaimable-at-peak test is what proves there is slack to spend.
    """
    if int(num_alloc_retries or 0) > 0:
        return False
    return float(reclaimable_at_peak_bytes or 0) > float(block_bytes) + float(slack_pad_bytes)


# --- Hysteresis FSM (one transition per phase boundary) ---------------------
#
# States mirror the plan's state machine. DEMOTE_REQUIRED is folded into the
# transition (emit "demote" and land in COLD) since the ring resize is a
# synchronous boundary transaction, not a state that waits a window.

FSM_COLD = "cold"                          # measurements invalid (post-compile/retrace/demote)
FSM_STABLE = "stable"                      # clean + eligible for the from-below climb
FSM_CAP_VERIFY = "cap_verify"              # cap raised, confirming it took
FSM_PROMOTION_VERIFY = "promotion_verify"  # one block promoted, confirming clean
FSM_COOLDOWN = "cooldown"                  # re-promotion barred N windows after a rollback

ACT_HOLD = "hold"
ACT_RAISE_CAP = "raise_cap"
ACT_PROMOTE = "promote"
ACT_DEMOTE = "demote"
ACT_ROLLBACK = "rollback"


@dataclass(frozen=True)
class ResidencyFsmState:
    name: str = FSM_COLD
    windows_in_state: int = 0


def residency_fsm_step(
    state: ResidencyFsmState,
    signals: dict,
    *,
    k_clean: int = 2,
    k_verify: int = 2,
    cooldown_n: int = 4,
) -> tuple[ResidencyFsmState, str]:
    """Advance the residency controller one phase boundary (pure, CPU-testable).

    ``signals`` (all read as bools unless noted):
      * ``measurements_invalid`` -- a recompile / retrace / layout change happened;
        every counter read across it is meaningless -> force COLD.
      * ``binding`` -- retries or external pressure this window (allowance too low).
      * ``cap_can_relieve`` -- a cap raise can restore the pad at current live
        (cliff has room); if False under pressure, only a demote can.
      * ``promote_gate`` -- :func:`residency_promote_ok` verdict (there is slack).
      * ``cap_covers_promo`` -- the cliff already hosts the promotion with no raise
        (:func:`cap_can_host_promotion` at the current cap headroom).

    Returns ``(next_state, action)`` with ``action`` in the ``ACT_*`` set. The
    verify phases each span ``k_verify`` windows because a cap/residency move only
    shows its signal on the *next* step (the one-window GC lag).
    """
    name = state.name
    w = state.windows_in_state + 1

    invalid = bool(signals.get("measurements_invalid"))
    binding = bool(signals.get("binding"))
    cap_relieve = bool(signals.get("cap_can_relieve"))
    promote_gate = bool(signals.get("promote_gate"))
    cap_covers = bool(signals.get("cap_covers_promo"))

    def stay(action=ACT_HOLD):
        return ResidencyFsmState(name, w), action

    def enter(new_name, action=ACT_HOLD):
        return ResidencyFsmState(new_name, 0), action

    # A demote is a synchronous transaction that invalidates the layout -> COLD.
    def demote():
        return ResidencyFsmState(FSM_COLD, 0), ACT_DEMOTE

    if invalid and name not in (FSM_COLD,):
        return enter(FSM_COLD)

    if name == FSM_COLD:
        if invalid or binding:
            return ResidencyFsmState(FSM_COLD, 0 if invalid else w), ACT_HOLD
        return enter(FSM_STABLE) if w >= k_clean else stay()

    if name == FSM_STABLE:
        if binding:
            return enter(FSM_CAP_VERIFY, ACT_RAISE_CAP) if cap_relieve else demote()
        # Eligible to climb only once the state has held clean for k_clean windows.
        if promote_gate and w >= k_clean:
            if cap_covers:
                return enter(FSM_PROMOTION_VERIFY, ACT_PROMOTE)
            return enter(FSM_CAP_VERIFY, ACT_RAISE_CAP)  # pre-fund, promote after verify
        return stay()

    if name == FSM_CAP_VERIFY:
        if binding:
            return demote()  # the raise didn't relieve -> shed live
        return enter(FSM_STABLE) if w >= k_verify else stay()

    if name == FSM_PROMOTION_VERIFY:
        if w == 1:
            return stay()  # ignore the first (cold) window: new layout re-primes
        if binding:
            return enter(FSM_COOLDOWN, ACT_ROLLBACK)
        return enter(FSM_STABLE) if w >= k_verify + 1 else stay()

    if name == FSM_COOLDOWN:
        if binding:
            return demote()  # demote still allowed during cooldown
        return enter(FSM_STABLE) if w >= cooldown_n else stay()

    # Unknown state: fail safe to COLD.
    return enter(FSM_COLD)
