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
