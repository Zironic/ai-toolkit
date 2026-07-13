"""Public integration surface for arena offload.

Everything a model integration or the shared trainer is allowed to touch lives
here. The rule the rest of the codebase must follow:

    from toolkit.memory_management.arena_offload import (
        prepare_arena_offload, get_arena_runtime, ...
    )

and nothing else. In particular, no `CanonicalArena`, `ResidencyState`,
`ResidencyPlan`, or `prepare_immutable_runtime` imports outside this package.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .runtime import ArenaOffloadRuntime

RUNTIME_ATTR = "_arena_offload_runtime"

_FP8_QTYPES = ("qfloat8", "float8")


def unwrap(model):
    """Peel Accelerate / DDP / torch.compile wrappers without importing them.

    The arena package must stay importable from a bare CPU test process, so this
    does not go through `toolkit.accelerator.unwrap_model` (which constructs a
    global `Accelerator`).
    """
    seen = set()
    while model is not None and id(model) not in seen:
        seen.add(id(model))
        orig = getattr(model, "_orig_mod", None)
        if orig is not None and orig is not model:
            model = orig
            continue
        if getattr(model, RUNTIME_ATTR, None) is not None:
            return model
        if hasattr(model, "_memory_manager"):
            return model
        inner = getattr(model, "module", None)
        if inner is None or inner is model:
            return model
        model = inner
    return model


@dataclass(frozen=True)
class LegacyPlannerOptions:
    """Fork-only planner knobs the Phase-1 facade still forwards verbatim.

    These are NOT part of the upstream configuration surface (see Phase 8: no
    user-facing residency / reserve / WDDM / prefetch controls). They exist so
    Phase 1 stays behavior-preserving while planning is still delegated to the
    legacy manager. Phase 2 replaces the training half with `policy.py` and this
    shrinks to whatever the arena planner genuinely needs.
    """

    working_reserve_gib: float | None = None
    wddm_margin_gib: float | None = None
    wddm_hard_gib: float | None = None
    wddm_spill_reserve_pct: float = 0.10
    block_stream_only: bool = False
    checkpoint_keep_last: int = 0
    prefetch_depth: int = 2
    # Free-margin (GiB) the live residency climb should aim to KEEP. 0 = off (the
    # conservative one-block-per-cadence climb).
    eager_promote_free_gib: float = 0.0
    eager_promote_max_blocks: int = 4

    sampling_working_reserve_gib: float | None = None
    sampling_wddm_margin_gib: float | None = None
    sampling_wddm_hard_gib: float | None = 1.0


@dataclass(frozen=True)
class ArenaOffloadConfig:
    """The whole public configuration surface of arena offload."""

    enabled: bool = False
    fp8_forward: bool = False
    fp8_backward: bool = False
    fp8_sampling: bool = False
    compile_blocks: bool = False
    compile_dynamic: bool | None = True
    compile_dynamic_hints: tuple[tuple[int, int | None, int | None], ...] = ()
    # Validation knob: pretend the card is this many GiB, so small-card
    # behaviour (deeper streaming, tighter caps, a residency plan that cannot
    # fit) is exercisable on a bigger one. 0/None = use the real card.
    simulated_vram_gib: float | None = None
    # Violating the allocator cap raises instead of widening the cap. Off in
    # production: the cap is a lever, not a kill switch.
    wddm_cap_strict: bool = False

    legacy: LegacyPlannerOptions = field(default_factory=LegacyPlannerOptions)

    @classmethod
    def from_model_config(cls, model_config) -> ArenaOffloadConfig:
        def get(name: str, default: Any = None) -> Any:
            return getattr(model_config, name, default)

        fp8_weights = bool(get("quantize", False)) and get("qtype") in _FP8_QTYPES

        return cls(
            enabled=bool(
                get("layer_offloading", False)
                and get("layer_offloading_smart", False)
            ),
            fp8_forward=fp8_weights and bool(get("layer_offloading_fp8_forward", False)),
            fp8_backward=fp8_weights
            and bool(get("layer_offloading_fp8_grad_input", False)),
            fp8_sampling=fp8_weights
            and bool(get("layer_offloading_fp8_sampling", False)),
            compile_blocks=bool(
                get("compile", False)
                or get("compile_sample", False)
                or get("train_compile_blocks", False)
            ),
            compile_dynamic=(
                None
                if get("compile_dynamic", True) is None
                else bool(get("compile_dynamic", True))
            ),
            compile_dynamic_hints=tuple(
                tuple(hint) for hint in (get("compile_dynamic_hints", ()) or ())
            ),
            simulated_vram_gib=(
                float(get("layer_offloading_simulated_vram_gb") or 0.0) or None
            ),
            wddm_cap_strict=bool(get("layer_offloading_wddm_cap_strict", False)),
            legacy=LegacyPlannerOptions(
                working_reserve_gib=get("layer_offloading_smart_working_reserve_gb"),
                wddm_margin_gib=get("layer_offloading_smart_wddm_margin_gb"),
                wddm_hard_gib=get("layer_offloading_smart_wddm_hard_gb"),
                wddm_spill_reserve_pct=float(
                    get("layer_offloading_wddm_spill_reserve_pct", 0.10) or 0.10
                ),
                block_stream_only=bool(get("layer_offloading_block_stream_only", False)),
                checkpoint_keep_last=max(
                    0, int(get("layer_offloading_checkpoint_keep_last", 0) or 0)
                ),
                prefetch_depth=int(get("layer_offloading_prefetch_depth", 2) or 2),
                eager_promote_free_gib=max(
                    0.0,
                    float(get("layer_offloading_eager_promote_free_gb", 0.0) or 0.0),
                ),
                eager_promote_max_blocks=max(
                    1,
                    int(get("layer_offloading_eager_promote_max_blocks", 4) or 4),
                ),
                sampling_working_reserve_gib=get(
                    "layer_offloading_smart_sampling_working_reserve_gb"
                ),
                sampling_wddm_margin_gib=get(
                    "layer_offloading_smart_sampling_wddm_margin_gb"
                ),
                sampling_wddm_hard_gib=get(
                    "layer_offloading_smart_sampling_wddm_hard_gb", 1.0
                ),
            ),
        )



def prepare_canonical_storage(transformer, adapter, *, defer_blocks: bool = False):
    """Prepare final arena destinations without publishing model Parameters."""
    from ..canonical_arena import CanonicalArena

    blocks = adapter.execution_blocks(transformer)
    entries = {} if defer_blocks else {
        adapter.block_key(transformer, index): list(adapter.leaf_entries(block))
        for index, block in enumerate(blocks)
    }
    arena = CanonicalArena()
    build = arena.prepare(entries, model=transformer)
    return build

def prepare_arena_offload(
    transformer,
    *,
    device,
    adapter,
    config: ArenaOffloadConfig,
    ignore_modules: Sequence[Any] | None = None,
    canonical_build=None,
) -> ArenaOffloadRuntime:
    """Canonicalize the model's execution blocks and prepare the arena runtime.

    Call after the base weights are final and BEFORE the training network is
    applied. Loaders that populate final arena destinations directly pass their
    populated ``canonical_build``; other models use the compatibility source,
    which copies from the already-materialized model. The runtime comes back
    unfinalized; the trainer calls ``finalize()`` once the network is installed.

    The runtime is published on `transformer._arena_offload_runtime`.
    """
    return ArenaOffloadRuntime._prepare(
        transformer,
        device=device,
        adapter=adapter,
        config=config,
        ignore_modules=ignore_modules,
        canonical_build=canonical_build,
    )


def get_arena_runtime(model) -> ArenaOffloadRuntime | None:
    """The arena runtime for `model`, or None. Unwraps Accelerate/DDP/compile."""
    if model is None:
        return None
    return getattr(unwrap(model), RUNTIME_ATTR, None)


def is_arena_offloaded(model) -> bool:
    return get_arena_runtime(model) is not None


def is_memory_managed(model) -> bool:
    """True for either offload backend.

    Replaces bare `hasattr(module, '_memory_manager')` checks in shared code,
    which silently answer "not offloaded" for an arena model.
    """
    if model is None:
        return False
    inner = unwrap(model)
    return hasattr(inner, "_memory_manager") or is_arena_offloaded(inner)


def memory_runtime_owns_compile(model) -> bool:
    """True when the memory runtime compiles the blocks itself.

    Generic block compile must be skipped for these models -- compile ownership
    is exclusive.
    """
    return is_arena_offloaded(model)


def close_arena_offload(model) -> None:
    runtime = get_arena_runtime(model)
    if runtime is not None:
        runtime.close()
