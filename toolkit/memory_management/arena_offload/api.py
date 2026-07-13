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

from ..runtime import (
    RUNTIME_ATTR,
    close_memory_runtime,
    get_memory_runtime,
    is_memory_managed,
    memory_runtime_owns_compile,
    unwrap_memory_model,
)
from .runtime import ArenaOffloadRuntime

_FP8_QTYPES = ("qfloat8", "float8")


def unwrap(model):
    """Peel Accelerate / DDP / torch.compile wrappers without importing them.

    The arena package must stay importable from a bare CPU test process, so this
    does not go through `toolkit.accelerator.unwrap_model` (which constructs a
    global `Accelerator`).
    """
    return unwrap_memory_model(model)


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



def prepare_canonical_storage(
    transformer, adapter, *, device=None, defer_blocks: bool = False
):
    """Prepare final arena destinations without publishing model Parameters."""
    from ..canonical_arena import CanonicalArena
    from .resources import ArenaRuntimeResources

    resources = None
    if device is not None:
        resources = ArenaRuntimeResources(transformer, device)
        resources.acquire_process_owner()
    try:
        blocks = adapter.execution_blocks(transformer)
        entries = {} if defer_blocks else {
            adapter.block_key(transformer, index): list(adapter.leaf_entries(block))
            for index, block in enumerate(blocks)
        }
        arena = CanonicalArena()
        build = arena.prepare(entries, model=transformer)
        if resources is not None:
            resources.adopt_canonical_build(build)
        return build
    except BaseException:
        if resources is not None:
            resources.release()
        raise

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
    return get_memory_runtime(model)


def is_arena_offloaded(model) -> bool:
    return get_arena_runtime(model) is not None


def close_arena_offload(model) -> None:
    close_memory_runtime(model)
