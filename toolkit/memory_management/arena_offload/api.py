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
import warnings

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
_COMPATIBILITY_ALIASES = {
    "layer_offloading_smart_working_reserve_gb": (
        "layer_offloading_smart_headroom_gb",
    ),
    "layer_offloading_smart_wddm_margin_gb": (
        "layer_offloading_smart_buffer_gb",
    ),
    "layer_offloading_smart_wddm_hard_gb": (
        "layer_offloading_smart_hard_buffer_gb",
    ),
    "layer_offloading_smart_sampling_working_reserve_gb": (
        "layer_offloading_smart_sampling_headroom_gb",
    ),
    "layer_offloading_smart_sampling_wddm_margin_gb": (
        "layer_offloading_smart_sampling_buffer_gb",
    ),
    "layer_offloading_smart_sampling_wddm_hard_gb": (
        "layer_offloading_smart_sampling_hard_buffer_gb",
    ),
}


def unwrap(model):
    """Peel Accelerate / DDP / torch.compile wrappers without importing them.

    The arena package must stay importable from a bare CPU test process, so this
    does not go through `toolkit.accelerator.unwrap_model` (which constructs a
    global `Accelerator`).
    """
    return unwrap_memory_model(model)


@dataclass(frozen=True)
class _ArenaPolicyOptions:
    """Internal policy inputs retained while fork job aliases are migrated."""

    working_reserve_gib: float | None = None
    wddm_margin_gib: float | None = None
    wddm_hard_gib: float | None = None
    checkpoint_keep_last: int = 0
    prefetch_depth: int = 2

    sampling_working_reserve_gib: float | None = None
    sampling_wddm_margin_gib: float | None = None
    sampling_wddm_hard_gib: float | None = 1.0


@dataclass(frozen=True)
class ArenaOffloadConfig:
    """The narrow public configuration surface of arena offload.

    Fields prefixed with ``_`` are derived integration details, not additional
    user-facing arena controls.
    """

    enabled: bool = False
    fp8_forward: bool = False
    fp8_backward: bool = False
    fp8_sampling: bool = False
    compile_blocks: bool = False
    _compile_dynamic: bool | None = True
    _compile_dynamic_hints: tuple[tuple[int, int | None, int | None], ...] = ()
    # Validation knob: pretend the card is this many GiB, so small-card
    # behaviour (deeper streaming, tighter caps, a residency plan that cannot
    # fit) is exercisable on a bigger one. 0/None = use the real card.
    _simulated_vram_gib: float | None = None
    _policy: _ArenaPolicyOptions = field(
        default_factory=_ArenaPolicyOptions, repr=False
    )

    @classmethod
    def from_model_config(cls, model_config) -> ArenaOffloadConfig:
        def get(name: str, default: Any = None) -> Any:
            if hasattr(model_config, name):
                return getattr(model_config, name)
            for alias in _COMPATIBILITY_ALIASES.get(name, ()):
                if hasattr(model_config, alias):
                    return getattr(model_config, alias)
            return default

        fp8_weights = bool(get("quantize", False)) and get("qtype") in _FP8_QTYPES
        requested_forward = bool(get("layer_offloading_fp8_forward", False))
        requested_backward = bool(get("layer_offloading_fp8_grad_input", False))
        requested_sampling = bool(get("layer_offloading_fp8_sampling", False))
        ignored = []
        if not fp8_weights:
            ignored.extend(
                name
                for name, requested in (
                    ("fp8_forward", requested_forward),
                    ("fp8_backward", requested_backward),
                    ("fp8_sampling", requested_sampling),
                )
                if requested
            )
        elif requested_backward and not requested_forward:
            ignored.append("fp8_backward_without_fp8_forward")
        if ignored:
            warnings.warn(
                "arena offload ignored irrelevant FP8 options: "
                + ", ".join(ignored),
                RuntimeWarning,
                stacklevel=2,
            )

        return cls(
            enabled=bool(
                get("layer_offloading", False)
                and get("layer_offloading_smart", False)
            ),
            fp8_forward=fp8_weights and requested_forward,
            fp8_backward=fp8_weights
            and requested_forward
            and requested_backward,
            fp8_sampling=fp8_weights
            and requested_sampling,
            compile_blocks=bool(
                get("compile", False)
                or get("compile_sample", False)
                or get("train_compile_blocks", False)
            ),
            _compile_dynamic=(
                None
                if get("compile_dynamic", True) is None
                else bool(get("compile_dynamic", True))
            ),
            _compile_dynamic_hints=tuple(
                tuple(hint) for hint in (get("compile_dynamic_hints", ()) or ())
            ),
            _simulated_vram_gib=(
                float(get("layer_offloading_simulated_vram_gb") or 0.0) or None
            ),
            _policy=_ArenaPolicyOptions(
                working_reserve_gib=get("layer_offloading_smart_working_reserve_gb"),
                wddm_margin_gib=get("layer_offloading_smart_wddm_margin_gb"),
                wddm_hard_gib=get("layer_offloading_smart_wddm_hard_gb"),
                checkpoint_keep_last=max(
                    0, int(get("layer_offloading_checkpoint_keep_last", 0) or 0)
                ),
                prefetch_depth=int(get("layer_offloading_prefetch_depth", 2) or 2),
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

    validator = getattr(adapter, "validate_transformer", None)
    if validator is not None:
        validator(transformer)
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
    if not config.enabled:
        raise ValueError("arena_offload_not_enabled")
    validator = getattr(adapter, "validate_transformer", None)
    if validator is not None:
        validator(transformer)
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
