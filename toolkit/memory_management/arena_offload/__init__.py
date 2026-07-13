"""Arena offload: the block-native weight-streaming backend.

This package is the only supported integration surface for arena offload.
Model integrations and the shared trainer must go through `api`; they must not
construct `CanonicalArena`, `ResidencyState`, `ResidencyPlan`, or the immutable
runtime directly.

Dependency rule (three tiers):

    host_memory (pin_manager, vram_budget, nvml_meminfo, dxgi_meminfo)
        imports neither backend

    arena_offload  -> may import host_memory; must NOT import MemoryManager
    MemoryManager  -> may import host_memory; must NOT import arena_offload

The `must NOT import MemoryManager` half is not true yet: Phase 1 is a
behavior-preserving facade and still delegates planning to the legacy manager.
Phase 2 (`policy.py`) cuts those calls. See
`tasks/open/UPSTREAM_ARENA_EXTRACTION_PLAN.md`.
"""

from .api import (
    ArenaOffloadConfig,
    LegacyPlannerOptions,
    close_arena_offload,
    get_arena_runtime,
    is_arena_offloaded,
    is_memory_managed,
    memory_runtime_owns_compile,
    prepare_canonical_storage,
    prepare_arena_offload,
)
from .runtime import ArenaOffloadRuntime

__all__ = [
    "ArenaOffloadConfig",
    "ArenaOffloadRuntime",
    "LegacyPlannerOptions",
    "close_arena_offload",
    "get_arena_runtime",
    "is_arena_offloaded",
    "is_memory_managed",
    "memory_runtime_owns_compile",
    "prepare_canonical_storage",
    "prepare_arena_offload",
]
