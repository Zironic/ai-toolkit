"""Shared fixtures for the memory-management test suite."""

import pytest


@pytest.fixture(autouse=True)
def _release_leaked_pinned_arenas():
    """Release any PinnedWeightArena a test forgot to tear down.

    A leaked arena keeps its flats cudaHostRegister'd for the process
    lifetime; when the CPU allocator later recycles those pages for a fresh
    torch.empty, re-registering them raises 'resource already mapped' and
    every subsequent register-mechanism pin in the suite flips pageable
    (the 763bb75 collision, reproduced order-dependently across tests).
    Production owners release explicitly (_destroy_pinned_arena); this sweep
    is test hygiene only.
    """
    yield
    from toolkit.memory_management.pinned_arena import _LIVE_ARENAS

    # release() discards from _LIVE_ARENAS, so snapshot first.
    for arena in list(_LIVE_ARENAS):
        try:
            arena.release()
        except Exception:
            pass
    _LIVE_ARENAS.clear()
