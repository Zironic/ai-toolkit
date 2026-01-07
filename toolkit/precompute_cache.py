"""In-process registry for precomputed artifacts (e.g., preencoded zimage control contexts).

This module provides a minimal, process-local cache for precompute results keyed by
absolute file path. It intentionally does NOT provide persistence across processes or
across runs—it's for sharing results between components in the same process.
"""
from typing import Dict, Optional
import threading
import os
import copy

# Map: normalized_abs_path -> dict(size->torch.Tensor)
_registry: Dict[str, dict] = {}
_lock = threading.Lock()


def _normalize_path(path: str) -> str:
    """Return a normalized, case-normalized absolute path suitable for use as a registry key."""
    if path is None:
        return None
    try:
        # abs -> normpath -> normcase -> realpath (resolve symlinks) for robust matching on Windows/Unix
        return os.path.normcase(os.path.normpath(os.path.realpath(os.path.abspath(path))))
    except Exception:
        # best-effort fallback
        return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def set_preencoded_control_contexts(path: str, contexts: dict):
    """Store a mapping of size->tensor for the given source image path."""
    if path is None:
        return
    key = _normalize_path(path)
    with _lock:
        # store a deep copy to avoid external mutation
        _registry[key] = copy.deepcopy(contexts) if contexts is not None else None


def get_preencoded_control_contexts(path: str) -> Optional[dict]:
    """Return a deep copy of the stored contexts dict or None if not present."""
    if path is None:
        return None
    key = _normalize_path(path)
    with _lock:
        v = _registry.get(key, None)
        return copy.deepcopy(v) if v is not None else None


def clear():
    """Clear the entire registry (useful for tests)."""
    with _lock:
        _registry.clear()


def keys():
    with _lock:
        return list(_registry.keys())
