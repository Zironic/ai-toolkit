"""Persist torch.compile's mega-cache across process restarts.

torch.compiler.save_cache_artifacts()/load_cache_artifacts() serialize the
Inductor/AOTAutograd/Triton compile caches into a single portable blob. A
loaded blob that doesn't match the current graph/guards is a safe cache miss
(torch's own keying handles that) -- callers don't need to validate staleness
themselves.
"""

import os
import re

import torch


def _blob_path(cache_dir: str, key: str) -> str:
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return os.path.join(cache_dir, f"{safe_key}.torchcompile_cache")


def load_compile_cache(cache_dir: str, key: str) -> bool:
    """Load a saved mega-cache blob for `key` if present. Returns True if loaded."""
    path = _blob_path(cache_dir, key)
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        data = f.read()
    return torch.compiler.load_cache_artifacts(data) is not None


def save_compile_cache(cache_dir: str, key: str) -> bool:
    """Snapshot the current process's compile caches to disk for `key`."""
    result = torch.compiler.save_cache_artifacts()
    if result is None:
        return False
    artifacts, _info = result
    os.makedirs(cache_dir, exist_ok=True)
    path = _blob_path(cache_dir, key)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(artifacts)
    os.replace(tmp_path, path)
    return True
