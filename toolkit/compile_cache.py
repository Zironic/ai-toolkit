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


def compiler_stance_supported(stance: str) -> bool:
    """Probe whether the installed torch actually accepts this compiler stance.

    `torch.compiler.set_stance` takes a bare string and does NOT validate it
    -- it stores whatever you pass and only raises
    ``RuntimeError("invalid torch.compile stance ...")`` later, the first
    time a compiled call actually dispatches through it. So newer/
    experimental stances (e.g. "aot_eager_then_compile") can't be probed by
    calling set_stance alone; the check has to exercise a real compiled call.

    A version-number floor would also be guesswork here: some installs in
    this repo track torch ahead of the last tagged release, so "torch >= X"
    can't be trusted to mean "has stance Y" (or its absence). Ask torch
    directly by compiling a throwaway function under the stance instead.

    Two calls, not one: the "*_then_compile" stances defer their actual
    dispatch through the stance check to the SECOND call (the first runs
    eager/AOT-eager), so a single call would pass for an unsupported stance
    string too. `backend="eager"` keeps this a pure stance-validity probe --
    Inductor's C++ codegen (needs a system compiler like `cl.exe` on
    Windows) is an unrelated environment dependency this must not conflate
    with stance support.
    """
    import torch._dynamo

    try:
        torch.compiler.set_stance(stance)

        @torch.compile(backend="eager", fullgraph=True)
        def _stance_probe(x):
            return x + 1

        _stance_probe(torch.zeros(1))
        _stance_probe(torch.zeros(1))
    except Exception:
        return False
    finally:
        torch.compiler.set_stance("default")
        torch._dynamo.reset()
    return True
