"""Persist torch.compile's mega-cache across process restarts.

torch.compiler.save_cache_artifacts()/load_cache_artifacts() serialize the
Inductor/AOTAutograd/Triton compile caches into a single portable blob. A
loaded blob that doesn't match the current graph/guards is a safe cache miss
(torch's own keying handles that) -- callers don't need to validate staleness
themselves.
"""

from dataclasses import dataclass
import os
import re
import shutil
import threading
from pathlib import Path

import torch


_TRITON_BUNDLE_LOCK = threading.RLock()
_WINDOWS_TRITON_FALLBACK_INSTALLED = False
_WINDOWS_TRITON_FALLBACK_COPIES = 0


def _recover_windows_triton_bundle_replace(error) -> bool:
    """Recover one PyTorch TritonBundler directory rename on Windows."""
    global _WINDOWS_TRITON_FALLBACK_COPIES

    if os.name != "nt" or getattr(error, "winerror", None) != 5:
        return False
    source_value = getattr(error, "filename", None)
    destination_value = getattr(error, "filename2", None)
    if not source_value or not destination_value:
        return False
    source = Path(source_value)
    destination = Path(destination_value)
    if (
        not source.is_dir()
        or not source.name.startswith("tmp.")
        or source.parent != destination.parent
    ):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    shutil.rmtree(source, ignore_errors=True)
    _WINDOWS_TRITON_FALLBACK_COPIES += 1
    return True


def _install_windows_triton_bundle_fallback() -> None:
    """Make restored Triton bundle emission robust to WinError 5 renames."""
    global _WINDOWS_TRITON_FALLBACK_INSTALLED

    if os.name != "nt" or _WINDOWS_TRITON_FALLBACK_INSTALLED:
        return
    from torch._inductor.triton_bundler import TritonBundler

    original = TritonBundler.read_and_emit

    def guarded_read_and_emit(bundle):
        maximum_recoveries = max(1, len(bundle.kernel_artifacts) + 1)
        with _TRITON_BUNDLE_LOCK:
            for _attempt in range(maximum_recoveries):
                try:
                    return original(bundle)
                except PermissionError as error:
                    if not _recover_windows_triton_bundle_replace(error):
                        raise
            raise RuntimeError(
                "restored Triton bundle exceeded its Windows rename recovery limit"
            )

    TritonBundler.read_and_emit = staticmethod(guarded_read_and_emit)
    _WINDOWS_TRITON_FALLBACK_INSTALLED = True


def windows_triton_bundle_fallback_stats() -> dict[str, int | bool]:
    return {
        "installed": _WINDOWS_TRITON_FALLBACK_INSTALLED,
        "copy_recoveries": _WINDOWS_TRITON_FALLBACK_COPIES,
    }


@dataclass(frozen=True)
class CompileCacheArtifact:
    path: str
    byte_count: int
    info: object

    @property
    def artifact_counts(self) -> dict[str, int]:
        return compile_cache_artifact_counts(self.info)


def compile_cache_artifact_counts(info) -> dict[str, int]:
    if info is None:
        return {}
    return {
        str(kind): len(keys)
        for kind, keys in sorted(info.artifacts.items(), key=lambda item: str(item[0]))
    }


def compile_cache_path(cache_dir: str, key: str) -> str:
    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)
    return os.path.join(cache_dir, f"{safe_key}.torchcompile_cache")


def load_compile_cache_artifact(path) -> CompileCacheArtifact | None:
    path = str(Path(path))
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as file:
        data = file.read()
    _install_windows_triton_bundle_fallback()
    info = torch.compiler.load_cache_artifacts(data)
    if info is None:
        return None
    return CompileCacheArtifact(path=path, byte_count=len(data), info=info)


def save_compile_cache_artifact(path) -> CompileCacheArtifact | None:
    result = torch.compiler.save_cache_artifacts()
    if result is None:
        return None
    artifacts, info = result
    path = str(Path(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as file:
        file.write(artifacts)
    os.replace(tmp_path, path)
    return CompileCacheArtifact(
        path=path,
        byte_count=len(artifacts),
        info=info,
    )


def load_compile_cache(cache_dir: str, key: str) -> bool:
    """Load a saved mega-cache blob for `key` if present. Returns True if loaded."""
    path = compile_cache_path(cache_dir, key)
    return load_compile_cache_artifact(path) is not None


def save_compile_cache(cache_dir: str, key: str) -> bool:
    """Snapshot the current process's compile caches to disk for `key`."""
    path = compile_cache_path(cache_dir, key)
    return save_compile_cache_artifact(path) is not None


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
