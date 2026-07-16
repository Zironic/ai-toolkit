"""Persist torch.compile's mega-cache across process restarts.

torch.compiler.save_cache_artifacts()/load_cache_artifacts() serialize the
Inductor/AOTAutograd/Triton compile caches into a single portable blob. A
loaded blob that doesn't match the current graph/guards is a safe cache miss
(torch's own keying handles that) -- callers don't need to validate staleness
themselves.
"""

from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import threading
from pathlib import Path

import torch


_TRITON_BUNDLE_LOCK = threading.RLock()
_WINDOWS_TRITON_FALLBACK_INSTALLED = False
_WINDOWS_TRITON_FALLBACK_COPIES = 0

DEFAULT_COMPILE_CACHE_BASENAME = ".torch_compile_cache"
_COMPILE_CACHE_KEY_SCHEMA = "aitk-megacache-v2"


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


def _normalized_checkpoint_identity(model, model_config) -> str:
    checkpoint = getattr(model, "_resolved_checkpoint_path", None)
    if checkpoint is None:
        checkpoint = getattr(model_config, "name_or_path", None)
    if checkpoint is None:
        checkpoint = getattr(model_config, "model_name_or_path", None)
    if checkpoint is None:
        return "unknown"
    checkpoint = os.fspath(checkpoint)
    if os.path.exists(checkpoint):
        return os.path.normcase(os.path.abspath(checkpoint))
    return checkpoint


def model_compile_cache_key(model, model_config=None) -> str:
    """Return the coarse identity for one cumulative model MegaCache.

    Torch guards every entry inside the blob. This identity therefore includes
    compiler-policy inputs that would otherwise create avoidable cache churn,
    but intentionally excludes call shapes, adapter topology, and Arena
    residency. Those variants can safely coexist in one cumulative artifact.
    """
    if model_config is None:
        model_config = getattr(model, "model_config", None)
    if model_config is None:
        raise ValueError("model_compile_cache_key requires a model config")

    try:
        from toolkit.memory_management.arena_offload import DISPATCHER_GENERATION
    except Exception:
        dispatcher_generation = None
    else:
        dispatcher_generation = str(DISPATCHER_GENERATION)

    model_type = type(model)
    identity = {
        "schema": _COMPILE_CACHE_KEY_SCHEMA,
        "torch_version": torch.__version__,
        "model_type": f"{model_type.__module__}.{model_type.__qualname__}",
        "checkpoint": _normalized_checkpoint_identity(model, model_config),
        "arch": getattr(model_config, "arch", None),
        "qtype": getattr(model_config, "qtype", None),
        "compile_mode": getattr(model_config, "compile_mode", "default"),
        "compile_fullgraph": bool(
            getattr(model_config, "compile_fullgraph", False)
        ),
        "compile_dynamic": getattr(model_config, "compile_dynamic", True),
        "compile_dynamic_hints": tuple(
            getattr(model_config, "compile_dynamic_hints", ()) or ()
        ),
        "compile_coordinate_descent": getattr(
            model_config, "compile_coordinate_descent", None
        ),
        "fp8_forward": bool(
            getattr(model_config, "layer_offloading_fp8_forward", False)
        ),
        "fp8_grad_input": bool(
            getattr(model_config, "layer_offloading_fp8_grad_input", False)
        ),
        "dispatcher_generation": dispatcher_generation,
    }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:24]
    arch = re.sub(
        r"[^A-Za-z0-9_.-]+", "_", str(identity["arch"] or model_type.__name__)
    )
    return f"{_COMPILE_CACHE_KEY_SCHEMA}_{arch}_{digest}"


def compile_cache_requested(model_config) -> bool:
    """Whether this model configuration can make a compiled call."""
    return bool(
        getattr(model_config, "compile", False)
        or getattr(model_config, "compile_sample", False)
        or getattr(model_config, "train_compile_blocks", False)
    )


def _dynamo_frame_count() -> int:
    try:
        return int(torch._dynamo.utils.counters["frames"].get("total", 0))
    except Exception:
        return 0


class CompileCacheSession:
    """Best-effort default-on persistence around a lazy compile lifecycle.

    The low-level artifact functions deliberately propagate errors for strict
    diagnostics. Production and ordinary smoke paths use this session so a
    missing, stale, rejected, or unwritable cache degrades to an ordinary cold
    compile instead of failing the workload.
    """

    def __init__(self, cache_dir, key, *, enabled=True, logger=None):
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.key = str(key)
        self.enabled = bool(enabled and self.cache_dir)
        self.logger = logger
        self.path = (
            compile_cache_path(self.cache_dir, self.key) if self.enabled else None
        )
        self.loaded = False
        self.load_attempted = False
        self.saved = False
        self._last_saved_frames = _dynamo_frame_count()
        self._lock = threading.RLock()

    @classmethod
    def for_model(
        cls,
        model,
        model_config=None,
        *,
        default_cache_dir=None,
        compile_enabled=None,
        logger=None,
    ):
        if model_config is None:
            model_config = getattr(model, "model_config", None)
        if model_config is None:
            raise ValueError("CompileCacheSession.for_model requires a model config")
        if compile_enabled is None:
            compile_enabled = compile_cache_requested(model_config)
        enabled = bool(
            compile_enabled and getattr(model_config, "compile_cache", True)
        )
        cache_dir = getattr(model_config, "compile_cache_dir", None)
        if not cache_dir:
            cache_dir = default_cache_dir
        return cls(
            cache_dir,
            model_compile_cache_key(model, model_config),
            enabled=enabled,
            logger=logger,
        )

    def _log(self, message) -> None:
        if self.logger is not None:
            self.logger(message)

    def load(self) -> CompileCacheArtifact | None:
        with self._lock:
            if not self.enabled or self.load_attempted:
                return None
            self.load_attempted = True
            try:
                artifact = load_compile_cache_artifact(self.path)
            except Exception as error:
                self._log(
                    "WARNING: torch.compile MegaCache load failed; continuing "
                    f"with a cold compile: {type(error).__name__}: {error}"
                )
                return None
            self.loaded = artifact is not None
            self._last_saved_frames = _dynamo_frame_count()
            if artifact is not None:
                self._log(
                    "Loaded torch.compile MegaCache "
                    f"({artifact.byte_count / 1024**2:.1f} MiB, key={self.key})"
                )
            return artifact

    def save(self, *, force=False) -> CompileCacheArtifact | None:
        with self._lock:
            if not self.enabled:
                return None
            frames = _dynamo_frame_count()
            if not force and frames <= self._last_saved_frames:
                return None
            try:
                artifact = save_compile_cache_artifact(self.path)
            except Exception as error:
                self._last_saved_frames = frames
                self._log(
                    "WARNING: torch.compile MegaCache save failed; continuing "
                    f"without persistence: {type(error).__name__}: {error}"
                )
                return None
            self._last_saved_frames = frames
            if artifact is not None:
                first_save = not self.saved
                self.saved = True
                if first_save:
                    self._log(
                        "Saved torch.compile MegaCache "
                        f"({artifact.byte_count / 1024**2:.1f} MiB, key={self.key})"
                    )
            return artifact


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
