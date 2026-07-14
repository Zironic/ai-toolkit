"""Shared startup policy for manual smoke scripts.

Two guards, catching different things:

- ``fail_if_vram_contended``: refuses to start when *foreign* processes
  (the desktop, a game, ComfyUI) already hold VRAM.
- ``gpu_lock`` / ``run_locked``: serializes *our own* GPU scripts against
  each other. Parallel agent sessions cannot see one another, and two
  11+ GiB smokes on a 12 GB card do not merely measure badly -- the second
  OOMs, and can take the first down with it.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK_PATH = REPO_ROOT / ".gpu.lock"
POLL_SECONDS = 5.0
PAGING_LOAD_MODE = (
    "YesIWantToCauseTBOfPagingOnPurposeBecauseImExplicitlyBenchmarkingDiskLoad"
)
SMOKE_DIRECT_LOAD_MODE = "smoke-direct-to-arena"
PRODUCTION_LOAD_MODE = "production-model-load"
LOAD_MODES = (SMOKE_DIRECT_LOAD_MODE, PRODUCTION_LOAD_MODE, PAGING_LOAD_MODE)


class GpuBusy(RuntimeError):
    """Another of our GPU scripts holds the lock."""


class CudaPhysicalFreeMonitor:
    """Sample NVML-backed physical free VRAM across one smoke step."""

    def __init__(self, device, interval_s=0.02):
        self.device = device
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread = None
        self.min_free_bytes = None
        self.total_bytes = None
        self.samples = 0

    def start(self):
        if not torch.cuda.is_available():
            return self
        try:
            from toolkit.memory_management import vram_budget

            free_b, total_b = vram_budget.device_mem_info(self.device)
        except Exception:
            return self
        self.min_free_bytes = int(free_b)
        self.total_bytes = int(total_b)
        self.samples = 1
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        from toolkit.memory_management import vram_budget

        while not self._stop.wait(self.interval_s):
            try:
                free_b, total_b = vram_budget.device_mem_info(self.device)
            except Exception:
                continue
            self.min_free_bytes = min(self.min_free_bytes, int(free_b))
            self.total_bytes = int(total_b)
            self.samples += 1

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.25)
        if self.min_free_bytes is None or self.total_bytes is None:
            return None
        return {
            "min_free_bytes": int(self.min_free_bytes),
            "total_bytes": int(self.total_bytes),
            "samples": int(self.samples),
        }


def lock_path() -> Path:
    override = os.environ.get("AI_TOOLKIT_GPU_LOCK_PATH")
    return Path(override) if override else DEFAULT_LOCK_PATH


def _disabled() -> bool:
    return os.environ.get("AI_TOOLKIT_GPU_LOCK", "1").lower() in (
        "0",
        "false",
        "no",
        "off",
    )


def _wait_requested() -> bool:
    return os.environ.get("AI_TOOLKIT_GPU_LOCK_WAIT", "0").lower() not in (
        "0",
        "false",
        "no",
        "off",
        "",
    )


def _holder_is_alive(record: dict) -> bool:
    """Is the recorded holder still running?

    Checks the process start time as well as the pid: pids are recycled, and a
    lock that mistakes an unrelated new process for its dead holder would block
    the GPU until someone deletes the file by hand -- at which point people
    start deleting it by reflex and the lock is worse than useless.
    """
    pid = record.get("pid")
    if not isinstance(pid, int):
        return False
    try:
        proc = psutil.Process(pid)
        started = float(record.get("started", 0.0))
        # Same pid AND same start time (1s tolerance for clock granularity).
        return abs(proc.create_time() - started) < 1.0
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
        return False


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Unreadable or half-written: treat as stale rather than wedging.
        return None


def _describe(record: dict | None) -> str:
    if not record:
        return "an unreadable lock file"
    age = time.time() - float(record.get("started", time.time()))
    detail = record.get("detail") or ""
    return (
        f"{record.get('name', '?')} (pid {record.get('pid', '?')}, "
        f"running {age / 60:.1f} min{', ' + detail if detail else ''})"
    )


def _try_acquire(path: Path, record: dict) -> dict | None:
    """Claim the lock, or return the live holder's record."""
    payload = json.dumps(record, indent=2).encode("utf-8")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        existing = _read(path)
        if existing is not None and _holder_is_alive(existing):
            return existing
        # Stale (crashed / killed / unreadable): reclaim it.
        print(
            f"[gpu-lock] reclaiming stale lock from {_describe(existing)}",
            file=sys.stderr,
        )
        with contextlib.suppress(OSError):
            path.unlink()
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # Someone else won the reclaim race; report them as the holder.
            return _read(path) or {}
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    return None


@contextlib.contextmanager
def gpu_lock(name: str, *, detail: str = "", wait: bool | None = None):
    """Hold the exclusive GPU lock for the duration of the block."""
    if _disabled():
        yield
        return

    path = lock_path()
    if wait is None:
        wait = _wait_requested()

    proc = psutil.Process()
    record = {
        "name": name,
        "pid": proc.pid,
        "started": proc.create_time(),
        "detail": detail,
        "cmdline": " ".join(sys.argv),
    }

    deadline_notice = False
    while True:
        holder = _try_acquire(path, record)
        if holder is None:
            break
        if not wait:
            raise GpuBusy(
                f"the GPU is held by {_describe(holder)}.\n"
                f"Lock file: {path}\n"
                "Wait for it, re-run with --wait-for-gpu, or set "
                "AI_TOOLKIT_GPU_LOCK=0 to bypass (it will contend for VRAM)."
            )
        if not deadline_notice:
            print(
                f"[gpu-lock] waiting for {_describe(holder)}...",
                file=sys.stderr,
            )
            deadline_notice = True
        time.sleep(POLL_SECONDS)

    try:
        yield
    finally:
        # Only remove our own lock: if we were wrongly reclaimed while running,
        # the file now belongs to someone else and deleting it would hand the
        # GPU to a third process.
        current = _read(path)
        if current and current.get("pid") == record["pid"]:
            with contextlib.suppress(OSError):
                path.unlink()


def acquire_gpu_lock(name: str, *, detail: str = "", wait: bool | None = None) -> None:
    """Hold the lock for the rest of the process.

    For top-level scripts that have no ``main()`` to wrap. Released at exit --
    including on an uncaught exception, since atexit still runs then.
    """
    holder = gpu_lock(name, detail=detail, wait=wait)
    try:
        holder.__enter__()
    except GpuBusy as busy:
        print(f"[gpu-lock] {busy}", file=sys.stderr)
        raise SystemExit(2) from None
    atexit.register(holder.__exit__, None, None, None)


def configure_cuda_smoke_inductor() -> None:
    """Skip Inductor's irrelevant Windows CPU ISA compiler probe.

    CUDA smoke graphs compile through Triton. PyTorch may still validate CPU
    vector ISA support while preparing Inductor, which invokes ``cl.exe`` on
    Windows even when the graph itself is CUDA-only. Marking that ISA probe as
    unavailable avoids the dry compile without enabling a CPU fallback.
    """
    torch._dynamo.config.suppress_errors = False
    if sys.platform == "win32":
        from torch._inductor import config

        config.cpp.vec_isa_ok = False


def run_locked(name: str, entry, *, detail: str = "") -> int:
    """Run a script's main() under the lock; exit 2 (not a traceback) if busy."""
    configure_cuda_smoke_inductor()
    try:
        if "--no-gpu-lock" in sys.argv:
            result = entry()
        else:
            with gpu_lock(
                name,
                detail=detail,
                wait="--wait-for-gpu" in sys.argv,
            ):
                result = entry()
    except GpuBusy as busy:
        print(f"[gpu-lock] {busy}", file=sys.stderr)
        return 2
    return 0 if result is None else int(result)


def add_lock_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--wait-for-gpu",
        action="store_true",
        help="queue behind another smoke/bench instead of refusing to start.",
    )
    parser.add_argument(
        "--no-gpu-lock",
        action="store_true",
        help="skip the GPU lock entirely (will contend for VRAM).",
    )


def add_load_mode_arg(
    parser: argparse.ArgumentParser, *, default: str = SMOKE_DIRECT_LOAD_MODE
) -> None:
    """Add the explicit full-model smoke loading lifecycle selector."""
    parser.add_argument(
        "--load-mode",
        choices=LOAD_MODES,
        default=default,
        help=(
            "smoke-direct-to-arena is the intended smoke/benchmark mode and "
            "populates canonical storage during checkpoint load; "
            "production-model-load mirrors the production generic load "
            "session and is for testing production loading code; the "
            "deliberately long alternative opts into legacy load-then-copy "
            "behavior, which can cause heavy paging "
            "(default: %(default)s)"
        ),
    )


def configure_smoke_load_mode(model, load_mode: str) -> None:
    """Apply a smoke-only loading mode without changing production config."""
    if load_mode not in LOAD_MODES:
        raise ValueError(f"unknown_smoke_load_mode:{load_mode}")
    model._smoke_direct_arena_load = load_mode == SMOKE_DIRECT_LOAD_MODE


def smoke_model_load_session(model, load_mode: str):
    """Return the production context only when that lifecycle is requested."""
    if load_mode not in LOAD_MODES:
        raise ValueError(f"unknown_smoke_load_mode:{load_mode}")
    if load_mode != PRODUCTION_LOAD_MODE:
        return contextlib.nullcontext()
    from toolkit.memory_management.arena_offload import model_load_arena_session

    return model_load_arena_session(model)


def assert_smoke_load_mode(model, load_mode: str) -> None:
    """Fail when a direct-capable model did not exercise the requested mode."""
    direct = getattr(model, "_prepared_canonical_build", None) is not None
    expected = load_mode == SMOKE_DIRECT_LOAD_MODE
    if direct != expected:
        actual = SMOKE_DIRECT_LOAD_MODE if direct else "non-direct"
        raise RuntimeError(
            f"requested smoke load mode {load_mode!r}, got {actual!r}"
        )


VRAM_CONTENTION_LIMIT = 0.30
GIB = 1024 ** 3


def add_contention_args(parser) -> None:
    parser.add_argument(
        "--ignore-contention",
        action="store_true",
        help=(
            "run even when more than 30%% of VRAM is already in use at "
            "startup"
        ),
    )


def fail_if_vram_contended(
    device,
    *,
    ignore_contention: bool,
    limit: float = VRAM_CONTENTION_LIMIT,
) -> None:
    """Reject a CUDA smoke when startup VRAM usage exceeds ``limit``."""
    device = torch.device(device)
    if ignore_contention or device.type != "cuda" or not torch.cuda.is_available():
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = int(total_bytes) - int(free_bytes)
    used_fraction = used_bytes / int(total_bytes)
    if used_fraction <= float(limit):
        return

    raise SystemExit(
        "CUDA smoke refused to start: "
        f"{used_fraction * 100.0:.1f}% VRAM is already in use "
        f"({used_bytes / GIB:.2f}/{int(total_bytes) / GIB:.2f} GiB), above "
        f"the {float(limit) * 100.0:.0f}% contention limit. Close other GPU "
        "workloads or pass --ignore-contention to run anyway."
    )
