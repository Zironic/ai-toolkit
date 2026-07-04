"""Off-thread, crash-atomic checkpoint writing.

Saving a LoRA stalls training for seconds even though the payload is tens of MB.
Most of that cost is *not* the disk write -- it is the synchronous device->host
copy and `empty_cache()`/`gc.collect()` bookkeeping -- but the disk write, hash,
and fsync are pure CPU/IO work that has no business on the training thread.

This module provides:

* ``atomic_save_file`` / ``atomic_torch_save`` -- write to ``<path>.tmp``, fsync,
  then ``os.replace`` onto the final path. ``os.replace`` is atomic on POSIX and
  on Windows (``MoveFileEx`` with ``REPLACE_EXISTING``), so a crash mid-write can
  never leave a torn file at the canonical path: a reader sees either the old
  complete file or the new complete file, never a partial one.
* ``AsyncSaver`` -- a single background daemon thread that runs write closures
  submitted from the training thread. The caller is responsible for handing it an
  *immutable snapshot* (CPU tensors that training will not mutate); this class
  only moves the IO off the hot path, it does not snapshot for you.

The training thread must build the snapshot itself (the device->host copy has to
happen at a consistent point, before the next ``optimizer.step``); only the
CPU-side serialize + write is deferred here.
"""

from __future__ import annotations

import os
import queue
import threading
import traceback
from typing import Callable, Optional

import torch
from toolkit.memory_management import pin_manager



def _fsync_path(path: str) -> None:
    """Best-effort flush of a file's data to stable storage.

    Works on Windows (``os.fsync`` -> ``FlushFileBuffers``) and POSIX. Failures
    are swallowed: fsync is a durability hardening, not a correctness
    requirement -- the ``os.replace`` below is what guarantees atomicity.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_save_file(save_dict, path: str, metadata=None, fsync: bool = True) -> None:
    """safetensors write that is atomic at ``path`` even across a crash."""
    from safetensors.torch import save_file

    tmp = path + ".tmp"
    save_file(save_dict, tmp, metadata)
    if fsync:
        _fsync_path(tmp)
    os.replace(tmp, path)


def atomic_torch_save(obj, path: str, fsync: bool = True) -> None:
    """``torch.save`` that is atomic at ``path`` even across a crash."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp, path)


class PinnedStager:
    """Reusable pinned buffer for fast, chunked device->host snapshots.

    The per-tensor ``.to("cpu")`` loop in ``get_state_dict`` forces one CUDA sync
    per tensor; under offload-stream contention that serializes behind the copy
    engine and blows a ~200 ms save up to seconds. This stages tensors into a
    fixed pinned buffer in chunks and syncs *once per chunk* instead.

    The buffer is capped (default 64 MB) and reused across saves, so it adds a
    bounded, budgeted amount to the WDDM shared-memory pin footprint -- not a
    per-save spike that could tip an already-full shared budget over the crash
    cliff. Bigger cap -> fewer syncs -> faster, at more pinned bytes; the caller
    picks the point on that curve.

    Not thread-safe: call ``snapshot`` only from the thread that owns the CUDA
    context (the training thread). The returned tensors are pageable CPU copies,
    independent of the pinned buffer, so they are safe to hand to an AsyncSaver.
    """

    def __init__(self, cap_bytes: int = 64 * 1024 * 1024, register: bool = True):
        self.cap_bytes = max(1, int(cap_bytes))
        self._buf = None
        self._register = register
        self._registered_bytes = 0
        self._pin_handle = None

    def _ensure_buf(self):
        if self._buf is None:
            handle = pin_manager.pin_alloc(
                self.cap_bytes,
                "save_stager",
                required=True,
                mode="training",
            )
            self._pin_handle = handle
            self._buf = handle.tensor
            self._registered_bytes = self.cap_bytes if handle.pinned else 0

    def snapshot(self, items, out_dtype=torch.float16) -> "OrderedDict":
        """items: iterable of (key, device_tensor) -> OrderedDict[key -> cpu tensor]."""
        from collections import OrderedDict

        if not torch.cuda.is_available():
            # No device staging to do; mirror the plain per-tensor path.
            return OrderedDict(
                (k, v.detach().to("cpu").to(out_dtype)) for k, v in items
            )

        self._ensure_buf()
        out = OrderedDict()
        chunk = []  # (key, offset, nbytes, dtype, shape)
        offset = 0

        def drain():
            nonlocal chunk, offset
            if not chunk:
                return
            torch.cuda.synchronize()
            for key, off, nb, dt, shape in chunk:
                view = self._buf[off:off + nb].view(dt).view(shape)
                # .to() allocates a fresh pageable CPU tensor, freeing the slot.
                out[key] = view.to(out_dtype)
            chunk = []
            offset = 0

        for key, v in items:
            v = v.detach()
            elt = v.element_size()
            nb = v.numel() * elt
            # align the slot so .view(dtype) is legal
            if offset % elt:
                offset += elt - (offset % elt)
            if nb > self.cap_bytes:
                # single tensor larger than the whole buffer: copy it directly.
                drain()
                out[key] = v.to("cpu").to(out_dtype)
                continue
            if offset + nb > self.cap_bytes:
                drain()
            view = self._buf[offset:offset + nb].view(v.dtype).view(v.shape)
            view.copy_(v, non_blocking=True)
            chunk.append((key, offset, nb, v.dtype, tuple(v.shape)))
            offset += nb
        drain()
        return out

    def close(self):
        if self._registered_bytes:
            pin_manager.release(self._pin_handle)
            self._registered_bytes = 0
        self._pin_handle = None
        self._buf = None


class _Job:
    __slots__ = ("fn", "coalesce_key", "description")

    def __init__(self, fn: Callable[[], None], coalesce_key, description: str):
        self.fn = fn
        self.coalesce_key = coalesce_key
        self.description = description


_SHUTDOWN = object()


class AsyncSaver:
    """Runs write closures on one background thread, in submission order.

    ``submit(fn, coalesce_key=...)`` with a repeated ``coalesce_key`` keeps only
    the newest pending job for that key (older, not-yet-started ones are dropped
    without running). That is exactly what a high-frequency "latest LoRA wins"
    recovery snapshot wants; leave ``coalesce_key=None`` (the default) for
    distinct periodic checkpoints, which must all be written.

    A failure in a background write is captured and re-raised on the training
    thread at the next ``submit``/``wait_idle``/``close`` -- so a full disk
    surfaces loudly instead of silently dropping checkpoints.
    """

    def __init__(self, name: str = "async-saver"):
        self._q: "queue.Queue" = queue.Queue()
        self._lock = threading.Lock()
        # coalesce_key -> most-recently-submitted job for that key
        self._latest: dict = {}
        self._inflight = 0
        self._idle = threading.Event()
        self._idle.set()
        self._error: Optional[BaseException] = None
        self._error_desc: str = ""
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- training-thread API --------------------------------------------------

    def submit(self, fn: Callable[[], None], coalesce_key=None, description: str = "") -> None:
        self._raise_if_error()
        if self._closed:
            raise RuntimeError("AsyncSaver is closed")
        job = _Job(fn, coalesce_key, description)
        with self._lock:
            if coalesce_key is not None:
                self._latest[coalesce_key] = job
            self._inflight += 1
            self._idle.clear()
        self._q.put(job)

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Block until all submitted writes have finished. Returns False on timeout."""
        ok = self._idle.wait(timeout)
        if ok:
            self._raise_if_error()
        return ok

    def close(self, timeout: Optional[float] = None) -> None:
        """Flush pending writes, stop the worker, and re-raise any error."""
        if self._closed:
            return
        self._closed = True
        self._q.put(_SHUTDOWN)
        self._thread.join(timeout)
        self._raise_if_error()

    # -- worker ---------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job = self._q.get()
            if job is _SHUTDOWN:
                self._q.task_done()
                break
            try:
                # Skip superseded coalesced jobs without writing them.
                if job.coalesce_key is not None:
                    with self._lock:
                        if self._latest.get(job.coalesce_key) is not job:
                            continue
                job.fn()
                if job.coalesce_key is not None:
                    with self._lock:
                        if self._latest.get(job.coalesce_key) is job:
                            del self._latest[job.coalesce_key]
            except BaseException as e:  # noqa: BLE001 - surfaced to training thread
                with self._lock:
                    if self._error is None:
                        self._error = e
                        self._error_desc = job.description
                traceback.print_exc()
            finally:
                with self._lock:
                    self._inflight -= 1
                    if self._inflight <= 0 and self._q.empty():
                        self._idle.set()
                self._q.task_done()

    def _raise_if_error(self) -> None:
        with self._lock:
            err, desc = self._error, self._error_desc
            self._error = None
            self._error_desc = ""
        if err is not None:
            raise RuntimeError(
                f"async checkpoint write failed ({desc or 'unknown'}): {err!r}"
            ) from err
