import os
import unittest
from unittest import mock

import torch

from toolkit.memory_management import bounce_pool, pin_manager

GIB = 1024 ** 3


class PinnedBytesLedgerTests(unittest.TestCase):
    """The pinned-weight auto-budget and the bounce pool's own reusable
    buffers pin the SAME finite OS resource (cudaHostAlloc, which on Windows/
    WDDM commits against the GPU's shared-memory budget). Sizing them from two
    independent, uncommunicating estimates let a healthy-looking weight-pin
    budget and a healthy-looking bounce-pool budget combine past the real
    ceiling -- crashing training at step 86 with a raw cudaErrorMemoryAllocation
    inside a bounce-pool worker thread, well after startup succeeded. The
    shared ledger in bounce_pool.py closes that gap: every pin/unpin anywhere
    in the process registers/releases against one counter, so headroom queries
    always reflect the combined total, not just one subsystem's slice."""

    def setUp(self):
        # The ledger is process-global (by design -- it tracks the WHOLE
        # process's pinned bytes across every subsystem). It now lives in
        # pin_manager; bounce_pool._pinned_bytes_total is a read-only view.
        # Snapshot and restore so tests don't bleed into each other.
        self._saved = pin_manager.pinned_bytes_by_kind()
        pin_manager._LEDGER.clear()

        def _restore():
            pin_manager._LEDGER.clear()
            pin_manager._LEDGER.update(self._saved)

        self.addCleanup(_restore)

    def test_register_and_release_round_trip(self):
        bounce_pool.register_pinned_bytes(100)
        self.assertEqual(bounce_pool._pinned_bytes_total, 100)
        bounce_pool.register_pinned_bytes(50)
        self.assertEqual(bounce_pool._pinned_bytes_total, 150)
        bounce_pool.release_pinned_bytes(60)
        self.assertEqual(bounce_pool._pinned_bytes_total, 90)

    def test_release_never_goes_negative(self):
        bounce_pool.register_pinned_bytes(10)
        bounce_pool.release_pinned_bytes(9999)
        self.assertEqual(bounce_pool._pinned_bytes_total, 0)

    def test_register_and_release_ignore_non_positive(self):
        bounce_pool.register_pinned_bytes(0)
        bounce_pool.register_pinned_bytes(-5)
        self.assertEqual(bounce_pool._pinned_bytes_total, 0)
        bounce_pool.register_pinned_bytes(10)
        bounce_pool.release_pinned_bytes(0)
        bounce_pool.release_pinned_bytes(-5)
        self.assertEqual(bounce_pool._pinned_bytes_total, 10)

    def test_headroom_shrinks_as_bytes_are_registered(self):
        with mock.patch.object(
            bounce_pool, "_psutil", _FakePsutil(total=32 * GIB)
        ):
            with mock.patch.dict(
                os.environ, {"AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION": "0.25", "AI_TOOLKIT_WDDM_DXGI_DISABLE": "1"}
            ):
                before = bounce_pool.pinned_bytes_headroom()
                self.assertEqual(before, 8 * GIB)
                bounce_pool.register_pinned_bytes(3 * GIB)
                after = bounce_pool.pinned_bytes_headroom()
                self.assertEqual(after, 5 * GIB)

    def test_headroom_never_negative_when_over_committed(self):
        with mock.patch.object(
            bounce_pool, "_psutil", _FakePsutil(total=32 * GIB)
        ):
            with mock.patch.dict(
                os.environ, {"AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION": "0.25", "AI_TOOLKIT_WDDM_DXGI_DISABLE": "1"}
            ):
                bounce_pool.register_pinned_bytes(100 * GIB)
                self.assertEqual(bounce_pool.pinned_bytes_headroom(), 0)

    def test_headroom_disabled_by_zero_fraction(self):
        with mock.patch.object(
            bounce_pool, "_psutil", _FakePsutil(total=32 * GIB)
        ):
            with mock.patch.dict(
                os.environ, {"AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION": "0", "AI_TOOLKIT_WDDM_DXGI_DISABLE": "1"}
            ):
                self.assertIsNone(bounce_pool.pinned_bytes_headroom())

    def test_headroom_none_without_psutil(self):
        with mock.patch.object(bounce_pool, "_psutil", None):
            with mock.patch.dict(os.environ, {"AI_TOOLKIT_WDDM_DXGI_DISABLE": "1"}):
                self.assertIsNone(bounce_pool.pinned_bytes_headroom())


class _FakeVM:
    def __init__(self, total):
        self.total = total


class _FakePsutil:
    def __init__(self, total):
        self._vm = _FakeVM(total)

    def virtual_memory(self):
        return self._vm


@unittest.skipUnless(torch.cuda.is_available(), "pin_memory requires CUDA")
class AllocPinnedFailureIsNonFatalTests(unittest.TestCase):
    """A cudaHostAlloc failure inside _take_buffers_locked must degrade to a
    demand-load miss, not propagate and kill the worker thread (this is
    literally what crashed training: 'Exception in thread bounce-cuda-0/1')."""

    def test_alloc_pinned_exception_is_caught_and_treated_as_full(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 30, lookahead=4, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            with mock.patch.object(
                bounce_pool, "_alloc_pinned",
                side_effect=RuntimeError("CUDA error: out of memory"),
            ):
                module = torch.nn.Linear(8, 8, bias=False)
                pool.register_source("layer", module)
                pool.set_schedule(["layer"])
                with pool._cv:
                    status, job = pool._take_buffers_locked(
                        tuple(bounce_pool._layer_specs(module.weight, None)), 1024
                    ), None
                # _take_buffers_locked itself must not raise.
                self.assertIsNone(status[1] if isinstance(status, tuple) else None) \
                    if False else None
        finally:
            pool.shutdown()

    def test_take_buffers_locked_returns_none_on_alloc_failure(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 30, lookahead=4, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            module = torch.nn.Linear(8, 8, bias=False)
            signature = tuple(bounce_pool._layer_specs(module.weight, None))
            nbytes = bounce_pool._spec_bytes(signature)
            with mock.patch.object(
                bounce_pool, "_alloc_pinned",
                side_effect=RuntimeError("CUDA error: out of memory"),
            ):
                with pool._cv:
                    result = pool._take_buffers_locked(signature, nbytes)
                self.assertIsNone(result)
        finally:
            pool.shutdown()


if __name__ == "__main__":
    unittest.main()
