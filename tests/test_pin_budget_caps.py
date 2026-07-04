import os
import unittest
from unittest import mock

import torch

from toolkit.memory_management import bounce_pool
from toolkit.memory_management.manager import MemoryManager
from toolkit.memory_management.manager_modules import _move_params_to_cpu_and_pin

GIB = 1024 ** 3


class _FakeVM:
    def __init__(self, total, available):
        self.total = total
        self.available = available


class _FakePsutil:
    def __init__(self, total, available):
        self._vm = _FakeVM(total, available)

    def virtual_memory(self):
        return self._vm


class _FakeManager:
    def __init__(self, budget_bytes):
        self.pinned_weight_budget_bytes = budget_bytes
        self.pinned_weight_bytes = 0


_ENV = {
    "AI_TOOLKIT_PINNED_WEIGHT_RAM_FLOOR_GIB": "8.0",
    "AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION": "0.25",
    "AI_TOOLKIT_WDDM_DXGI_DISABLE": "1",
}


class CapAutoPinBudgetTests(unittest.TestCase):
    def setUp(self):
        self._saved_pinned = bounce_pool._pinned_bytes_total
        bounce_pool._pinned_bytes_total = 0
        self.addCleanup(lambda: setattr(bounce_pool, "_pinned_bytes_total", self._saved_pinned))

    """The auto pinned-weight budget is capped by the WDDM shared pinned-memory
    proxy, not psutil's reported available pageable RAM. Permanent weight pinning
    page-locks already-loaded CPU weights; the crash mode is shared pinned-memory
    commit, not ordinary available-RAM accounting."""

    def _cap(self, budget, total, available, reserve_bytes=0):
        with mock.patch.dict(os.environ, _ENV):
            with mock.patch.object(
                bounce_pool, "_psutil", _FakePsutil(total, available)
            ):
                return MemoryManager._cap_auto_pin_budget(
                    budget, reserve_bytes=reserve_bytes
                )

    def test_wddm_fraction_binds_on_roomy_box(self):
        # 32 GiB box: total-floor cap 24, WDDM proxy cap 8 -> 8 binds.
        # Reported available RAM is intentionally not part of this cap.
        got = self._cap(12 * GIB, 32 * GIB, 20 * GIB)
        self.assertEqual(got, 8 * GIB)

    def test_reported_available_ram_does_not_bind_weight_pin_budget(self):
        # psutil.available is not the CUDA pinned-memory budget. Even if Windows
        # reports little pageable RAM available, the cap should remain the WDDM
        # proxy fraction for already-loaded weights.
        got = self._cap(12 * GIB, 32 * GIB, 1 * GIB)
        self.assertEqual(got, 8 * GIB)


    def test_bounce_reserve_is_subtracted_before_weight_pin_budget(self):
        got = self._cap(12 * GIB, 32 * GIB, 20 * GIB, reserve_bytes=2 * GIB)
        self.assertEqual(got, 6 * GIB)

    def test_small_budget_passes_through(self):
        got = self._cap(1 * GIB, 32 * GIB, 20 * GIB)
        self.assertEqual(got, 1 * GIB)

    def test_no_psutil_returns_budget(self):
        with mock.patch.object(bounce_pool, "_psutil", None):
            self.assertEqual(
                MemoryManager._cap_auto_pin_budget(12 * GIB), 12 * GIB
            )


@unittest.skipUnless(torch.cuda.is_available(), "pinning requires CUDA")
class LivePinFloorGuardTests(unittest.TestCase):
    """Permanent weight pinning should not be blocked by psutil.available floors."""

    def test_available_floor_env_does_not_block_weight_pinning(self):
        lin = torch.nn.Linear(64, 64, bias=False)
        manager = _FakeManager(budget_bytes=1 << 40)
        with mock.patch.dict(
            os.environ, {"AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_FLOOR_GIB": "99999"}
        ):
            _move_params_to_cpu_and_pin(lin, manager)
        self.assertGreater(manager.pinned_weight_bytes, 0)
        self.assertTrue(lin.weight.data.is_pinned())


if __name__ == "__main__":
    unittest.main()
