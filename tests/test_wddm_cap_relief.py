"""Cap violations widen the cap; they do not end the run (unless strict).

The WDDM allocator cap is a tuning lever -- it recycles idle cache on demand and
is the residency controller's cheap inner lever -- so overshooting it must cost
memory, not the run. Strict mode is the validation/attribution setting where the
OOM is the point.
"""

import unittest
from unittest import mock

from toolkit.memory_management import allocator_cap
from toolkit.memory_management import manager as manager_module
from toolkit.memory_management.manager import MemoryManager

GIB = 1024 ** 3


class WddmCapReliefTests(unittest.TestCase):
    def setUp(self):
        MemoryManager.set_wddm_cap_strict(False)
        MemoryManager._wddm_hard_cap_applied.clear()
        manager_module._WDDM_CAP_RELIEF_BYTES.clear()
        self.addCleanup(MemoryManager.set_wddm_cap_strict, False)
        self.addCleanup(MemoryManager._wddm_hard_cap_applied.clear)
        self.addCleanup(manager_module._WDDM_CAP_RELIEF_BYTES.clear)

        self.set_fraction = mock.patch.object(
            manager_module.torch.cuda, "set_per_process_memory_fraction"
        ).start()
        mock.patch.object(
            manager_module.torch.cuda, "is_available", return_value=True
        ).start()
        mock.patch.object(
            manager_module.torch.cuda, "current_device", return_value=0
        ).start()
        mock.patch.object(
            manager_module.vram_budget,
            "real_device_total_bytes",
            return_value=12 * GIB,
        ).start()
        mock.patch.object(manager_module.sys, "platform", "win32").start()
        self.addCleanup(mock.patch.stopall)

    def test_relief_widens_the_cap_and_lets_the_run_continue(self):
        MemoryManager._wddm_hard_cap_applied[0] = 9.0 / 12.0  # 9 GiB cap

        self.assertTrue(MemoryManager.relieve_wddm_cap_after_oom("cuda:0"))

        applied = MemoryManager._wddm_hard_cap_applied[0]
        self.assertAlmostEqual(applied * 12, 9.5, places=4)
        self.set_fraction.assert_called_once_with(applied, 0)
        self.assertEqual(
            manager_module._WDDM_CAP_RELIEF_BYTES[0],
            MemoryManager.WDDM_CAP_RELIEF_BYTES,
        )

        # Each further violation buys another rung.
        self.assertTrue(MemoryManager.relieve_wddm_cap_after_oom("cuda:0"))
        self.assertAlmostEqual(
            MemoryManager._wddm_hard_cap_applied[0] * 12, 10.0, places=4
        )

    def test_strict_mode_declines_so_the_oom_stands(self):
        MemoryManager.set_wddm_cap_strict(True)
        MemoryManager._wddm_hard_cap_applied[0] = 9.0 / 12.0

        self.assertFalse(MemoryManager.relieve_wddm_cap_after_oom("cuda:0"))
        self.set_fraction.assert_not_called()
        self.assertEqual(manager_module._WDDM_CAP_RELIEF_BYTES, {})

    def test_a_physical_oom_is_not_ours_to_forgive(self):
        # Already uncapped: the card, not our cap, is what ran out.
        MemoryManager._wddm_hard_cap_applied[0] = 1.0
        self.assertFalse(MemoryManager.relieve_wddm_cap_after_oom("cuda:0"))

        # No cap applied at all (non-Windows, or before attach).
        MemoryManager._wddm_hard_cap_applied.clear()
        self.assertFalse(MemoryManager.relieve_wddm_cap_after_oom("cuda:0"))
        self.set_fraction.assert_not_called()

    def test_relief_survives_the_next_phase_boundary(self):
        """A cap that snapped back to the cliff bound would re-crash instantly."""
        manager_module._WDDM_CAP_RELIEF_BYTES[0] = GIB
        with mock.patch.object(
            manager_module.vram_budget, "device_total_bytes", return_value=12 * GIB
        ), mock.patch.object(
            manager_module.vram_budget,
            "device_mem_info",
            return_value=(10 * GIB, 12 * GIB),
        ), mock.patch.object(
            manager_module.torch.cuda, "memory_reserved", return_value=0
        ):
            MemoryManager._apply_wddm_hard_allocator_cap("cuda:0", 1.0)

        # cliff bound = 12 - 2 (non_torch) - 1 (hard) = 9 GiB, plus 1 GiB relief.
        applied = MemoryManager._wddm_hard_cap_applied[0]
        self.assertAlmostEqual(applied * 12, 10.0, places=2)


    def test_applied_cap_bytes_reports_the_active_allocator_bound(self):
        MemoryManager._wddm_hard_cap_applied[0] = 9.0 / 12.0
        self.assertEqual(allocator_cap.applied_cap_bytes("cuda:0"), 9 * GIB)


if __name__ == "__main__":
    unittest.main()
