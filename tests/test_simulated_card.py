"""The simulated-smaller-card knob: a phantom ballast on total AND free.

Pure arithmetic here (no CUDA): the point is that every governing quantity stays
self-consistent, so an 8 GiB simulation on a 12 GiB card plans exactly like the
real 8 GiB card would with the same non-torch tenant.
"""

import unittest
from unittest import mock

from toolkit.memory_management import vram_budget

GIB = 1024 ** 3


class SimulatedCardTests(unittest.TestCase):
    def setUp(self):
        vram_budget.set_simulated_card_bytes(None)
        self.addCleanup(vram_budget.set_simulated_card_bytes, None)

    def _fake_card(self, total_gib=12.0, free_gib=10.0):
        """Patch the two sensors vram_budget reads: driver and NVML."""
        mem_get_info = mock.patch.object(
            vram_budget.torch.cuda,
            "mem_get_info",
            return_value=(int(free_gib * GIB), int(total_gib * GIB)),
        )
        physical = mock.patch.object(
            vram_budget.nvml_meminfo,
            "physical_free_bytes",
            return_value=int(free_gib * GIB),
        )
        mem_get_info.start()
        physical.start()
        self.addCleanup(mem_get_info.stop)
        self.addCleanup(physical.stop)

    def test_off_by_default(self):
        self._fake_card()
        self.assertIsNone(vram_budget.simulated_card_bytes())
        self.assertEqual(vram_budget.simulated_ballast_bytes(0), 0)
        self.assertEqual(vram_budget.device_total_bytes(0), 12 * GIB)
        self.assertEqual(vram_budget.device_free_bytes(0), 10 * GIB)

    def test_ballast_hides_bytes_from_total_and_free(self):
        self._fake_card(total_gib=12.0, free_gib=10.0)
        vram_budget.set_simulated_card_bytes(8 * GIB)

        self.assertEqual(vram_budget.device_total_bytes(0), 8 * GIB)
        # 4 GiB hidden from free as well: the 2 GiB non-torch tenant survives.
        self.assertEqual(vram_budget.device_free_bytes(0), 6 * GIB)
        free, total = vram_budget.device_mem_info(0)
        self.assertEqual((free, total), (6 * GIB, 8 * GIB))
        # The real card is still reachable -- the allocator cap needs it.
        self.assertEqual(vram_budget.real_device_total_bytes(0), 12 * GIB)

    def test_non_torch_is_preserved_so_planning_matches_the_small_card(self):
        self._fake_card(total_gib=12.0, free_gib=10.0)
        vram_budget.set_simulated_card_bytes(8 * GIB)
        with mock.patch.object(
            vram_budget.torch.cuda, "memory_reserved", return_value=GIB
        ), mock.patch.object(
            vram_budget.torch.cuda, "memory_allocated", return_value=GIB
        ), mock.patch.object(
            vram_budget.torch.cuda, "is_available", return_value=True
        ):
            snapshot = vram_budget.DeviceSnapshot.capture("cuda:0")

        self.assertEqual(snapshot.total, 8 * GIB)
        self.assertEqual(snapshot.free, 6 * GIB)
        # used = 8 - 6 = 2 GiB, of which torch reserved 1 -> 1 GiB foreign.
        self.assertEqual(snapshot.used, 2 * GIB)
        self.assertEqual(snapshot.non_torch, GIB)

    def test_free_never_goes_negative(self):
        self._fake_card(total_gib=12.0, free_gib=2.0)
        vram_budget.set_simulated_card_bytes(6 * GIB)
        # A 6 GiB card with 6 GiB already spoken for by others: nothing left,
        # not a negative number.
        self.assertEqual(vram_budget.device_free_bytes(0), 0)

    def test_auto_margin_scales_with_the_simulated_card(self):
        self._fake_card(total_gib=12.0, free_gib=10.0)
        vram_budget.set_simulated_card_bytes(6 * GIB)
        self.assertAlmostEqual(
            vram_budget.auto_margin_gib(0, pct=0.10, floor_gib=0.1), 0.6, places=6
        )

    def test_apply_rejects_a_bigger_card_and_clears_on_none(self):
        self._fake_card(total_gib=12.0, free_gib=10.0)
        with mock.patch.object(
            vram_budget.torch.cuda, "is_available", return_value=True
        ):
            self.assertIsNone(vram_budget.apply_simulated_card(16.0, device=0))
            self.assertIsNone(vram_budget.simulated_card_bytes())

            self.assertEqual(
                vram_budget.apply_simulated_card(8.0, device=0), 8 * GIB
            )
            self.assertEqual(vram_budget.simulated_card_bytes(), 8 * GIB)

            self.assertIsNone(vram_budget.apply_simulated_card(0, device=0))
            self.assertIsNone(vram_budget.simulated_card_bytes())


if __name__ == "__main__":
    unittest.main()
