import unittest

from toolkit.memory_management import MemoryManager

GIB = 1024 ** 3


class WddmCapFractionTests(unittest.TestCase):
    """The WDDM hard allocator cap must govern the DEVICE, not just torch.

    Capping torch reserved at (total - hard) alone let device_used reach
    total (observed: reserved 10.97 + non_torch 1.02 = 11.99/11.99 GiB,
    device_free 0.00, silent WDDM paging -- ticket ffc59a1). The fraction
    subtracts the measured non-torch share so device_used <= total - hard.
    """

    def _cap_gib(self, total_gib, free_gib, reserved_gib, hard_gib=1.0):
        fraction = MemoryManager._wddm_cap_fraction(
            int(total_gib * GIB), int(free_gib * GIB), int(reserved_gib * GIB), hard_gib
        )
        return fraction * total_gib

    def test_no_non_torch_matches_total_minus_hard(self):
        # Everything not free is torch's own reserved pool: cap = total - hard.
        self.assertAlmostEqual(self._cap_gib(12.0, 4.0, 8.0), 11.0, places=6)

    def test_non_torch_usage_tightens_the_cap(self):
        # 1 GiB of the card is not torch's: torch must stop 1 GiB earlier.
        self.assertAlmostEqual(self._cap_gib(12.0, 3.0, 8.0), 10.0, places=6)

    def test_smoke_run_numbers_keep_the_device_margin(self):
        # The failing ingraph smoke at sampling start: total 11.99, free 4.63,
        # reserved 6.22 -> non_torch 1.14. Cap lands ~9.85 so reserved can no
        # longer ride to 10.97 while the device sits at 11.99/11.99.
        cap = self._cap_gib(11.99, 4.63, 6.22)
        self.assertAlmostEqual(cap, 11.99 - 1.0 - 1.14, places=6)
        self.assertLess(cap, 10.0)

    def test_reserved_exceeding_used_clamps_non_torch_at_zero(self):
        # (total - free) < reserved can appear transiently; non_torch floors at 0.
        self.assertAlmostEqual(self._cap_gib(12.0, 5.0, 8.0), 11.0, places=6)

    def test_fraction_floor(self):
        # Pathological pressure never caps below 10% of the card.
        fraction = MemoryManager._wddm_cap_fraction(
            int(12 * GIB), int(0.5 * GIB), 0, 4.0
        )
        self.assertAlmostEqual(fraction, 0.1, places=6)

    def test_fraction_ceiling(self):
        fraction = MemoryManager._wddm_cap_fraction(int(12 * GIB), int(12 * GIB), 0, 0.0)
        self.assertAlmostEqual(fraction, 1.0, places=6)


class TrainingGuardPressureTests(unittest.TestCase):
    """The pre-step guard trips when EITHER the DXGI LOCAL or the physical
    (mem_get_info) signal predicts the next peak crosses its floor. DXGI's
    per-process usage/budget can bless a layout the physical view already
    knows overfills the card, and vice versa."""

    def _dxgi(self, pressure):
        return {
            "source": "dxgi_local",
            "pressure": pressure,
            "predicted_local_usage_gib": 10.0,
            "target_local_usage_gib": 10.4,
            "predicted_peak_free_gib": 0.4,
        }

    def _physical(self, pressure):
        return {
            "source": "cuda_free",
            "pressure": pressure,
            "predicted_peak_free_gib": 0.6,
            "target_free_gib": 1.5,
        }

    def test_physical_pressure_trips_even_when_dxgi_is_calm(self):
        merged = MemoryManager._training_guard_pressure(
            self._dxgi(False), self._physical(True)
        )
        self.assertTrue(merged["pressure"])
        self.assertEqual(merged["pressure_sources"], ["cuda_free"])

    def test_dxgi_pressure_trips_even_when_physical_is_calm(self):
        merged = MemoryManager._training_guard_pressure(
            self._dxgi(True), self._physical(False)
        )
        self.assertTrue(merged["pressure"])
        self.assertEqual(merged["pressure_sources"], ["dxgi_local"])

    def test_no_pressure_when_both_calm(self):
        merged = MemoryManager._training_guard_pressure(
            self._dxgi(False), self._physical(False)
        )
        self.assertFalse(merged["pressure"])
        self.assertEqual(merged["pressure_sources"], [])

    def test_merged_keeps_dxgi_fields_and_carries_physical(self):
        merged = MemoryManager._training_guard_pressure(
            self._dxgi(False), self._physical(True)
        )
        self.assertEqual(merged["source"], "dxgi_local")
        self.assertEqual(merged["predicted_local_usage_gib"], 10.0)
        self.assertEqual(merged["physical_predicted_peak_free_gib"], 0.6)
        self.assertEqual(merged["physical_target_free_gib"], 1.5)


if __name__ == "__main__":
    unittest.main()
