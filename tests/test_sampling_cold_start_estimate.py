import os
import unittest
from unittest import mock

import torch  # noqa: F401  -- keeps import order consistent with the suite

from toolkit.memory_management import MemoryManager, vram_budget

GIB = 1024 ** 3


class EstimatorTests(unittest.TestCase):
    """Shape-aware cold-start reserve, calibrated on the 2026-07 smoke runs:
    512px fp8 ~2.2 GiB extra, 2000px fp8 ~2.8 GiB, 512px dequant ~3.6 GiB."""

    def test_matches_512_calibration_point(self):
        est = vram_budget.estimate_sampling_working_reserve_bytes(
            1024, 512, batch_cfg=False, fp8_native=True
        )
        # base 2.2 + 1536 tokens * 40 KiB ~= 2.26, x1.15 safety ~= 2.60,
        # +1.0 GiB deliberate-overestimate headroom
        self.assertAlmostEqual(est / GIB, 3.60, delta=0.05)

    def test_matches_2000_calibration_point(self):
        est = vram_budget.estimate_sampling_working_reserve_bytes(
            15625, 512, batch_cfg=False, fp8_native=True
        )
        # base 2.2 + 16137 tokens * 40 KiB ~= 2.82, x1.15 ~= 3.24, +1.0 headroom
        self.assertAlmostEqual(est / GIB, 4.24, delta=0.06)

    def test_batch_cfg_scales_token_share_only(self):
        # x2.5, not x2.0: batched CFG also doubles the fp32 intermediates
        # ((2, L, features) fp32, 740 MiB each at 2000px), which the
        # batch-1-calibrated per-token constant underweights -- the x2.0
        # estimate ran ~1 GiB short at 2000px (two demote rounds, 2026-07-08).
        seq = vram_budget.estimate_sampling_working_reserve_bytes(15625, 512)
        bat = vram_budget.estimate_sampling_working_reserve_bytes(
            15625, 512, batch_cfg=True
        )
        token_bytes = 16137 * 40 * 1024
        self.assertAlmostEqual(
            bat - seq, int(token_bytes * 1.5 * 1.15), delta=2 ** 20
        )

    def test_dequant_fallback_adds_pad(self):
        fp8 = vram_budget.estimate_sampling_working_reserve_bytes(1024, 512)
        deq = vram_budget.estimate_sampling_working_reserve_bytes(
            1024, 512, fp8_native=False
        )
        self.assertAlmostEqual((deq - fp8) / GIB, 1.4 * 1.15, delta=0.02)

    def test_monotonic_in_resolution(self):
        sizes = [512, 1024, 1536, 2000]
        estimates = [
            vram_budget.estimate_sampling_working_reserve_bytes(
                (s // 16) ** 2, 512
            )
            for s in sizes
        ]
        self.assertEqual(estimates, sorted(estimates))


class ColdStartPrecedenceTests(unittest.TestCase):
    """Env override beats the hint beats the flat default; a learned measured
    reserve beats all of them (via _resolve_sampling_working_reserve)."""

    def _resolve(self, cold_start_bytes, learned=0):
        return MemoryManager._resolve_sampling_working_reserve(
            None,
            learned,
            cold_start_bytes=cold_start_bytes,
            floor_bytes=int(1.5 * GIB),
            pad_bytes=int(0.5 * GIB),
        )

    def test_hint_used_when_no_learned_value(self):
        reserve, source = self._resolve(int(3.3 * GIB))
        self.assertEqual(reserve, int(3.3 * GIB))
        self.assertEqual(source, "cold-start")

    def test_learned_value_beats_cold_start(self):
        reserve, source = self._resolve(int(3.3 * GIB), learned=int(2.0 * GIB))
        self.assertEqual(reserve, int(2.5 * GIB))  # learned + pad
        self.assertEqual(source, "measured")


if __name__ == "__main__":
    unittest.main()
