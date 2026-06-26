"""Invariants for the auto-working_reserve controller, checked via the offline simulator.

These guard the control *dynamics* (converge, hold, no-spill, recover) without a
GPU. The simulator drives the real decision functions
(MemoryManager._available_vram_gib / _training_layout_action), so a regression in
those — or a constant retune that breaks convergence — fails here in milliseconds
instead of after a multi-hour training run.
"""

import importlib.util as u
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = u.spec_from_file_location(
    "sim_working_reserve_controller", ROOT / "scripts" / "sim_working_reserve_controller.py"
)
sim = u.module_from_spec(spec)
# Register before exec: @dataclass resolves field types via sys.modules[__module__].
sys.modules[spec.name] = sim
spec.loader.exec_module(sim)


class WorkingReserveSimTests(unittest.TestCase):
    def _assert_converges_and_holds(self, cfg, history, *, allow_spills=0):
        s = sim.summarize(history, cfg)
        self.assertIsNotNone(s["converged_at"], f"never converged: {s}")
        # The whole point of the hold band: zero layout moves after convergence,
        # so the prefetch trace survives.
        self.assertEqual(s["tail_moves"], 0, f"churned after converging: {s}")
        self.assertTrue(s["settled_in_band"], f"final cycle out of band: {s}")
        self.assertLessEqual(s["spills"], allow_spills, f"too many spills: {s}")
        return s

    def test_single_layer_converges_and_holds(self):
        cfg, history = sim.scenario_default()
        s = self._assert_converges_and_holds(cfg, history)
        # Documents the cost of the current single-layer cadence: slow warmup.
        self.assertGreater(s["promotes"], 40)

    def test_batch_converges_far_faster_than_single(self):
        cfg_s, hist_s = sim.scenario_default()
        cfg_b, hist_b = sim.scenario_batch()
        self._assert_converges_and_holds(cfg_b, hist_b)
        single = sim.summarize(hist_s, cfg_s)
        batch = sim.summarize(hist_b, cfg_b)
        # Batch promotion = far fewer layout changes (= far fewer trace resets)
        # and far earlier convergence.
        self.assertLess(batch["promotes"], single["promotes"])
        self.assertLess(batch["converged_at"], single["converged_at"])

    def test_no_spill_in_steady_state(self):
        cfg, history = sim.scenario_batch()
        self.assertEqual(sim.summarize(history, cfg)["spills"], 0)

    def test_small_card_still_holds_without_spill(self):
        cfg, history = sim.scenario_small_card()
        self._assert_converges_and_holds(cfg, history)

    def test_recovers_from_external_pressure(self):
        cfg, history = sim.scenario_pressure()
        s = sim.summarize(history, cfg)
        # A sudden external app at the ceiling may briefly spill, but the
        # controller must demote, then re-promote once it leaves, and end held.
        self.assertGreater(s["demotes"], 0, f"never demoted under pressure: {s}")
        self.assertTrue(s["settled_in_band"], f"did not re-settle: {s}")
        self.assertEqual(s["tail_moves"], 0, f"churning after recovery: {s}")

    def test_noise_does_not_cause_steady_state_spills(self):
        # Under realistic per-step noise the controller must never settle so close
        # to the ceiling that ordinary activation jitter spills.
        cfg, history = sim.scenario_noisy_batch()
        s = sim.summarize(history, cfg)
        self.assertEqual(s["steady_spills"], 0, f"noise spilled in steady state: {s}")
        self.assertTrue(s["settled_in_band"], f"noise knocked it out of band: {s}")

    def test_smoothing_reduces_noise_churn(self):
        # The deterministic sim hid it, but instantaneous governing churns under
        # noise (every move = a trace reset). EMA smoothing must reduce that.
        cfg_raw, hist_raw = sim.scenario_noisy_batch()
        cfg_sm, hist_sm = sim.scenario_noisy_smoothed()
        raw = sim.summarize(hist_raw, cfg_raw)
        smooth = sim.summarize(hist_sm, cfg_sm)
        self.assertLessEqual(smooth["steady_moves"], raw["steady_moves"])
        self.assertEqual(smooth["steady_spills"], 0)

    def test_available_vram_accounts_for_other_residents(self):
        from toolkit.memory_management import MemoryManager
        # total=16, reserved=9, other = used - reserved = 12 - 9 = 3, safety=0.5
        # available = (16 - 0.5 - 3) - 9 = 3.5
        avail = MemoryManager._available_vram_gib(16.0, 12.0, 9.0, 9.0, safety_gib=0.5)
        self.assertAlmostEqual(avail, 3.5, places=6)
        # More "other" (e.g. a browser) shrinks available one-for-one.
        avail2 = MemoryManager._available_vram_gib(16.0, 14.0, 9.0, 9.0, safety_gib=0.5)
        self.assertAlmostEqual(avail2, 1.5, places=6)


if __name__ == "__main__":
    unittest.main()
