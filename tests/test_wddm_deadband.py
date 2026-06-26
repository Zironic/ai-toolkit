import unittest

from toolkit.memory_management import MemoryManager


class WddmDeadbandTests(unittest.TestCase):
    """The free-VRAM deadband that governs auto-working_reserve layout moves.

    The hold band is the whole point: inside it the controller makes no move and
    issues no trace reset, so the prefetch schedule survives and auto-working_reserve
    converges to the same steady state as a hand-picked smart_working_reserve value.
    """

    def _act(self, free, **kw):
        return MemoryManager._training_layout_action(
            free, wddm_hard_gib=1.0, wddm_hold_high_gib=2.0, **kw
        )

    def test_below_hard_floor_demotes(self):
        self.assertEqual(self._act(0.5), "down")
        self.assertEqual(self._act(0.99), "down")

    def test_inside_band_holds(self):
        # Hold band is [hard, wddm_hold_high] inclusive on both edges.
        self.assertEqual(self._act(1.0), "hold")
        self.assertEqual(self._act(1.5), "hold")
        self.assertEqual(self._act(2.0), "hold")

    def test_above_band_promotes(self):
        self.assertEqual(self._act(2.01), "up")
        self.assertEqual(self._act(5.0), "up")

    def test_oom_forces_demote_even_with_free(self):
        # An OOM is a breach regardless of the working_reserve math.
        self.assertEqual(self._act(9.0, did_oom=True), "down")

    def test_custom_band(self):
        # Wider band: hold from 0.5..3.0.
        act = MemoryManager._training_layout_action
        self.assertEqual(act(0.4, wddm_hard_gib=0.5, wddm_hold_high_gib=3.0), "down")
        self.assertEqual(act(2.0, wddm_hard_gib=0.5, wddm_hold_high_gib=3.0), "hold")
        self.assertEqual(act(3.5, wddm_hard_gib=0.5, wddm_hold_high_gib=3.0), "up")


if __name__ == "__main__":
    unittest.main()
