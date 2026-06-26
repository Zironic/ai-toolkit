import unittest

from toolkit.memory_management import MemoryManager

GIB = 1024 ** 3


class SamplingWorkingReserveTests(unittest.TestCase):
    """The sampling VRAM reserve is configured independently from training.

    Sampling is forward-only (no optimizer/grad/backward-activation reserve), so
    it gets its own knob (layer_offloading_smart_sampling_working_reserve_gb) feeding
    inference_resident(working_reserve_gib=...). -1/None = auto (learn + converge down,
    floored); >= 0 = a fixed pinned reserve.
    """

    def _resolve(self, working_reserve_gib, learned_bytes):
        return MemoryManager._resolve_sampling_working_reserve(
            working_reserve_gib,
            learned_bytes,
            cold_start_bytes=3 * GIB,
            floor_bytes=int(1.5 * GIB),
            pad_bytes=int(0.5 * GIB),
        )

    def test_auto_cold_start_before_measurement(self):
        for auto in (None, -1.0, "auto"):
            bytes_, source = self._resolve(auto, 0)
            self.assertEqual(bytes_, 3 * GIB)
            self.assertEqual(source, "cold-start")

    def test_auto_converges_to_learned_plus_pad(self):
        # learned 2.0 GiB -> 2.0 + 0.5 pad = 2.5 GiB (above floor)
        bytes_, source = self._resolve(-1.0, 2 * GIB)
        self.assertEqual(bytes_, int(2.5 * GIB))
        self.assertEqual(source, "measured")

    def test_auto_never_below_floor(self):
        # learned 0.33 GiB -> 0.83 GiB < 1.5 floor -> clamp to floor
        bytes_, source = self._resolve(-1.0, int(0.33 * GIB))
        self.assertEqual(bytes_, int(1.5 * GIB))
        self.assertEqual(source, "measured")

    def test_fixed_overrides_learning(self):
        # A pinned value ignores the learned peak entirely.
        bytes_, source = self._resolve(2.0, 5 * GIB)
        self.assertEqual(bytes_, 2 * GIB)
        self.assertEqual(source, "fixed-config")

    def test_fixed_zero_is_allowed(self):
        # 0 is a valid (if aggressive) fixed reserve; the spill guard backs it up.
        bytes_, source = self._resolve(0.0, 0)
        self.assertEqual(bytes_, 0)
        self.assertEqual(source, "fixed-config")


if __name__ == "__main__":
    unittest.main()
