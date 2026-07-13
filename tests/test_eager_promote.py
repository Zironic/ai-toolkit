"""Eager residency fill (layer_offloading_eager_promote_free_gb).

The default climb promotes one block per cadence window and stops at the WDDM
hold floor, which strands VRAM on a roomy card. The eager knob turns that into a
bulk fill down to a configured free-margin target. These cover the pure block-count
policy. The arena runtime no longer exposes this legacy controller knob.
"""

import unittest

from toolkit.memory_management import vram_budget


def _blocks(**overrides):
    kwargs = dict(
        resident_gib=0.0,
        block_gib=0.4,
        ring_gib=1.0,
        worst_working_reserve_gib=3.0,
        other_gib=2.0,
        total_gib=16.0,
        promote_floor_gib=3.0,
        max_blocks=4,
    )
    kwargs.update(overrides)
    return vram_budget.training_eager_promote_blocks(**kwargs)


class EagerPromoteBlockCountTest(unittest.TestCase):
    def test_fills_up_to_the_per_step_bound(self):
        # free now = 16 - (0 + 1 + 3 + 2) = 10 GiB; room above the 3 GiB floor is
        # 7 GiB = 17 blocks of 0.4, so the per-step bound is what binds.
        self.assertEqual(_blocks(), 4)

    def test_room_binds_below_the_bound(self):
        # free now = 16 - (8.2 + 1 + 3 + 2) = 1.8; room above a 1.0 floor = 0.8 GiB.
        self.assertEqual(
            _blocks(resident_gib=8.2, promote_floor_gib=1.0),
            2,
        )

    def test_zero_when_not_even_one_block_fits(self):
        self.assertEqual(_blocks(resident_gib=9.5, promote_floor_gib=3.0), 0)

    def test_zero_when_already_below_the_floor(self):
        self.assertEqual(_blocks(resident_gib=14.0), 0)

    def test_a_higher_floor_keeps_more_free(self):
        low = _blocks(resident_gib=8.0, promote_floor_gib=1.0, max_blocks=64)
        high = _blocks(resident_gib=8.0, promote_floor_gib=3.0, max_blocks=64)
        self.assertGreater(low, high)

    def test_degenerate_inputs_promote_nothing(self):
        self.assertEqual(_blocks(block_gib=0.0), 0)
        self.assertEqual(_blocks(max_blocks=0), 0)

    def test_prediction_agrees_with_the_count(self):
        # What the controller asserts after choosing k: the k-block promotion still
        # leaves the floor free on the worst measured shape.
        count = _blocks(resident_gib=8.0, promote_floor_gib=1.0, max_blocks=64)
        predicted = vram_budget.training_promotion_worst_shape_free_gib(
            resident_gib=8.0,
            added_block_gib=0.4 * count,
            ring_gib=1.0,
            worst_working_reserve_gib=3.0,
            other_gib=2.0,
            total_gib=16.0,
        )
        self.assertGreaterEqual(predicted, 1.0)


if __name__ == "__main__":
    unittest.main()
