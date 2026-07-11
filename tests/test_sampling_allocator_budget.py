import unittest

from toolkit.memory_management import vram_budget

GIB = 1024 ** 3


class SamplingAllocatorBudgetTests(unittest.TestCase):
    """Allocated-side sampling budget: plan against the gc target
    (threshold * cap) minus live allocations, instead of driver-free, which
    counts torch's reclaimable idle cache as used. The +hard term cancels the
    planner's later margin subtraction down to (margin - hard), since the
    hard floor is already inside the cap.
    """

    def test_idle_cache_does_not_shrink_budget(self):
        total = 12 * GIB
        cap_fraction = 10.5 / 12.0  # cap 10.5 GiB (total - 1 hard - 0.5 other)
        hard = 1 * GIB
        # 3 GiB live; idle cache does not appear anywhere in the formula.
        budget = vram_budget.sampling_allocator_budget_free_bytes(
            total, 3 * GIB, cap_fraction, hard
        )
        # 0.95 * 10.5 - 3 + 1 = 7.975 GiB
        self.assertAlmostEqual(budget / GIB, 7.975, delta=0.01)

    def test_usable_equals_target_minus_live_wr_and_extra_margin(self):
        # usable = free' - wr - margin must equal
        #          threshold*cap - allocated - wr - (margin - hard)
        total = 12 * GIB
        cap_fraction = 10.0 / 12.0
        hard = 1 * GIB
        margin = int(1.7 * GIB)
        wr = 2 * GIB
        allocated = 3 * GIB
        free_prime = vram_budget.sampling_allocator_budget_free_bytes(
            total, allocated, cap_fraction, hard
        )
        usable = free_prime - wr - margin
        expected = 0.95 * 10.0 * GIB - allocated - wr - (margin - hard)
        self.assertAlmostEqual(usable / GIB, expected / GIB, delta=0.01)

    def test_no_cap_returns_none(self):
        self.assertIsNone(
            vram_budget.sampling_allocator_budget_free_bytes(
                12 * GIB, 3 * GIB, None, 1 * GIB
            )
        )

    def test_custom_threshold(self):
        total = 12 * GIB
        budget = vram_budget.sampling_allocator_budget_free_bytes(
            total, 0, 0.5, 0, gc_threshold=1.0
        )
        self.assertEqual(budget, 6 * GIB)


if __name__ == "__main__":
    unittest.main()
