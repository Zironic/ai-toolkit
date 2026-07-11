"""Warn when ANOTHER GPU tenant is why the model cannot stay resident.

Without this the failure mode is invisible: residency silently shrinks, the step
time multiplies, and nothing in the log points at the orphaned job / ComfyUI /
game actually holding the memory. The verdict is a counterfactual -- contention
only earns a warning when it CHANGED THE OUTCOME.
"""

import unittest

from toolkit.memory_management import vram_budget

GIB = 1024 ** 3


def assess(*, foreign_gib, want_gib, have_gib, total_gib=12.0, reserved_gib=2.0):
    """Build an assessment from a desired foreign figure.

    foreign == (total - free) - torch_reserved, so free is back-solved.
    """
    free_gib = total_gib - reserved_gib - foreign_gib
    return vram_budget.assess_foreign_vram(
        total_bytes=int(total_gib * GIB),
        free_bytes=int(free_gib * GIB),
        torch_reserved_bytes=int(reserved_gib * GIB),
        want_bytes=int(want_gib * GIB),
        have_bytes=int(have_gib * GIB),
    )


class ForeignVramAssessmentTests(unittest.TestCase):
    def test_quiet_box_says_nothing(self):
        # ~1.15 GiB of CUDA context + desktop is the cost of doing business.
        report = assess(foreign_gib=1.15, want_gib=10.0, have_gib=5.0)
        self.assertEqual(report.severity, "none")
        self.assertIsNone(
            vram_budget.format_foreign_vram_warning(report, phase="training")
        )

    def test_blocking_when_the_model_would_have_fit(self):
        # 5 GiB foreign (3.5 excess). We hold 7, need 10 -- but 7 + 3.5 >= 10,
        # so the other tenant is precisely why we are streaming.
        report = assess(foreign_gib=5.0, want_gib=10.0, have_gib=7.0)
        self.assertEqual(report.severity, "blocking")
        self.assertTrue(report.is_blocking)
        message = vram_budget.format_foreign_vram_warning(report, phase="training")
        self.assertIn("WARNING", message)
        self.assertIn("WOULD fit", message)

    def test_contributing_when_it_would_not_have_fit_anyway(self):
        # Excess 3.5 GiB, but we only have 3 and need 10: 3 + 3.5 < 10. The
        # tenant costs us residency but is not the reason we cannot fit.
        report = assess(foreign_gib=5.0, want_gib=10.0, have_gib=3.0)
        self.assertEqual(report.severity, "contributing")
        self.assertFalse(report.is_blocking)
        message = vram_budget.format_foreign_vram_warning(report, phase="training")
        # Still a WARNING: 3.5 GiB of lost residency is streamed every step, and
        # volume is set by harm, not by whether full residency was reachable.
        self.assertIn("WARNING", message)
        self.assertIn("heavy VRAM contention", message)

    def test_big_model_that_never_fits_still_warns_loudly(self):
        # Krea2 on a 12 GB card can never be fully resident, so the fit
        # counterfactual can never fire. Lost residency must still be loud.
        report = assess(foreign_gib=7.88, want_gib=10.0, have_gib=0.71)
        self.assertEqual(report.severity, "contributing")
        message = vram_budget.format_foreign_vram_warning(report, phase="sampling")
        self.assertIn("WARNING", message)
        self.assertIn("every step", message)

    def test_modest_excess_is_only_a_note(self):
        # 2.7 foreign -> 1.2 excess: over the floor, under the severe bar.
        report = assess(foreign_gib=2.7, want_gib=10.0, have_gib=5.0)
        self.assertEqual(report.severity, "contributing")
        message = vram_budget.format_foreign_vram_warning(report, phase="training")
        self.assertNotIn("WARNING", message)
        self.assertIn("note", message)

    def test_benign_when_we_fit_regardless(self):
        # Someone else is on the card, but the whole model is resident anyway.
        report = assess(foreign_gib=5.0, want_gib=4.0, have_gib=6.0)
        self.assertEqual(report.severity, "benign")
        self.assertIsNone(
            vram_budget.format_foreign_vram_warning(report, phase="sampling")
        )

    def test_the_production_regression_is_flagged(self):
        # The real 304 s/it run: an orphaned duplicate of the job held ~4 GiB
        # while the sampler wanted the full ~10 GiB model resident.
        report = assess(foreign_gib=5.5, want_gib=10.0, have_gib=6.5)
        self.assertEqual(report.severity, "blocking")
        message = vram_budget.format_foreign_vram_warning(report, phase="sampling")
        self.assertIn("orphaned", message)

    def test_excess_is_measured_above_the_nominal_baseline(self):
        report = assess(foreign_gib=4.0, want_gib=10.0, have_gib=8.0)
        # 4.0 foreign - 1.5 nominal = 2.5 excess
        self.assertAlmostEqual(report.excess_bytes / GIB, 2.5, places=2)
        self.assertAlmostEqual(report.foreign_bytes / GIB, 4.0, places=2)

    def test_excess_just_under_the_floor_stays_quiet(self):
        # 2.4 foreign -> 0.9 excess, below the 1.0 GiB floor: not worth a line.
        report = assess(foreign_gib=2.4, want_gib=10.0, have_gib=9.5)
        self.assertEqual(report.severity, "none")

    def test_foreign_never_negative_when_reserved_exceeds_used(self):
        report = vram_budget.assess_foreign_vram(
            total_bytes=12 * GIB,
            free_bytes=11 * GIB,
            torch_reserved_bytes=8 * GIB,  # nonsensical, must not go negative
            want_bytes=10 * GIB,
            have_bytes=1 * GIB,
        )
        self.assertGreaterEqual(report.foreign_bytes, 0)
        self.assertGreaterEqual(report.excess_bytes, 0)


if __name__ == "__main__":
    unittest.main()
