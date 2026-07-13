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


class LayoutMoveSplitSignalTests(unittest.TestCase):
    """``_training_layout_move`` deliberately watches two different signals:
    demote must stay conservative across every resolution bucket (a generous
    low-res step must never license a layout that spills at high-res), but
    promote only needs the CURRENT bucket's own headroom. Requiring every
    bucket to be simultaneously comfortable before promoting meant a
    chronically tight high-res bucket (activation-bound, not fixable by
    shedding resident weight bytes) vetoed promotion forever, even during a
    roomy low-res step -- resident VRAM only ever ratcheted down."""

    def _move(self, demote_free, current_free, **kw):
        kw.setdefault("wddm_hard_gib", 1.0)
        kw.setdefault("wddm_hold_high_gib", 2.0)
        kw.setdefault("did_oom", False)
        return MemoryManager._training_layout_move(demote_free, current_free, **kw)

    def test_promotes_on_current_bucket_headroom_despite_tight_other_bucket(self):
        # Cross-bucket worst-case (1.5, from some other bucket sitting in the
        # hold band -- not low enough to demote, but too low to clear
        # wddm_hold_high) would veto promotion under the old single-signal
        # gate; the current bucket's own 5.0 GiB margin must still win "up".
        self.assertEqual(self._move(demote_free=1.5, current_free=5.0), "up")

    def test_demote_still_wins_and_stays_conservative(self):
        # A tight OTHER bucket recorded below the hard floor must still force
        # "down" even though the current bucket looks comfortable right now --
        # demote's safety is unchanged by this fix.
        self.assertEqual(self._move(demote_free=0.5, current_free=5.0), "down")

    def test_oom_forces_demote_regardless_of_current_headroom(self):
        self.assertEqual(
            self._move(demote_free=9.0, current_free=9.0, did_oom=True), "down"
        )

    def test_holds_when_neither_threshold_cleared(self):
        self.assertEqual(self._move(demote_free=1.2, current_free=1.5), "hold")


class SharedCliffReliefTests(unittest.TestCase):
    def test_no_pressure_holds_when_raw_headroom_clears_margin(self):
        self.assertEqual(
            MemoryManager._shared_cliff_relief_decision(
                shared_raw_headroom_gib=4.0,
                shared_margin_gib=3.0,
                dedicated_free_gib=1.0,
                dedicated_promote_free_gib=2.0,
            ),
            "hold",
        )

    def test_shared_pressure_unpins_when_dedicated_is_tight(self):
        self.assertEqual(
            MemoryManager._shared_cliff_relief_decision(
                shared_raw_headroom_gib=2.5,
                shared_margin_gib=3.0,
                dedicated_free_gib=1.5,
                dedicated_promote_free_gib=2.0,
            ),
            "unpin",
        )

    def test_shared_pressure_promotes_only_when_dedicated_is_roomy(self):
        self.assertEqual(
            MemoryManager._shared_cliff_relief_decision(
                shared_raw_headroom_gib=2.5,
                shared_margin_gib=3.0,
                dedicated_free_gib=2.5,
                dedicated_promote_free_gib=2.0,
            ),
            "promote",
        )

    def test_unpin_relief_picks_largest_pinned_streamed_layer(self):
        import toolkit.memory_management.manager as manager_mod

        class FakeManager:
            _attach_args = {"ignore_modules": []}
            _training_pinned_resident_keys = set()

        class FakeLayer:
            pass

        small = FakeLayer()
        small._mm_pinned_bytes = 10
        large = FakeLayer()
        large._mm_pinned_bytes = 30
        resident = FakeLayer()
        resident._mm_pinned_bytes = 100
        rows = [
            {"module": small, "managed": True, "resident_bytes": 10},
            {"module": large, "managed": True, "resident_bytes": 30},
            {"module": resident, "managed": False, "resident_bytes": 100},
        ]
        calls = []

        def fake_candidates(*_args, **_kwargs):
            return list(rows)

        def fake_unpin(layer):
            calls.append(layer)
            return int(getattr(layer, "_mm_pinned_bytes", 0) or 0)

        old_candidates = MemoryManager._training_layout_candidates
        old_unpin = manager_mod.unpin_layer
        try:
            MemoryManager._training_layout_candidates = staticmethod(fake_candidates)
            manager_mod.unpin_layer = fake_unpin
            changed, action, released = MemoryManager._unpin_training_layer_for_shared_relief(
                object(), FakeManager()
            )
        finally:
            MemoryManager._training_layout_candidates = old_candidates
            manager_mod.unpin_layer = old_unpin

        self.assertEqual(changed, 1)
        self.assertEqual(action, "unpin_shared")
        self.assertEqual(released, 30)
        self.assertEqual(calls, [large])


class NextPromotionLayerTests(unittest.TestCase):
    """The worst-shape promotion guard must size the next promotion off the same
    block ``_promote_training_layer`` will actually convert first -- pinned-
    resident candidates first, then ascending by resident size."""

    def _drive(self, rows):
        class FakeManager:
            _attach_args = {"ignore_modules": []}
            _training_pinned_resident_keys = set()

        def fake_candidates(*_args, **_kwargs):
            return list(rows)

        old = MemoryManager._training_layout_candidates
        try:
            MemoryManager._training_layout_candidates = staticmethod(fake_candidates)
            return MemoryManager._next_promotion_layer_bytes(object(), FakeManager())
        finally:
            MemoryManager._training_layout_candidates = old

    def test_picks_smallest_managed_layer(self):
        rows = [
            {"module": object(), "managed": True, "resident_bytes": 30},
            {"module": object(), "managed": True, "resident_bytes": 10},
            {"module": object(), "managed": False, "resident_bytes": 5},  # resident, skip
        ]
        self.assertEqual(self._drive(rows), 10)

    def test_pinned_resident_is_promoted_before_a_smaller_unpinned(self):
        rows = [
            {"module": object(), "managed": True, "resident_bytes": 10},
            {
                "module": object(),
                "managed": True,
                "resident_bytes": 30,
                "pinned_resident": True,
            },
        ]
        # The 30-byte pinned block sorts first, so the guard must predict with 30.
        self.assertEqual(self._drive(rows), 30)

    def test_zero_when_nothing_streamed(self):
        rows = [
            {"module": object(), "managed": False, "resident_bytes": 5},
            {"module": object(), "managed": False, "resident_bytes": 8},
        ]
        self.assertEqual(self._drive(rows), 0)


if __name__ == "__main__":
    unittest.main()

class DxgiLocalPrestepGuardTests(unittest.TestCase):
    def test_dxgi_prediction_ignores_historical_reserved_peak(self):
        gib = 1024 ** 3
        predicted = MemoryManager._predict_dxgi_local_peak_bytes(
            {"usage_bytes": 8 * gib},
            current_reserved_bytes=4 * gib,
            current_allocated_bytes=4 * gib,
            peak_reserved_bytes=20 * gib,
            peak_allocated_bytes=8 * gib,
        )
        self.assertEqual(predicted, 12 * gib)

    def test_physical_prediction_ignores_historical_reserved_peak(self):
        import torch
        import toolkit.memory_management.manager as manager_mod

        gib = 1024 ** 3

        class FakeManager:
            process_device = torch.device("cuda:0")
            _smart_training_plan = {
                "resident_bytes": 4 * gib,
                "wddm_margin_bytes": 1 * gib,
                "wddm_hard_bytes": 1 * gib,
            }
            _training_autotune_enabled = True

        class FakeModule:
            pass

        module = FakeModule()
        module._memory_manager = FakeManager()
        MemoryManager._record_manual_training_shape_peak(
            module._memory_manager,
            (512, 512),
            peak_allocated_gib=20.0,
            peak_reserved_gib=20.0,
        )
        MemoryManager._record_manual_training_shape_peak(
            module._memory_manager,
            (512, 512),
            peak_allocated_gib=5.0,
            peak_reserved_gib=20.0,
        )

        old_is_available = manager_mod.torch.cuda.is_available
        old_reserved = manager_mod.torch.cuda.memory_reserved
        old_allocated = manager_mod.torch.cuda.memory_allocated
        old_info = manager_mod.torch.cuda.mem_get_info
        old_local = MemoryManager._dxgi_local_budget_snapshot_bytes
        try:
            manager_mod.torch.cuda.is_available = lambda: True
            manager_mod.torch.cuda.memory_reserved = lambda _device: 4 * gib
            manager_mod.torch.cuda.memory_allocated = lambda _device: 4 * gib
            manager_mod.torch.cuda.mem_get_info = lambda _device: (6 * gib, 12 * gib)
            MemoryManager._dxgi_local_budget_snapshot_bytes = staticmethod(
                lambda _device: None
            )
            result = MemoryManager.prepare_training_memory_for_shape(
                module, torch.device("cuda:0"), shape_key=(512, 512)
            )
        finally:
            manager_mod.torch.cuda.is_available = old_is_available
            manager_mod.torch.cuda.memory_reserved = old_reserved
            manager_mod.torch.cuda.memory_allocated = old_allocated
            manager_mod.torch.cuda.mem_get_info = old_info
            MemoryManager._dxgi_local_budget_snapshot_bytes = old_local

        self.assertIsNone(result)
