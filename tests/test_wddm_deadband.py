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
    def test_immutable_promotion_uses_runtime_without_trace_reset(self):
        import torch
        import toolkit.memory_management.manager as manager_mod

        gib = 1024 ** 3

        class FakeRuntime:
            def __init__(self):
                self.calls = []

            def increase_training_residency(
                self,
                available_growth_bytes,
                *,
                max_blocks,
            ):
                self.calls.append((available_growth_bytes, max_blocks))
                return {
                    "added_blocks": ("blocks.0",),
                    "added_leaf_keys": (
                        ("blocks.0", "attn.wq"),
                        ("blocks.0", "attn.wk"),
                    ),
                    "actual_growth_bytes": 256,
                    "previous_plan": object(),
                }

        class FakeManager:
            _smart_training_plan = {
                "resident_bytes": 1024,
                "generic_resident_bytes": 1024,
                "offloaded_layers": 8,
            }

        class FakeModule:
            pass

        runtime = FakeRuntime()
        module = FakeModule()
        module._immutable_runtime = runtime
        mm = FakeManager()
        trace_resets = []

        old_device_free = manager_mod.vram_budget.device_free_bytes
        old_sync = manager_mod.torch.cuda.synchronize
        old_allocatable = MemoryManager._torch_allocatable_bytes
        old_invalidate = MemoryManager._invalidate_manual_training_shape_peaks
        old_reset = MemoryManager.reset_trace_due_to_execution_shape_change
        try:
            manager_mod.vram_budget.device_free_bytes = (
                lambda _device: 4 * gib
            )
            manager_mod.torch.cuda.synchronize = lambda _device: None
            MemoryManager._torch_allocatable_bytes = classmethod(
                lambda cls, _device: 5 * gib
            )
            MemoryManager._invalidate_manual_training_shape_peaks = classmethod(
                lambda cls, _mm: None
            )
            MemoryManager.reset_trace_due_to_execution_shape_change = classmethod(
                lambda cls: trace_resets.append(True)
            )

            changed, action = MemoryManager._promote_training_layer(
                module,
                mm,
                torch.device("cuda:0"),
                cache_pad_gib=0.5,
                wddm_stop_gib=1.0,
            )
        finally:
            manager_mod.vram_budget.device_free_bytes = old_device_free
            manager_mod.torch.cuda.synchronize = old_sync
            MemoryManager._torch_allocatable_bytes = old_allocatable
            MemoryManager._invalidate_manual_training_shape_peaks = old_invalidate
            MemoryManager.reset_trace_due_to_execution_shape_change = old_reset

        self.assertEqual((changed, action), (1, "promote_immutable_block"))
        self.assertEqual(runtime.calls, [(int(3.5 * gib), 1)])
        self.assertEqual(trace_resets, [])
        self.assertEqual(mm._smart_training_plan["resident_bytes"], 1280)
        self.assertEqual(mm._smart_training_plan["offloaded_layers"], 6)

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

class AutoWddmMarginTests(unittest.TestCase):
    def test_auto_margin_scales_with_device_memory_and_floor(self):
        import toolkit.memory_management.manager as manager_mod

        class Props:
            def __init__(self, total_memory):
                self.total_memory = total_memory

        old_get_props = manager_mod.torch.cuda.get_device_properties
        try:
            manager_mod.torch.cuda.get_device_properties = lambda _device: Props(8 * 1024 ** 3)
            self.assertEqual(MemoryManager._auto_wddm_margin_gib("cuda:0"), 1.0)

            manager_mod.torch.cuda.get_device_properties = lambda _device: Props(12 * 1024 ** 3)
            self.assertAlmostEqual(MemoryManager._auto_wddm_margin_gib("cuda:0"), 1.2)

            manager_mod.torch.cuda.get_device_properties = lambda _device: Props(24 * 1024 ** 3)
            self.assertAlmostEqual(MemoryManager._auto_wddm_margin_gib("cuda:0"), 2.4)
        finally:
            manager_mod.torch.cuda.get_device_properties = old_get_props

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


    def test_pre_step_guard_reduces_canonical_before_singleton(self):
        import torch
        import toolkit.memory_management.manager as manager_mod

        gib = 1024 ** 3

        class FakeManager:
            process_device = torch.device("cuda:0")
            _smart_training_plan = {
                "resident_bytes": 4 * gib,
                "generic_resident_bytes": 4 * gib,
                "offloaded_layers": 0,
                "wddm_margin_bytes": 1 * gib,
                "wddm_hard_bytes": 1 * gib,
            }
            _training_autotune_enabled = True

        class FakeExecutor:
            def __init__(self):
                self.calls = []

            def reduce_training_residency(self, required_relief_bytes):
                self.calls.append(required_relief_bytes)
                return {
                    "relieved_bytes": required_relief_bytes,
                    "removed_leaf_keys": (("blocks.0", "mlp.down"),),
                    "remaining_adjustable_bytes": 0,
                }

        class FakeModule:
            pass

        module = FakeModule()
        module._memory_manager = FakeManager()
        module._immutable_runtime = FakeExecutor()
        MemoryManager._record_manual_training_shape_peak(
            module._memory_manager,
            (512, 512),
            peak_allocated_gib=20.0,
            peak_reserved_gib=20.0,
        )
        MemoryManager._record_manual_training_shape_peak(
            module._memory_manager,
            (512, 512),
            peak_allocated_gib=8.0,
            peak_reserved_gib=20.0,
        )

        singleton_calls = []
        old_is_available = manager_mod.torch.cuda.is_available
        old_reserved = manager_mod.torch.cuda.memory_reserved
        old_allocated = manager_mod.torch.cuda.memory_allocated
        old_info = manager_mod.torch.cuda.mem_get_info
        old_local = MemoryManager._dxgi_local_budget_snapshot_bytes
        old_demote = MemoryManager._demote_training_layers
        try:
            manager_mod.torch.cuda.is_available = lambda: True
            manager_mod.torch.cuda.memory_reserved = lambda _device: 4 * gib
            manager_mod.torch.cuda.memory_allocated = lambda _device: 4 * gib
            manager_mod.torch.cuda.mem_get_info = lambda _device: (4 * gib, 12 * gib)
            MemoryManager._dxgi_local_budget_snapshot_bytes = staticmethod(
                lambda _device: {
                    "budget_bytes": 10 * gib,
                    "usage_bytes": 8 * gib,
                    "raw_headroom_bytes": 2 * gib,
                }
            )
            MemoryManager._demote_training_layers = classmethod(
                lambda cls, *_args, **_kwargs: singleton_calls.append(True) or 1
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
            MemoryManager._demote_training_layers = old_demote

        self.assertEqual(result["action"], "prestep_reduce_canonical")
        self.assertEqual(result["demoted_layers"], 0)
        self.assertEqual(singleton_calls, [])
        self.assertEqual(len(module._immutable_runtime.calls), 1)
        self.assertEqual(
            result["canonical_relief"]["relieved_bytes"],
            result["required_relief_bytes"],
        )
        self.assertEqual(
            module._memory_manager._smart_training_plan["resident_bytes"],
            1 * gib,
        )
        self.assertEqual(
            result["after"]["reason"],
            "canonical_layout_changed_relearn_required",
        )

    def test_pre_step_guard_uses_live_peak_then_invalidates_layout(self):
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
        # Cold compile/retrace high-water is ignored. The second observation is
        # the steady-state live peak; reserved remains diagnostic only.
        self.assertIsNone(
            MemoryManager._record_manual_training_shape_peak(
                module._memory_manager,
                (512, 512),
                peak_allocated_gib=20.0,
                peak_reserved_gib=20.0,
            )
        )
        MemoryManager._record_manual_training_shape_peak(
            module._memory_manager,
            (512, 512),
            peak_allocated_gib=8.0,
            peak_reserved_gib=20.0,
        )

        calls = []
        old_is_available = manager_mod.torch.cuda.is_available
        old_reserved = manager_mod.torch.cuda.memory_reserved
        old_allocated = manager_mod.torch.cuda.memory_allocated
        old_info = manager_mod.torch.cuda.mem_get_info
        old_local = MemoryManager._dxgi_local_budget_snapshot_bytes
        old_demote = MemoryManager._demote_training_layers
        try:
            manager_mod.torch.cuda.is_available = lambda: True
            manager_mod.torch.cuda.memory_reserved = lambda _device: 4 * gib
            manager_mod.torch.cuda.memory_allocated = lambda _device: 4 * gib
            manager_mod.torch.cuda.mem_get_info = lambda _device: (4 * gib, 12 * gib)
            MemoryManager._dxgi_local_budget_snapshot_bytes = staticmethod(
                lambda _device: {
                    "budget_bytes": 10 * gib,
                    "usage_bytes": 8 * gib,
                    "raw_headroom_bytes": 2 * gib,
                }
            )

            def fake_demote(_module, mm, count, largest=True):
                calls.append((count, largest))
                return 1

            MemoryManager._demote_training_layers = classmethod(
                lambda cls, _module, mm, count, largest=True: fake_demote(
                    _module, mm, count, largest
                )
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
            MemoryManager._demote_training_layers = old_demote

        self.assertEqual(result["source"], "dxgi_local")
        self.assertEqual(result["action"], "prestep_demote_singleton")
        self.assertEqual(result["before"]["predicted_local_usage_gib"], 12.0)
        self.assertEqual(result["demoted_layers"], 1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(result["after"]["prediction_valid"])
        self.assertEqual(
            result["after"]["reason"],
            "singleton_layout_changed_relearn_required",
        )
        self.assertTrue(result["shape_peak_invalidated"])
        self.assertIsNone(
            MemoryManager._training_shape_peak_bucket(
                module._memory_manager, (512, 512)
            )
        )
