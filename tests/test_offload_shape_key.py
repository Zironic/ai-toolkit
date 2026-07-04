import unittest

import torch

from toolkit.memory_management import MemoryManager


class OffloadShapeKeyTests(unittest.TestCase):
    def test_preservation_resolution_partitions_macro_step_traces(self):
        batch = {"latents": torch.zeros(1, 4, 64, 64)}

        plain_512 = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=False,
            dop_resolution=None,
            dop_single_backward=False,
            dop_prior_cache=False,
            blank_preservation_enabled=False,
            blank_preservation_resolution=None,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )
        dop_256 = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=True,
            dop_resolution=256,
            dop_single_backward=False,
            dop_prior_cache=False,
            blank_preservation_enabled=False,
            blank_preservation_resolution=None,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )
        dop_full_res = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=True,
            dop_resolution=None,
            dop_single_backward=False,
            dop_prior_cache=False,
            blank_preservation_enabled=False,
            blank_preservation_resolution=None,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )

        self.assertNotEqual(plain_512, dop_256)
        self.assertNotEqual(dop_full_res, dop_256)

    def test_dop_cache_policy_partitions_macro_step_traces(self):
        batch = {"latents": torch.zeros(1, 4, 64, 64)}

        no_cache = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=True,
            dop_resolution=256,
            dop_single_backward=False,
            dop_prior_cache=False,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )
        cache_enabled = MemoryManager.offload_shape_key_from_batch(
            batch,
            dop_enabled=True,
            dop_resolution=256,
            dop_single_backward=False,
            dop_prior_cache=True,
            checkpoint_policy_id=0,
            fp8_forward_enabled=True,
        )

        self.assertNotEqual(no_cache, cache_enabled)

    def test_resolution_change_uses_compatible_trace_provisionally(self):
        from toolkit.memory_management import manager_modules

        policy = (("dop_enabled", False), ("checkpoint_policy_id", 0))
        key_512 = ((("torch.float32", (1, 4, 64, 64)),), policy)
        key_768 = ((("torch.float32", (1, 4, 96, 96)),), policy)
        incompatible = (
            (("torch.float32", (1, 4, 96, 96)),),
            (("dop_enabled", True), ("checkpoint_policy_id", 0)),
        )

        manager_modules.set_offload_trace_enabled(True)
        try:
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None
            manager_modules.offload_step_begin(shape_key=key_512)
            manager_modules.record_weight_access("a", "forward")
            manager_modules.offload_step_end()

            self.assertEqual(
                manager_modules.offload_trace_schedule(key_768),
                [("a", "forward", 0)],
            )
            self.assertEqual(
                manager_modules.offload_trace_schedule_confidence(key_768),
                "compatible",
            )
            self.assertIsNone(manager_modules.offload_trace_schedule(incompatible))
            self.assertEqual(
                manager_modules.offload_trace_schedule_confidence(incompatible),
                "cold",
            )

            manager_modules.invalidate_offload_trace_for_shape(key_768)
            self.assertIsNone(manager_modules.offload_trace_schedule(key_768))
            self.assertEqual(
                manager_modules.offload_trace_schedule_confidence(key_768),
                "cold",
            )
            self.assertEqual(
                manager_modules.offload_trace_schedule(key_512),
                [("a", "forward", 0)],
            )

            manager_modules.offload_step_begin(shape_key=key_768)
            manager_modules.record_weight_access("b", "forward")
            manager_modules.offload_step_end()
            self.assertEqual(
                manager_modules.offload_trace_schedule(key_768),
                [("b", "forward", 0)],
            )
            self.assertEqual(
                manager_modules.offload_trace_schedule_confidence(key_768),
                "exact",
            )
        finally:
            manager_modules.set_offload_trace_enabled(False)
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None

    def test_shifted_occurrence_path_change_requires_fresh_trace(self):
        from toolkit.memory_management import manager_modules

        same_shape = (("torch.float32", (1, 4, 64, 64)),)
        recompute_policy = (("checkpoint_policy_id", "recompute"),)
        no_recompute_policy = (("checkpoint_policy_id", "none"),)
        recompute_key = (same_shape, recompute_policy)
        no_recompute_key = (same_shape, no_recompute_policy)
        manager_modules.set_offload_trace_enabled(True)
        try:
            trace = manager_modules._OFFLOAD_TRACE
            trace.schedule_by_shape_key.clear()
            trace.compatible_fallback_blocked_shape_keys.clear()
            trace.frozen = None

            manager_modules.offload_step_begin(shape_key=recompute_key)
            manager_modules.record_weight_access("a", "forward")
            manager_modules.record_weight_access("a", "recompute")
            manager_modules.record_weight_access("a", "backward")
            manager_modules.offload_step_end()

            self.assertEqual(
                manager_modules.offload_trace_schedule(recompute_key),
                [
                    ("a", "forward", 0),
                    ("a", "recompute", 1),
                    ("a", "backward", 2),
                ],
            )
            self.assertIsNone(manager_modules.offload_trace_schedule(no_recompute_key))
            self.assertEqual(
                manager_modules.offload_trace_schedule_confidence(no_recompute_key),
                "cold",
            )
        finally:
            manager_modules.set_offload_trace_enabled(False)
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.compatible_fallback_blocked_shape_keys.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None

    def test_resident_promotion_does_not_clear_execution_trace(self):
        from toolkit.memory_management import manager_modules

        shape_key = ((("torch.float32", (1, 4, 64, 64)),), (("policy", "same"),))
        manager_modules.set_offload_trace_enabled(True)
        try:
            trace = manager_modules._OFFLOAD_TRACE
            trace.schedule_by_shape_key.clear()
            trace.compatible_fallback_blocked_shape_keys.clear()
            trace.frozen = None
            manager_modules.offload_step_begin(shape_key=shape_key)
            manager_modules.record_weight_access("a", "forward")
            manager_modules.record_weight_access("b", "backward")
            manager_modules.offload_step_end()
            version = manager_modules.offload_trace_version()

            manager_modules.mark_transfer_plan_dirty(reason="resident_promotion")

            self.assertEqual(
                manager_modules.offload_trace_schedule(shape_key),
                [("a", "forward", 0), ("b", "backward", 0)],
            )
            self.assertEqual(
                manager_modules.offload_trace_schedule_confidence(shape_key),
                "exact",
            )
            self.assertGreater(manager_modules.offload_trace_version(), version)
        finally:
            manager_modules.set_offload_trace_enabled(False)
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.compatible_fallback_blocked_shape_keys.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None

    def test_layout_change_rebuilds_transfer_plan_not_trace(self):
        from toolkit.memory_management import MemoryManager, manager_modules

        shape_key = ((("torch.float32", (1, 4, 64, 64)),), (("policy", "same"),))
        manager_modules.set_offload_trace_enabled(True)
        try:
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.compatible_fallback_blocked_shape_keys.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None
            manager_modules.offload_step_begin(shape_key=shape_key)
            manager_modules.record_weight_access("a", "forward")
            manager_modules.record_weight_access("b", "backward")
            manager_modules.offload_step_end()
            version = manager_modules.offload_trace_version()

            MemoryManager.reset_trace_due_to_execution_shape_change()

            self.assertEqual(
                manager_modules.offload_trace_schedule(shape_key),
                [("a", "forward", 0), ("b", "backward", 0)],
            )
            self.assertEqual(
                manager_modules.offload_trace_schedule_confidence(shape_key),
                "exact",
            )
            self.assertGreater(manager_modules.offload_trace_version(), version)
        finally:
            manager_modules.set_offload_trace_enabled(False)
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.compatible_fallback_blocked_shape_keys.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None

    def test_execution_invalidation_clears_blocked_compatible_fallbacks(self):
        from toolkit.memory_management import manager_modules

        policy = (("policy", "same"),)
        key_512 = ((("torch.float32", (1, 4, 64, 64)),), policy)
        key_768 = ((("torch.float32", (1, 4, 96, 96)),), policy)
        manager_modules.set_offload_trace_enabled(True)
        try:
            trace = manager_modules._OFFLOAD_TRACE
            trace.schedule_by_shape_key.clear()
            trace.compatible_fallback_blocked_shape_keys.clear()
            trace.frozen = None
            manager_modules.offload_step_begin(shape_key=key_512)
            manager_modules.record_weight_access("a", "forward")
            manager_modules.offload_step_end()
            manager_modules.invalidate_offload_trace_for_shape(key_768)
            self.assertIsNone(manager_modules.offload_trace_schedule(key_768))

            manager_modules.invalidate_execution_trace()
            manager_modules.offload_step_begin(shape_key=key_512)
            manager_modules.record_weight_access("b", "forward")
            manager_modules.offload_step_end()

            self.assertEqual(
                manager_modules.offload_trace_schedule(key_768),
                [("b", "forward", 0)],
            )
            self.assertEqual(
                manager_modules.offload_trace_schedule_confidence(key_768),
                "compatible",
            )
        finally:
            manager_modules.set_offload_trace_enabled(False)
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.compatible_fallback_blocked_shape_keys.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None

    def test_trace_rediscovery_bumps_version_and_blocks_compatible_fallback(self):
        from toolkit.memory_management import manager_modules

        policy = (("policy", "same"),)
        key_512 = ((("torch.float32", (1, 4, 64, 64)),), policy)
        key_768 = ((("torch.float32", (1, 4, 96, 96)),), policy)
        manager_modules.set_offload_trace_enabled(True)
        old_threshold = manager_modules._TRACE_REDISCOVER_AFTER
        try:
            manager_modules._TRACE_REDISCOVER_AFTER = 2
            trace = manager_modules._OFFLOAD_TRACE
            trace.schedule_by_shape_key.clear()
            trace.compatible_fallback_blocked_shape_keys.clear()
            trace.frozen = None
            trace.consecutive_divergences = 0
            manager_modules.offload_step_begin(shape_key=key_512)
            manager_modules.record_weight_access("a", "forward")
            manager_modules.offload_step_end()
            version = manager_modules.offload_trace_version()

            for _ in range(2):
                manager_modules.offload_step_begin(shape_key=key_512)
                manager_modules.record_weight_access("b", "forward")
                manager_modules.offload_step_end()

            self.assertGreater(manager_modules.offload_trace_version(), version)
            self.assertIsNone(manager_modules.offload_trace_schedule(key_512))
            self.assertIsNone(manager_modules.offload_trace_schedule(key_768))
        finally:
            manager_modules._TRACE_REDISCOVER_AFTER = old_threshold
            manager_modules.set_offload_trace_enabled(False)
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.compatible_fallback_blocked_shape_keys.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None

    def test_schedule_cache_evicts_least_recently_used_past_cap(self):
        """schedule_by_shape_key holds one full per-step access list per shape
        key ever seen (bucketed aspect-ratio datasets can produce dozens to
        hundreds of distinct keys over a run) with no natural end -- it must
        be capped, not grow for the life of the process."""
        from toolkit.memory_management import manager_modules

        manager_modules.set_offload_trace_enabled(True)
        old_cap = manager_modules._TRACE_MAX_SHAPE_KEYS
        try:
            manager_modules._TRACE_MAX_SHAPE_KEYS = 3
            trace = manager_modules._OFFLOAD_TRACE
            trace.schedule_by_shape_key.clear()
            trace.compatible_fallback_blocked_shape_keys.clear()
            trace.frozen = None

            keys = [
                ((("torch.float32", (1, 4, 64 + i, 64)),), (("policy", "same"),))
                for i in range(3)
            ]
            for key in keys:
                manager_modules.offload_step_begin(shape_key=key)
                manager_modules.record_weight_access("a", "forward")
                manager_modules.offload_step_end()
            self.assertEqual(len(trace.schedule_by_shape_key), 3)

            # Re-touch the oldest key so it is no longer the least-recently-used.
            manager_modules.offload_step_begin(shape_key=keys[0])
            manager_modules.record_weight_access("a", "forward")
            manager_modules.offload_step_end()

            # A 4th distinct key pushes the cache over the cap of 3; the
            # least-recently-used surviving key (keys[1], never re-touched)
            # must be the one evicted, not keys[0] (just refreshed).
            new_key = ((("torch.float32", (1, 4, 999, 64)),), (("policy", "same"),))
            manager_modules.offload_step_begin(shape_key=new_key)
            manager_modules.record_weight_access("a", "forward")
            manager_modules.offload_step_end()

            self.assertEqual(len(trace.schedule_by_shape_key), 3)
            self.assertIn(keys[0], trace.schedule_by_shape_key)
            self.assertIn(keys[2], trace.schedule_by_shape_key)
            self.assertIn(new_key, trace.schedule_by_shape_key)
            self.assertNotIn(keys[1], trace.schedule_by_shape_key)
        finally:
            manager_modules._TRACE_MAX_SHAPE_KEYS = old_cap
            manager_modules.set_offload_trace_enabled(False)
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.compatible_fallback_blocked_shape_keys.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None


if __name__ == "__main__":
    unittest.main()