import threading
import time
import unittest
from unittest import mock

import torch

from toolkit.memory_management import bounce_pool


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class BouncePoolTests(unittest.TestCase):
    def test_cold_schedule_recovers_from_observed_access_order(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=1, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            for key in ("a", "b", "c", "d"):
                pool.register_source(key, torch.nn.Linear(8, 8))
            pool.set_schedule(["a"])
            weight = torch.empty(4)
            for key in ("b", "c", "d"):
                pool.acquire(key, weight, None)

            pool.step_begin(warmup_bytes=0, warmup_timeout_s=0.0)

            with pool._cv:
                self.assertEqual(pool._scheduled, ["b", "c", "d"])
                self.assertEqual(pool.schedule_shape_key, "observed")
                self.assertEqual(pool.schedule_version, -2)
                self.assertEqual(pool._consume_pos, 0)
        finally:
            pool.shutdown()

    def test_manager_step_begin_promotes_observed_schedule_when_trace_absent(self):
        """End-to-end: the manager wrapper (not pool.step_begin directly) must let
        the observed-order promotion fire when no external trace hands off.

        The pool's own unit test calls ``pool.step_begin`` directly, which hides a
        defeating interaction: in real training the pool is only reached through
        ``MemoryManager.offload_step_begin``, which re-seeds the cold source-order
        schedule every step. If that re-seed wipes ``_observed_step`` (or keeps
        resetting ``schedule_shape_key``), the promotion can never take effect and
        the pool stays pinned to the cold schedule with a 100% hard-miss rate.
        """
        from toolkit.memory_management import MemoryManager

        bounce_pool.destroy_pool("cpu")
        pool = bounce_pool.create_pool(
            "cpu", budget_bytes=1 << 20, lookahead=1, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            for key in ("a", "b", "c", "d"):
                pool.register_source(key, torch.nn.Linear(8, 8))
            weight = torch.empty(4)

            # Step 0: cold start through the manager, then a real access stream
            # that is far longer than the 4-entry source-order seed.
            MemoryManager.offload_step_begin(shape_key=None)
            self.assertEqual(pool.schedule_shape_key, None)
            for key in ("a", "b", "c", "d") * 3:  # 12 accesses vs 4 scheduled
                pool.acquire(key, weight, None)

            # Step 1: the manager re-seeds, but the promotion must still fire.
            MemoryManager.offload_step_begin(shape_key=None)
            self.assertEqual(pool.schedule_shape_key, "observed")
            self.assertEqual(len(pool._scheduled), 12)

            # Step 2: once promoted, the manager must leave it alone (the guard),
            # so the observed schedule persists instead of being re-seeded away.
            for key in ("a", "b", "c", "d") * 3:
                pool.acquire(key, weight, None)
            MemoryManager.offload_step_begin(shape_key=None)
            self.assertEqual(pool.schedule_shape_key, "observed")
            self.assertEqual(len(pool._scheduled), 12)
        finally:
            bounce_pool.destroy_pool("cpu")

    def test_step_begin_does_not_recycle_cpu_filling_buffer(self):
        """A step rollover must not make a worker-owned destination reusable."""
        copy_started = threading.Event()
        release_copy = threading.Event()
        original_rebuild = bounce_pool._rebuild_into

        def blocked_rebuild(src, leaves_iter):
            copy_started.set()
            self.assertTrue(release_copy.wait(timeout=2.0))
            return original_rebuild(src, leaves_iter)

        module = torch.nn.Linear(8, 8)
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=1, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            with mock.patch.object(bounce_pool, "_rebuild_into", blocked_rebuild):
                pool.register_source("layer", module)
                pool.set_schedule(["layer"])
                pool.step_begin()
                self.assertTrue(copy_started.wait(timeout=2.0))

                with pool._cv:
                    filling = pool._slots[0]
                    leaves = filling.leaves
                    signature = filling.signature
                    self.assertEqual(filling.state, bounce_pool.CPU_FILLING)

                pool.step_begin()
                with pool._cv:
                    self.assertNotIn(filling, pool._slots.values())
                    self.assertTrue(all(
                        candidate is not leaves
                        for candidate in pool._free_buffers.get(signature, [])
                    ))
                    self.assertEqual(pool._inflight_bytes, filling.nbytes)

                # Stop new scheduling so the completed buffer remains observable
                # instead of immediately being acquired for another fill.
                pool.set_schedule([])
                release_copy.set()
                self.assertTrue(_wait_until(
                    lambda: any(
                        candidate is leaves
                        for candidate in pool._free_buffers.get(signature, [])
                    )
                ))
                with pool._cv:
                    self.assertEqual(pool._inflight_bytes, 0)
        finally:
            release_copy.set()
            pool.shutdown()

    def test_set_budget_trims_free_buffers_without_killing_filling_slot(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=1, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            signature = ((torch.float32, (4,)),)
            leaves = [torch.empty(4)]
            slot = bounce_pool._Slot()
            slot.position = 0
            slot.state = bounce_pool.CPU_FILLING
            slot.signature = signature
            slot.leaves = leaves
            slot.nbytes = bounce_pool._spec_bytes(signature)
            with pool._cv:
                pool._slots[0] = slot
                pool._inflight_bytes = slot.nbytes
                pool._free_buffers[signature] = [[torch.empty(4)], [torch.empty(4)]]

            pool.set_budget(1)

            with pool._cv:
                self.assertIs(pool._slots[0], slot)
                self.assertEqual(pool._slots[0].state, bounce_pool.CPU_FILLING)
                self.assertEqual(pool._free_buffer_bytes_locked(), 0)
                self.assertEqual(pool._inflight_bytes, slot.nbytes)
        finally:
            pool.shutdown()

    def test_seed_schedule_from_sources_restores_prefetch_after_clear(self):
        module = torch.nn.Linear(8, 8)
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=1,
            target_ready_bytes=1 << 20, num_workers=1, ram_floor_bytes=0,
        )
        try:
            pool.register_source("layer", module)
            pool.set_schedule([])
            pool.seed_schedule_from_sources()
            pool.step_begin(warmup_bytes=1, warmup_timeout_s=0.2)
            self.assertTrue(_wait_until(
                lambda: pool.stats()["ready_gib"] > 0.0,
                timeout=2.0,
            ))
            with pool._cv:
                self.assertEqual(pool._scheduled, ["layer"])
                self.assertIn(0, pool._slots)
        finally:
            pool.shutdown()

    def test_byte_target_limits_fill_even_with_larger_position_lookahead(self):
        module_a = torch.nn.Linear(8, 8)
        module_b = torch.nn.Linear(8, 8)
        nbytes = bounce_pool._spec_bytes(
            bounce_pool._layer_specs(module_a.weight, module_a.bias)
        )
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=8,
            target_ready_bytes=nbytes, num_workers=1, ram_floor_bytes=0,
        )
        try:
            pool.register_source("a", module_a)
            pool.register_source("b", module_b)
            pool.set_schedule(["a", "b"])
            pool.step_begin(warmup_bytes=1, warmup_timeout_s=0.2)
            self.assertTrue(_wait_until(
                lambda: pool.stats()["ready_gib"] > 0.0,
                timeout=2.0,
            ))
            with pool._cv:
                self.assertIn(0, pool._slots)
                self.assertNotIn(1, pool._slots)
                self.assertEqual(pool._fill_pos, 1)
        finally:
            pool.shutdown()



if __name__ == "__main__":
    unittest.main()
