import json
import os
import tempfile
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
                self.assertEqual(pool._scheduled, [("b", "forward", 0), ("c", "forward", 0), ("d", "forward", 0)])
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

    def test_acquire_resyncs_exact_semantic_access(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=4, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            weight = torch.empty(4)
            pool.set_schedule([
                ("a", "forward", 0),
                ("b", "forward", 0),
                ("a", "backward", 1),
                ("b", "backward", 1),
            ])
            pool.acquire("b", weight, None, operation="forward")
            stats = pool.stats()
            self.assertEqual(stats["resyncs"], 1)
            self.assertEqual(stats["mismatches"], 0)
            self.assertEqual(stats["consume_pos"], 2)
        finally:
            pool.shutdown()

    def test_resync_does_not_jump_to_later_duplicate_layer(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=4, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            weight = torch.empty(4)
            pool.set_schedule([
                ("a", "forward", 0),
                ("c", "forward", 0),
                ("a", "backward", 1),
                ("b", "backward", 1),
            ])
            pool.acquire("b", weight, None, operation="forward")
            stats = pool.stats()
            self.assertEqual(stats["resyncs"], 0)
            self.assertEqual(stats["mismatches"], 1)
            self.assertEqual(stats["duplicate_key_resync_blocked"], 1)
            self.assertEqual(stats["consume_pos"], 1)
        finally:
            pool.shutdown()

    def test_consume_without_transfer_resyncs_exact_semantic_access(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=4, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            pool.set_schedule([
                ("a", "forward", 0),
                ("b", "forward", 0),
                ("a", "backward", 1),
                ("b", "backward", 1),
            ])
            pool.consume_without_transfer("b", operation="forward")
            stats = pool.stats()
            self.assertEqual(stats["resyncs"], 1)
            self.assertEqual(stats["mismatches"], 0)
            self.assertEqual(stats["skips"], 1)
            self.assertEqual(stats["consume_pos"], 2)
        finally:
            pool.shutdown()

    def test_trace_schedule_preserves_operation_and_occurrence(self):
        from toolkit.memory_management import manager_modules

        manager_modules.set_offload_trace_enabled(True)
        try:
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None
            manager_modules._OFFLOAD_TRACE.version = 0
            manager_modules.offload_step_begin(shape_key="shape-a")
            manager_modules.record_weight_access("a", "forward")
            manager_modules.record_weight_access("b", "forward")
            manager_modules.record_weight_access("a", "backward")
            manager_modules.offload_step_end()

            expected = [
                ("a", "forward", 0),
                ("b", "forward", 0),
                ("a", "backward", 1),
            ]
            self.assertEqual(
                manager_modules.offload_trace_schedule("shape-a"),
                expected,
            )

            version = manager_modules.offload_trace_version()
            manager_modules.mark_transfer_plan_dirty()
            self.assertEqual(
                manager_modules.offload_trace_schedule("shape-a"),
                expected,
            )
            self.assertGreater(manager_modules.offload_trace_version(), version)
        finally:
            manager_modules.set_offload_trace_enabled(False)
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None

    def test_transfer_schedule_filters_to_registered_sources(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=4, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            module_b = torch.nn.Linear(8, 8)
            weight = torch.empty(4)
            pool.register_source("b", module_b)
            pool.set_schedule(
                [
                    ("a", "forward", 0),
                    ("b", "forward", 0),
                    ("c", "forward", 0),
                    ("b", "backward", 1),
                    ("a", "backward", 1),
                ],
                filter_to_sources=True,
            )

            pool.acquire("b", weight, None, operation="forward")
            pool.acquire("b", weight, None, operation="backward")

            with pool._cv:
                self.assertEqual(
                    pool._scheduled,
                    [("b", "forward", 0), ("b", "backward", 1)],
                )
            stats = pool.stats()
            self.assertEqual(stats["mismatches"], 0)
            self.assertEqual(stats["duplicate_key_resync_blocked"], 0)
            self.assertEqual(stats["consume_pos"], 2)
        finally:
            pool.shutdown()

    def test_sync_sources_replaces_sources_without_clearing_schedule(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=2, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            module_a = torch.nn.Linear(8, 8)
            module_b = torch.nn.Linear(8, 8)
            pool.register_source("a", module_a)
            pool.register_source("b", module_b)
            pool.set_schedule([("a", "forward", 0), ("b", "forward", 0)])

            pool.sync_sources([("b", module_b)])

            with pool._cv:
                self.assertNotIn("a", pool._sources)
                self.assertIn("b", pool._sources)
                self.assertEqual(
                    pool._scheduled,
                    [("a", "forward", 0), ("b", "forward", 0)],
                )
        finally:
            pool.shutdown()

    def test_trace_capture_writes_replay_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_path = os.path.join(tmp, "capture.jsonl")
            old_path = bounce_pool._TRACE_CAPTURE_PATH
            old_limit = bounce_pool._TRACE_CAPTURE_STEPS
            bounce_pool.configure_trace_capture(capture_path, 1)
            pool = None
            try:
                pool = bounce_pool.PinnedBouncePool(
                    "cpu", budget_bytes=1 << 20, lookahead=2, num_workers=1,
                    ram_floor_bytes=0,
                )
                weight = torch.empty(4)
                pool.set_schedule([("a", "forward", 0), ("b", "forward", 0)])
                pool.acquire("b", weight, None, operation="forward")

                pool.step_begin(warmup_bytes=0, warmup_timeout_s=0.0)

                with open(capture_path, "r", encoding="utf-8") as handle:
                    records = [json.loads(line) for line in handle if line.strip()]
                self.assertEqual(len(records), 1)
                self.assertEqual(
                    records[0]["schedule"],
                    [["a", "forward", 0], ["b", "forward", 0]],
                )
                self.assertEqual(records[0]["observed"], [["b", "forward", 0]])
                self.assertEqual(records[0]["resyncs"], 1)
            finally:
                if pool is not None:
                    pool.shutdown()
                bounce_pool.configure_trace_capture(old_path, old_limit)

    def test_trace_capture_configuration_updates_existing_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture_path = os.path.join(tmp, "capture.jsonl")
            old_path = bounce_pool._TRACE_CAPTURE_PATH
            old_limit = bounce_pool._TRACE_CAPTURE_STEPS
            pool = None
            try:
                bounce_pool.configure_trace_capture(None, 256)
                pool = bounce_pool.create_pool(
                    "cpu", budget_bytes=1 << 20, lookahead=2, num_workers=1,
                    ram_floor_bytes=0,
                )
                bounce_pool.configure_trace_capture(capture_path, 1)
                pool.set_schedule([("a", "forward", 0)])
                pool.acquire("a", torch.empty(4), None, operation="forward")

                pool.step_begin(warmup_bytes=0, warmup_timeout_s=0.0)

                with open(capture_path, "r", encoding="utf-8") as handle:
                    records = [json.loads(line) for line in handle if line.strip()]
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]["observed"], [["a", "forward", 0]])
            finally:
                if pool is not None:
                    bounce_pool.destroy_pool("cpu")
                bounce_pool.configure_trace_capture(old_path, old_limit)

    def test_ready_slot_wrong_layer_or_shape_is_hard_miss(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=2, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            requested = torch.nn.Linear(4, 4)
            wrong = torch.nn.Linear(8, 8)
            pool.register_source("requested", requested)
            pool.register_source("wrong", wrong)
            pool.set_schedule([("requested", "forward", 0)])

            wrong_signature = tuple(bounce_pool._layer_specs(wrong.weight, wrong.bias))
            wrong_leaves = bounce_pool._alloc_pinned(wrong_signature)
            slot = bounce_pool._Slot()
            slot.position = 0
            slot.layer_key = "wrong"
            slot.state = bounce_pool.CPU_READY
            slot.signature = wrong_signature
            slot.leaves = wrong_leaves
            slot.weight = wrong.weight
            slot.bias = wrong.bias
            slot.nbytes = bounce_pool._spec_bytes(wrong_signature)
            with pool._cv:
                pool._slots[0] = slot
                pool._inflight_bytes = slot.nbytes

            weight, bias, ticket = pool.acquire(
                "requested", requested.weight, requested.bias, operation="forward"
            )

            self.assertIs(weight, requested.weight)
            self.assertIs(bias, requested.bias)
            self.assertIsNone(ticket)
            stats = pool.stats()
            self.assertEqual(stats["hits"], 0)
            self.assertEqual(stats["hard_misses"], 1)
            with pool._cv:
                self.assertNotIn(0, pool._slots)
        finally:
            pool.shutdown()
    def test_demote_layer_preserves_named_layer_key(self):
        from toolkit.memory_management.manager import MemoryManager

        root = torch.nn.Sequential(torch.nn.Linear(4, 4))
        mm = MemoryManager(root, torch.device("cpu"))

        self.assertTrue(
            MemoryManager.demote_layer(root[0], mm, layer_key="blocks.0.mlp.up")
        )
        self.assertEqual(root[0]._mm_layer_key, "blocks.0.mlp.up")
    def test_resident_candidate_layers_are_recorded_in_trace(self):
        from toolkit.memory_management import manager_modules
        from toolkit.memory_management.manager import MemoryManager

        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        manager_modules.set_offload_trace_enabled(True)
        try:
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None
            manager_modules._OFFLOAD_TRACE.version = 0
            MemoryManager.attach(
                model,
                torch.device("cpu"),
                _offload_module_ids=set(),
            )

            manager_modules.offload_step_begin(shape_key="resident-shape")
            x = torch.randn(2, 4, requires_grad=True)
            model(x).sum().backward()
            manager_modules.offload_step_end()

            self.assertEqual(
                manager_modules.offload_trace_schedule("resident-shape"),
                [
                    ("0", "forward", 0),
                    ("1", "forward", 0),
                    ("1", "backward", 1),
                    ("0", "backward", 1),
                ],
            )
        finally:
            MemoryManager.detach(model)
            manager_modules.set_offload_trace_enabled(False)
            manager_modules._OFFLOAD_TRACE.schedule_by_shape_key.clear()
            manager_modules._OFFLOAD_TRACE.frozen = None

class BounceFillGroupTests(unittest.TestCase):
    """Block-granular worker fill batching (fill_group_size)."""

    def test_default_and_env_fill_group_size(self):
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, num_workers=1, ram_floor_bytes=0,
        )
        try:
            self.assertEqual(pool.fill_group_size, 1)
        finally:
            pool.shutdown()

        with mock.patch.dict(os.environ, {"AI_TOOLKIT_BOUNCE_FILL_GROUP": "8"}):
            pool = bounce_pool.PinnedBouncePool(
                "cpu", budget_bytes=1 << 20, num_workers=1, ram_floor_bytes=0,
            )
            try:
                self.assertEqual(pool.fill_group_size, 8)
            finally:
                pool.shutdown()

    def test_fill_batches_counts_block_cycles(self):
        # The worker-fill counters must show batching: same fills, but group=8
        # publishes in far fewer batches (worker lock-cycles) than group=1.
        batches_by_group = {}
        for group in (1, 8):
            pool = bounce_pool.PinnedBouncePool(
                "cpu", budget_bytes=1 << 24, lookahead=8, num_workers=1,
                ram_floor_bytes=0, target_ready_bytes=1 << 24,
                fill_group_size=group,
            )
            try:
                keys = [f"layer{i}" for i in range(8)]
                modules = [torch.nn.Linear(16, 16) for _ in keys]
                for k, m in zip(keys, modules):
                    pool.register_source(k, m)
                pool.set_schedule(keys)
                self.assertTrue(_wait_until(
                    lambda: pool.stats()["fills"] >= 8, timeout=3.0
                ), f"group={group} did not fill all positions")
                stats = pool.stats()
                self.assertEqual(stats["fills"], 8, f"group={group} fills")
                batches_by_group[group] = stats["fill_batches"]
            finally:
                pool.shutdown()
        self.assertEqual(batches_by_group[1], 8, "group=1 should be one batch per fill")
        self.assertLess(
            batches_by_group[8], batches_by_group[1],
            f"group=8 should batch: {batches_by_group}",
        )

    def test_batched_fill_respects_budget(self):
        # fill_group_size must never exceed the byte budget: a group larger than
        # what fits still stops at the budget instead of over-allocating.
        lin_bytes = bounce_pool._spec_bytes(
            bounce_pool._layer_specs(torch.nn.Linear(16, 16).weight, None)
        )
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=lin_bytes * 3, lookahead=8, num_workers=1,
            ram_floor_bytes=0, target_ready_bytes=lin_bytes * 8,
            fill_group_size=8,
        )
        try:
            keys = [f"layer{i}" for i in range(8)]
            modules = [torch.nn.Linear(16, 16) for _ in keys]  # keep strong refs
            for k, m in zip(keys, modules):
                pool.register_source(k, m)
            pool.set_schedule(keys)
            _wait_until(lambda: pool.stats()["inflight_gib"] > 0, timeout=3.0)
            time.sleep(0.05)
            with pool._cv:
                self.assertLessEqual(pool._inflight_bytes, pool.budget_bytes)
        finally:
            pool.shutdown()


@unittest.skipUnless(torch.cuda.is_available(), "pin_memory requires CUDA")
class PinnedSourceFillSkipTests(unittest.TestCase):
    """A layer that is already pinned (e.g. under the pinned-weight auto-budget)
    must not be re-bounced: the consumer transfers straight from it, so a worker
    fill would just burn a copy that gets discarded. _claim_one_fill_locked must
    skip it (like the existing no-source/too-big skips) without ever creating a
    slot for that position."""

    def test_claim_skips_pinned_weight_without_creating_slot(self):
        pinned_module = torch.nn.Linear(8, 8, bias=False)
        pinned_module.weight.data = pinned_module.weight.data.pin_memory()
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=4, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            pool.register_source("pinned", pinned_module)
            pool.set_schedule(["pinned"])
            with pool._cv:
                status, job = pool._claim_one_fill_locked()
                self.assertEqual(status, "skip")
                self.assertIsNone(job)
                self.assertEqual(pool._slots, {})
                self.assertEqual(pool._fill_pos, 1)
        finally:
            pool.shutdown()

    def test_claim_still_fills_pageable_weight(self):
        pageable_module = torch.nn.Linear(8, 8, bias=False)
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=4, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            pool.register_source("pageable", pageable_module)
            pool.set_schedule(["pageable"])
            with pool._cv:
                status, job = pool._claim_one_fill_locked()
                self.assertEqual(status, "job")
                self.assertIsNotNone(job)
                self.assertIn(0, pool._slots)
        finally:
            pool.shutdown()

    def test_worker_never_copies_an_already_pinned_layer(self):
        """End-to-end: let the real worker thread run against a mixed pinned
        + pageable schedule and confirm no fill was ever recorded for the
        pinned position (not just that the synchronous claim skips it)."""
        pinned_module = torch.nn.Linear(8, 8, bias=False)
        pinned_module.weight.data = pinned_module.weight.data.pin_memory()
        pageable_module = torch.nn.Linear(8, 8, bias=False)
        pool = bounce_pool.PinnedBouncePool(
            "cpu", budget_bytes=1 << 20, lookahead=4, num_workers=1,
            ram_floor_bytes=0,
        )
        try:
            pool.register_source("pinned", pinned_module)
            pool.register_source("pageable", pageable_module)
            pool.set_schedule(["pinned", "pageable"])
            _wait_until(lambda: pool.stats()["fills"] >= 1, timeout=2.0)
            time.sleep(0.05)
            with pool._cv:
                self.assertNotIn(0, pool._slots, "pinned position must never get a slot")
                pageable_slot = pool._slots.get(1)
                self.assertIsNotNone(pageable_slot)
        finally:
            pool.shutdown()


if __name__ == "__main__":
    unittest.main()

import pytest

pytestmark = pytest.mark.process_isolated
