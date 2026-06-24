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


if __name__ == "__main__":
    unittest.main()
