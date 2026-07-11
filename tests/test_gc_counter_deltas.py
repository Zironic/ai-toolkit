import unittest

from toolkit.memory_management import MemoryManager


class GcCounterDeltaTests(unittest.TestCase):
    """Per-window deltas of the allocator GC counters.

    num_alloc_retries ticks once per OOM-retry reclaim (the gc_threshold
    proactive sweep does not tick it); num_device_alloc/num_device_free
    expose cudaMalloc/cudaFree churn. The helper turns the process-lifetime
    monotonic counters into per-window deltas for the perf log.
    """

    def test_first_window_has_no_prev(self):
        deltas = MemoryManager._gc_counter_deltas(
            None,
            {"num_alloc_retries": 2, "num_device_alloc": 40, "num_device_free": 13},
        )
        self.assertEqual(deltas["num_alloc_retries"], 2)
        self.assertEqual(deltas["num_device_alloc"], 40)
        self.assertEqual(deltas["num_device_free"], 13)

    def test_steady_state_is_zero(self):
        counters = {
            "num_alloc_retries": 5,
            "num_device_alloc": 100,
            "num_device_free": 30,
        }
        deltas = MemoryManager._gc_counter_deltas(dict(counters), dict(counters))
        self.assertEqual(
            deltas,
            {"num_alloc_retries": 0, "num_device_alloc": 0, "num_device_free": 0},
        )

    def test_window_delta(self):
        prev = {"num_alloc_retries": 1, "num_device_alloc": 50, "num_device_free": 10}
        curr = {"num_alloc_retries": 1, "num_device_alloc": 53, "num_device_free": 23}
        deltas = MemoryManager._gc_counter_deltas(prev, curr)
        self.assertEqual(deltas["num_alloc_retries"], 0)
        self.assertEqual(deltas["num_device_alloc"], 3)
        self.assertEqual(deltas["num_device_free"], 13)

    def test_external_reset_reports_current(self):
        # torch.cuda.reset_* wipes the counters; the current value is then the
        # best available per-window delta rather than a bogus negative.
        prev = {"num_alloc_retries": 7, "num_device_alloc": 90, "num_device_free": 40}
        curr = {"num_alloc_retries": 1, "num_device_alloc": 4, "num_device_free": 2}
        deltas = MemoryManager._gc_counter_deltas(prev, curr)
        self.assertEqual(deltas["num_alloc_retries"], 1)
        self.assertEqual(deltas["num_device_alloc"], 4)
        self.assertEqual(deltas["num_device_free"], 2)

    def test_missing_keys_default_to_zero(self):
        deltas = MemoryManager._gc_counter_deltas({}, {})
        self.assertEqual(
            deltas,
            {"num_alloc_retries": 0, "num_device_alloc": 0, "num_device_free": 0},
        )


if __name__ == "__main__":
    unittest.main()
