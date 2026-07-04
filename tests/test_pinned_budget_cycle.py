import unittest
import torch

from toolkit.memory_management import MemoryManager


class PinnedBudgetCycleTests(unittest.TestCase):
    """The pinned-weight budget must be returned when a layer is promoted, so it
    does not leak upward across demote/promote cycles and starve later demotions.

    These tests exercise pinned-byte accounting, not host-memory policy, and must
    not depend on how much real RAM happens to be free on the machine running
    them."""

    def _manager(self, model):
        # cpu process device keeps promote/demote moves local; pinning still
        # exercises the byte accounting the same way.
        return MemoryManager(model, torch.device("cpu"), pinned_weight_gib=8.0)

    def test_promote_returns_budget_and_clears_marker(self):
        model = torch.nn.Sequential(torch.nn.Linear(32, 32, bias=False))
        lin = model[0]
        lin._mm_layer_key = "0"
        manager = self._manager(model)
        self.assertEqual(manager.pinned_weight_bytes, 0)

        self.assertTrue(MemoryManager.demote_layer(lin, manager, layer_key="0"))
        after_demote = manager.pinned_weight_bytes
        self.assertGreater(after_demote, 0, "demote should pin the weight")
        self.assertEqual(getattr(lin, "_mm_pinned_bytes", 0), after_demote)

        self.assertTrue(MemoryManager.promote_layer(lin))
        self.assertEqual(
            manager.pinned_weight_bytes, 0, "promote must return the pinned budget"
        )
        self.assertEqual(getattr(lin, "_mm_pinned_bytes", 0), 0)

    def test_repeated_cycles_do_not_leak(self):
        model = torch.nn.Sequential(torch.nn.Linear(48, 48, bias=False))
        lin = model[0]
        lin._mm_layer_key = "0"
        manager = self._manager(model)
        for _ in range(5):
            self.assertTrue(MemoryManager.demote_layer(lin, manager, layer_key="0"))
            self.assertGreater(manager.pinned_weight_bytes, 0)
            self.assertTrue(MemoryManager.promote_layer(lin))
            self.assertEqual(manager.pinned_weight_bytes, 0)


if __name__ == "__main__":
    unittest.main()
