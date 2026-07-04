import unittest

import torch

from toolkit.memory_management.manager import MemoryManager
from toolkit.memory_management import manager_modules


class FakeManager:
    def __init__(self, budget=1024):
        self.pinned_weight_budget_bytes = budget
        self.pinned_weight_bytes = 0
        self.process_device = torch.device("cpu")


class TransactionalPinPolicyTests(unittest.TestCase):
    def _patch(self, name, value):
        old = getattr(manager_modules, name)
        setattr(manager_modules, name, value)
        self.addCleanup(lambda: setattr(manager_modules, name, old))
        return old

    def test_dxgi_headroom_caps_pin_attempt_budget(self):
        budgets = []

        def fake_headroom(_device=None):
            return 16

        def fake_ensure(tensor, budget):
            budgets.append(budget)
            return tensor, 8 if budget else 0

        self._patch("_dxgi_signed_headroom_bytes", fake_headroom)
        self._patch("_ensure_cpu_pinned", fake_ensure)
        lin = torch.nn.Linear(4, 4, bias=False)
        manager = FakeManager(budget=1024)

        manager_modules._move_params_to_cpu_and_pin(lin, manager)

        self.assertEqual(budgets, [16])
        self.assertEqual(manager.pinned_weight_bytes, 8)
        self.assertEqual(getattr(lin, "_mm_pinned_bytes", 0), 8)

    def test_overshoot_rolls_back_layer_pin(self):
        calls = []
        headrooms = iter([64, -1])

        def fake_headroom(_device=None):
            return next(headrooms)

        def fake_ensure(tensor, budget):
            self.assertEqual(budget, 64)
            return tensor, 32

        def fake_unpin(module, manager):
            calls.append((module, manager))
            manager.pinned_weight_bytes = 0
            module._mm_pinned_bytes = 0
            return 32

        self._patch("_dxgi_signed_headroom_bytes", fake_headroom)
        self._patch("_ensure_cpu_pinned", fake_ensure)
        self._patch("_unpin_module_weights", fake_unpin)
        lin = torch.nn.Linear(4, 4, bias=False)
        manager = FakeManager(budget=1024)

        manager_modules._move_params_to_cpu_and_pin(lin, manager)

        self.assertEqual(len(calls), 1)
        self.assertEqual(manager.pinned_weight_bytes, 0)
        self.assertEqual(getattr(lin, "_mm_pinned_bytes", 0), 0)

    def test_no_control_eligible_sensor_keeps_legacy_pin_budget(self):
        budgets = []

        def fake_headroom(_device=None):
            return None

        def fake_ensure(tensor, budget):
            budgets.append(budget)
            return tensor, 4

        self._patch("_dxgi_signed_headroom_bytes", fake_headroom)
        self._patch("_ensure_cpu_pinned", fake_ensure)
        lin = torch.nn.Linear(4, 4, bias=False)
        manager = FakeManager(budget=128)

        manager_modules._move_params_to_cpu_and_pin(lin, manager)

        self.assertEqual(budgets, [128])
        self.assertEqual(manager.pinned_weight_bytes, 4)


class SamplingDxgiSettleTests(unittest.TestCase):
    def test_settle_status_unavailable(self):
        self.assertEqual(MemoryManager._dxgi_sampling_settle_status(None), "unavailable")

    def test_settle_status_pressure_until_raw_headroom_clears_margin(self):
        self.assertEqual(
            MemoryManager._dxgi_sampling_settle_status(
                {"raw_headroom_bytes": 10, "margin_bytes": 20}
            ),
            "pressure",
        )
        self.assertEqual(
            MemoryManager._dxgi_sampling_settle_status(
                {"raw_headroom_bytes": 20, "margin_bytes": 20}
            ),
            "settled",
        )

    def test_wait_for_sampling_dxgi_settle_times_out_bounded(self):
        calls = []
        old = MemoryManager._dxgi_shared_budget_snapshot_bytes
        try:
            MemoryManager._dxgi_shared_budget_snapshot_bytes = staticmethod(
                lambda _device: calls.append(1) or {"raw_headroom_bytes": 1, "margin_bytes": 2}
            )
            result = MemoryManager._wait_for_sampling_dxgi_settle(
                torch.device("cpu"), timeout_s=0.0, poll_s=0.0
            )
        finally:
            MemoryManager._dxgi_shared_budget_snapshot_bytes = old
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
