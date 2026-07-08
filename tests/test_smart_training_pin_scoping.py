"""Ticket 534ea49 Phase 2 Slice A, integration level: attach_smart_training's
auto pin-budget must scope to the layers actually selected for STREAMING, not
the whole model, when a smart-training plan keeps some blocks resident
(pinned_resident_keys). Before this fix, auto-pin sized from
plan["model_bytes"] regardless of how much of the model was actually
streamed."""

import unittest

import torch
import torch.nn as nn

from toolkit.memory_management.manager import MemoryManager


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, d, bias=True)
        self.b = nn.Linear(d, d, bias=True)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.b(self.a(x))


class _Model(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(d) for _ in range(n)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


@unittest.skipUnless(torch.cuda.is_available(), "attach_smart_training needs CUDA")
class SmartTrainingPinScopingTests(unittest.TestCase):
    def test_pinned_resident_block_is_excluded_from_pin_budget(self):
        device = torch.device("cuda:0")
        model = _Model(64, 2)
        try:
            # A tiny model fits fully resident with the default working
            # reserve (nothing to prove); force a small usable surplus with a
            # large manual working_reserve so block 1 stays genuinely
            # OFFLOADED instead of growing resident too.
            plan = MemoryManager.attach_smart_training(
                model, device,
                working_reserve_gib=10.5,
                pinned_resident_keys={"blocks.0"},
            )
            streamed_ids = plan["offload_ids"]
            self.assertNotIn(id(model.blocks[0].a), streamed_ids)
            self.assertIn(id(model.blocks[1].a), streamed_ids)

            expected_scoped = MemoryManager._desired_pin_bytes_for_offload_ids(
                model, streamed_ids, None
            )
            whole_model_bytes = int(MemoryManager._module_bytes(model) * 1.03)
            # The resident block's bytes must actually matter here, or this
            # test can't distinguish the fix from the bug.
            self.assertLess(expected_scoped, whole_model_bytes)

            # No WDDM/RAM cap should bind for a model this small -- the
            # resolved budget must equal the block-0-EXCLUDED figure, not
            # the whole-model one.
            got = model._memory_manager.pinned_weight_budget_bytes
            self.assertEqual(got, expected_scoped)
            self.assertLess(got, whole_model_bytes)
        finally:
            MemoryManager.detach(model)


if __name__ == "__main__":
    unittest.main()
