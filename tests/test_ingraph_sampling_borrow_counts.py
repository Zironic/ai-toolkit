"""Ticket 534ea49 Phase 2 Slice E: enable_ingraph_sampling exposes
borrowed-vs-owned pack counts for diagnostics (never a gate -- every
STREAMED pack must be pinned regardless of its source).

Per-leaf planning (the sampler-side port of 07563ad): only leaves the
memory manager streams are packed, so against a per-Linear arena the
borrow asks for exactly the leaves the arena holds. Leaves are marked
streamed here the same way the manager does it (`_layer_memory_manager`),
read before the strip removes it.
"""

import unittest

import torch

from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from toolkit.memory_management.pinned_arena import PinnedWeightArena


def _mark_streamed(entries):
    for _, module in entries:
        module._layer_memory_manager = object()


class IngraphSamplingBorrowCountTests(unittest.TestCase):
    def _model(self, layers=2):
        torch.manual_seed(123)
        model = SingleStreamDiT(
            SingleMMDiTConfig(
                features=32, tdim=16, txtdim=32, heads=4, multiplier=1,
                layers=layers, patch=1, channels=4, txtheads=4, txtkvheads=4,
            )
        ).eval()
        model.requires_grad_(False)
        return model

    def test_no_arena_means_every_pack_is_owned(self):
        model = self._model(layers=2)
        for block in model.blocks:
            _mark_streamed(model._block_linear_entries(block))
        try:
            model.enable_ingraph_sampling()
            self.assertEqual(model._ingraph_sampling_borrowed_count, 0)
            self.assertEqual(model._ingraph_sampling_owned_count, 2)
        finally:
            model.disable_ingraph_sampling()

    def test_arena_backed_block_is_borrowed_others_stay_owned(self):
        model = self._model(layers=2)
        for block in model.blocks:
            _mark_streamed(model._block_linear_entries(block))
        arena = PinnedWeightArena()
        arena.build({"blocks.0": list(model._block_linear_entries(model.blocks[0]))})
        model._mm_weight_arena = arena
        try:
            model.enable_ingraph_sampling()
            self.assertEqual(model._ingraph_sampling_borrowed_count, 1)
            self.assertEqual(model._ingraph_sampling_owned_count, 1)
            self.assertTrue(model._ingraph_sampling_packs[0].borrowed_from_arena)
            self.assertFalse(model._ingraph_sampling_packs[1].borrowed_from_arena)
        finally:
            model.disable_ingraph_sampling()
            arena.release()

    def test_partially_resident_block_borrows_streamed_subset(self):
        # The regression: the planner leaves some of a block's linears
        # resident, the arena holds ONLY the offloaded ones. Whole-block
        # packing asked the arena for all 8 leaves and got `borrow refused:
        # stale_modules`, fell back to owned packs, and blew the pin budget.
        model = self._model(layers=2)
        entries = list(model._block_linear_entries(model.blocks[0]))
        streamed = entries[:3]
        _mark_streamed(streamed)
        arena = PinnedWeightArena()
        arena.build({"blocks.0": list(streamed)})
        model._mm_weight_arena = arena
        try:
            model.enable_ingraph_sampling()
            self.assertEqual(model._ingraph_sampling_borrowed_count, 1)
            self.assertEqual(model._ingraph_sampling_owned_count, 0)
            plan = model._ingraph_sampling_plans[0]
            self.assertTrue(plan.streams)
            self.assertTrue(plan.borrowed_from_arena)
            self.assertEqual(
                sum(1 for from_pack, _ in plan.sources if from_pack), 3
            )
            self.assertEqual(len(plan.resident_args), len(entries) - 3)
            # Block 1 is fully resident: a plan with no pack, no fetch.
            self.assertFalse(model._ingraph_sampling_plans[1].streams)
            self.assertNotIn(1, model._ingraph_sampling_packs)
        finally:
            model.disable_ingraph_sampling()
            arena.release()

    def test_fully_resident_plans_match_eager_trunk(self):
        # No leaf streams: every plan reads straight off the Parameters and
        # the in-graph dispatch must be numerically identical to the plain
        # eager block calls.
        model = self._model(layers=2)
        x = torch.randn(1, 5, 32)
        vec = torch.randn(1, 192)
        pos = torch.zeros(1, 5, 3)
        freqs = model.posemb(pos)
        mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)
        with torch.no_grad():
            baseline = model._blocks_trunk(x, vec, freqs, mask)
        try:
            model.enable_ingraph_sampling()
            self.assertEqual(model._ingraph_sampling_resident_blocks, 2)
            self.assertEqual(model._ingraph_sampling_packs, {})
            with torch.no_grad():
                got = model._blocks_trunk(x, vec, freqs, mask)
        finally:
            model.disable_ingraph_sampling()
        torch.testing.assert_close(got, baseline, rtol=0, atol=0)

    def test_disable_resets_counters(self):
        model = self._model(layers=1)
        _mark_streamed(model._block_linear_entries(model.blocks[0]))
        model.enable_ingraph_sampling()
        model.disable_ingraph_sampling()
        self.assertEqual(model._ingraph_sampling_borrowed_count, 0)
        self.assertEqual(model._ingraph_sampling_owned_count, 0)
        self.assertEqual(model._ingraph_sampling_plans, {})


if __name__ == "__main__":
    unittest.main()
