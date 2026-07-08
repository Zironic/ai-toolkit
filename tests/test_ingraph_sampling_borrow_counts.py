"""Ticket 534ea49 Phase 2 Slice E: enable_ingraph_sampling exposes
borrowed-vs-owned pack counts for diagnostics (never a gate -- every
STREAMED pack must be pinned regardless of its source)."""

import unittest

import torch

from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from toolkit.memory_management.pinned_arena import PinnedWeightArena


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
        try:
            model.enable_ingraph_sampling()
            self.assertEqual(model._ingraph_sampling_borrowed_count, 0)
            self.assertEqual(model._ingraph_sampling_owned_count, 2)
        finally:
            model.disable_ingraph_sampling()

    def test_arena_backed_block_is_borrowed_others_stay_owned(self):
        model = self._model(layers=2)
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

    def test_disable_resets_counters(self):
        model = self._model(layers=1)
        model.enable_ingraph_sampling()
        model.disable_ingraph_sampling()
        self.assertEqual(model._ingraph_sampling_borrowed_count, 0)
        self.assertEqual(model._ingraph_sampling_owned_count, 0)


if __name__ == "__main__":
    unittest.main()
