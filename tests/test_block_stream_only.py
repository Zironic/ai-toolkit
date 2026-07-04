import unittest
from unittest import mock

import torch

from toolkit.memory_management import MemoryManager


class BlockParentDetectionTests(unittest.TestCase):
    """Pure-CPU tests for the block-vs-singleton classification that drives
    ``block_stream_only`` streaming. These do not need CUDA — they exercise the
    helpers ``smart_training_plan`` uses to decide which layers may stream."""

    def test_block_parent_of_identifies_indexed_groups(self):
        self.assertEqual(MemoryManager._block_parent_of("blocks.7"), "blocks")
        self.assertEqual(
            MemoryManager._block_parent_of("transformer.layers.12"),
            "transformer.layers",
        )
        # Sequential-indexed one-off (final_layer.adaLN_modulation.1) still
        # parses to a parent; whether it is a *streaming block* is decided by
        # sibling count in _streaming_block_parents, not here.
        self.assertEqual(
            MemoryManager._block_parent_of("final_layer.adaLN_modulation.1"),
            "final_layer.adaLN_modulation",
        )

    def test_block_parent_of_returns_none_for_non_indexed(self):
        self.assertIsNone(MemoryManager._block_parent_of("x_embedder.proj"))
        self.assertIsNone(MemoryManager._block_parent_of("final_layer"))

    def test_repeated_modulelist_is_a_streaming_block(self):
        group_keys = [f"blocks.{i}" for i in range(8)]
        parents = MemoryManager._streaming_block_parents(group_keys)
        self.assertEqual(parents, {"blocks"})

    def test_lone_sequential_index_is_not_a_streaming_block(self):
        # Embedders, final projection, and a single Sequential Linear must NOT be
        # treated as streaming blocks — in block_stream_only they stay resident.
        group_keys = [
            "x_embedder.proj",
            "context_embedder",
            "final_layer.adaLN_modulation.1",
        ] + [f"blocks.{i}" for i in range(4)]
        parents = MemoryManager._streaming_block_parents(group_keys)
        self.assertEqual(parents, {"blocks"})
        # The repeated blocks are streamable...
        self.assertIn(MemoryManager._block_parent_of("blocks.2"), parents)
        # ...while the one-off layers are not.
        self.assertNotIn(
            MemoryManager._block_parent_of("final_layer.adaLN_modulation.1"), parents
        )
        self.assertNotIn(MemoryManager._block_parent_of("x_embedder.proj"), parents)

    def test_multiple_independent_block_lists(self):
        group_keys = (
            [f"double_blocks.{i}" for i in range(6)]
            + [f"single_blocks.{i}" for i in range(3)]
            + ["img_in", "txt_in"]
        )
        parents = MemoryManager._streaming_block_parents(group_keys)
        self.assertEqual(parents, {"double_blocks", "single_blocks"})


class InterleavePriorityTests(unittest.TestCase):
    """Pure-CPU tests for the van der Corput / bit-reversal tie-break that
    spreads resident/pinned selection evenly across equal-sized candidates
    (repeated transformer blocks make most offloaded layers byte-identical, so
    the byte-size sort in ``smart_training_plan`` and the pin-budget order in
    ``attach`` tie constantly; this key decides how those ties are broken)."""

    def test_matches_bit_reversal_permutation(self):
        # Sorting indices 0..7 by priority must reproduce the classic
        # bit-reversal permutation, not the original 0..7 order.
        order = sorted(
            range(8), key=lambda i: MemoryManager._interleave_priority(i, 8)
        )
        self.assertEqual(order, [0, 4, 2, 6, 1, 5, 3, 7])

    def test_any_prefix_is_evenly_spread(self):
        # Taking the first K by priority (K unknown ahead of time, since it
        # depends on the remaining byte budget) must spread across the full
        # range, not cluster at one end.
        order = sorted(
            range(16), key=lambda i: MemoryManager._interleave_priority(i, 16)
        )
        self.assertEqual(set(order[:4]), {0, 4, 8, 12})
        self.assertEqual(set(order[:8]), {0, 2, 4, 6, 8, 10, 12, 14})

    def test_count_le_one_is_degenerate(self):
        self.assertEqual(MemoryManager._interleave_priority(0, 0), 0.0)
        self.assertEqual(MemoryManager._interleave_priority(0, 1), 0.0)

    def test_non_power_of_two_count_stays_in_unit_range(self):
        for i in range(24):
            priority = MemoryManager._interleave_priority(i, 24)
            self.assertGreaterEqual(priority, 0.0)
            self.assertLess(priority, 1.0)


class BounceBudgetDefaultTests(unittest.TestCase):
    def _model(self):
        return torch.nn.ModuleDict({
            "blocks": torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(8, 8),
                    torch.nn.Linear(8, 8),
                )
                for _ in range(3)
            ]),
            "head": torch.nn.Linear(8, 8),
        })

    def test_block_mode_defaults_to_small_block_window_not_history_pool(self):
        model = self._model()
        offload_ids = {id(child) for child in model.modules() if isinstance(child, torch.nn.Linear)}
        with mock.patch.dict(
            "os.environ",
            {
                "AI_TOOLKIT_BOUNCE_BLOCK_MIN_GIB": "0.75",
                "AI_TOOLKIT_BOUNCE_BLOCK_MAX_GIB": "1.50",
                "AI_TOOLKIT_BOUNCE_TARGET_FRACTION": "0.60",
            },
            clear=False,
        ):
            budget, target, mode = MemoryManager._training_bounce_pool_budget_defaults(
                model,
                offload_ids,
                block_stream_only=True,
                history={"budget_gib": 5.0, "target_ready_gib": 3.5},
            )
        self.assertEqual(mode, "block")
        self.assertAlmostEqual(budget, 0.75, places=6)
        self.assertAlmostEqual(target, 0.45, places=6)

    def test_layer_mode_defaults_to_modest_layer_window_not_history_pool(self):
        model = self._model()
        offload_ids = {id(child) for child in model.modules() if isinstance(child, torch.nn.Linear)}
        with mock.patch.dict(
            "os.environ",
            {
                "AI_TOOLKIT_BOUNCE_LAYER_MIN_GIB": "1.00",
                "AI_TOOLKIT_BOUNCE_LAYER_MAX_GIB": "2.00",
                "AI_TOOLKIT_BOUNCE_TARGET_FRACTION": "0.60",
            },
            clear=False,
        ):
            budget, target, mode = MemoryManager._training_bounce_pool_budget_defaults(
                model,
                offload_ids,
                block_stream_only=False,
                history={"budget_gib": 5.0, "target_ready_gib": 3.5},
            )
        self.assertEqual(mode, "layer")
        self.assertAlmostEqual(budget, 1.0, places=6)
        self.assertAlmostEqual(target, 0.6, places=6)


if __name__ == "__main__":
    unittest.main()
