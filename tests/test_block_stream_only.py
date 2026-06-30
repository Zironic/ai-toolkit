import unittest

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


if __name__ == "__main__":
    unittest.main()
