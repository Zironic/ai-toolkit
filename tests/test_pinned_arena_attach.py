import unittest

import torch
import torch.nn as nn

from toolkit.memory_management import pin_manager
from toolkit.memory_management.manager import MemoryManager
from toolkit.memory_management.manager_modules import unpin_layer


class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, d, bias=True)
        self.b = nn.Linear(d, d, bias=True)
        for p in self.a.parameters():
            p.requires_grad_(False)
        for p in self.b.parameters():
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


@unittest.skipUnless(torch.cuda.is_available(), "pinned-arena attach needs CUDA")
class PinnedArenaAttachTests(unittest.TestCase):
    def setUp(self):
        self._ledger_before = pin_manager.pinned_bytes_by_kind()
        pin_manager._LEDGER.clear()

        def _restore():
            pin_manager._LEDGER.clear()
            pin_manager._LEDGER.update(self._ledger_before)

        self.addCleanup(_restore)

    def _offload_ids(self, model):
        return {id(m) for m in model.modules() if isinstance(m, nn.Linear)}

    def test_attach_with_arena_repoints_offloaded_weights_by_block(self):
        model = _Model(16, 2)
        device = torch.device("cuda:0")
        MemoryManager.attach(
            model, device, _offload_module_ids=self._offload_ids(model),
            use_pinned_arena=True,
        )
        try:
            arena = model._mm_weight_arena
            self.assertIsNotNone(arena)
            block0 = arena.arena_block_of(model.blocks[0].a)
            self.assertIsNotNone(block0)
            self.assertEqual(block0, arena.arena_block_of(model.blocks[0].b))
            flat_ptr = arena.block_pack(block0).host_flat.untyped_storage().data_ptr()
            self.assertEqual(
                model.blocks[0].a.weight.untyped_storage().data_ptr(), flat_ptr
            )
            self.assertEqual(
                model.blocks[0].b.weight.untyped_storage().data_ptr(), flat_ptr
            )
        finally:
            MemoryManager.detach(model)

    def test_detach_preserves_arena_and_avoids_unpin_repin_churn(self):
        model = _Model(16, 2)
        device = torch.device("cuda:0")
        MemoryManager.attach(
            model, device, _offload_module_ids=self._offload_ids(model),
            use_pinned_arena=True,
        )
        arena = model._mm_weight_arena
        block0 = arena.arena_block_of(model.blocks[0].a)
        flat_ptr_before = arena.block_pack(block0).host_flat.untyped_storage().data_ptr()
        weights_before = pin_manager.pinned_bytes_by_kind().get("weights", 0)

        MemoryManager.detach(model)

        # The arena is hung off the model, not the (destroyed) MemoryManager --
        # it must survive detach untouched: same flat storage, same ledger.
        self.assertIs(model._mm_weight_arena, arena)
        self.assertEqual(
            model.blocks[0].a.weight.untyped_storage().data_ptr(), flat_ptr_before
        )
        self.assertEqual(
            pin_manager.pinned_bytes_by_kind().get("weights", 0), weights_before
        )

        MemoryManager.attach(
            model, device, _offload_module_ids=self._offload_ids(model),
            use_pinned_arena=True,
        )
        try:
            # Re-attach must not have re-pinned/rebuilt: same arena object,
            # same flat storage, ledger unchanged (pure accounting, no churn).
            self.assertIs(model._mm_weight_arena, arena)
            self.assertEqual(
                model.blocks[0].a.weight.untyped_storage().data_ptr(), flat_ptr_before
            )
            self.assertEqual(
                pin_manager.pinned_bytes_by_kind().get("weights", 0), weights_before
            )
        finally:
            MemoryManager.detach(model)

    def test_unpin_layer_is_noop_for_arena_backed_child(self):
        model = _Model(16, 1)
        device = torch.device("cuda:0")
        MemoryManager.attach(
            model, device, _offload_module_ids=self._offload_ids(model),
            use_pinned_arena=True,
        )
        try:
            arena = model._mm_weight_arena
            child = model.blocks[0].a
            flat_ptr = arena.block_pack(arena.arena_block_of(child)).host_flat.untyped_storage().data_ptr()
            released = unpin_layer(child)
            self.assertEqual(released, 0)
            self.assertEqual(child.weight.untyped_storage().data_ptr(), flat_ptr)
        finally:
            MemoryManager.detach(model)


if __name__ == "__main__":
    unittest.main()
