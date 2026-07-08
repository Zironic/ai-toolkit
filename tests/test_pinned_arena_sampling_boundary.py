"""Ticket 534ea49, Slice 3: sampling boundaries (inference_resident) must not
unpin/re-pin arena-backed weights. Before this slice, cls.detach() (called on
sampling entry) unpinned the whole ~9.9 GiB weight set, and _restore_offload's
re-attach (sampling exit) re-pinned it -- seconds of page-lock kernel work per
boundary. With use_pinned_arena threaded into inference_resident's internal
attach() calls, the arena survives the whole round trip untouched."""

import unittest

import torch
import torch.nn as nn

from toolkit.memory_management import pin_manager
from toolkit.memory_management.manager import MemoryManager


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


@unittest.skipUnless(torch.cuda.is_available(), "sampling boundary needs CUDA")
class PinnedArenaSamplingBoundaryTests(unittest.TestCase):
    def setUp(self):
        self._ledger_before = pin_manager.pinned_bytes_by_kind()
        pin_manager._LEDGER.clear()

        def _restore():
            pin_manager._LEDGER.clear()
            pin_manager._LEDGER.update(self._ledger_before)

        self.addCleanup(_restore)
        torch.cuda.empty_cache()

    def _offload_ids(self, model):
        return {id(m) for m in model.modules() if isinstance(m, nn.Linear)}

    def test_inference_resident_round_trip_leaves_arena_and_ledger_untouched(self):
        device = torch.device("cuda:0")
        model = _Model(16, 2)
        MemoryManager.attach(
            model, device, _offload_module_ids=self._offload_ids(model),
            use_pinned_arena=True,
        )
        try:
            arena = model._mm_weight_arena
            self.assertIsNotNone(arena)
            block0 = arena.arena_block_of(model.blocks[0].a)
            flat_ptr_before = arena.block_pack(block0).host_flat.untyped_storage().data_ptr()
            weights_before = pin_manager.pinned_bytes_by_kind().get("weights", 0)

            with MemoryManager.inference_resident(model, device):
                x = torch.randn(2, 16, device=device)
                model(x)

            # Same arena object, same flat storage, ledger unchanged -- the
            # sampling round trip must not have unpinned/re-pinned anything.
            self.assertIs(model._mm_weight_arena, arena)
            self.assertEqual(
                model.blocks[0].a.weight.untyped_storage().data_ptr(), flat_ptr_before
            )
            self.assertEqual(
                pin_manager.pinned_bytes_by_kind().get("weights", 0), weights_before
            )
            # Training layout is restored: the layer is streamed again.
            self.assertTrue(hasattr(model.blocks[0].a, "_layer_memory_manager"))
        finally:
            MemoryManager.detach(model)


if __name__ == "__main__":
    unittest.main()
