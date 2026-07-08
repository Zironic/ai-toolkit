"""Ticket 534ea49 Phase 2 Slice D: explicit arena teardown. Params must be
detached onto standalone storage BEFORE the arena releases its packs -- never
called from detach()/sampling boundaries (the arena persists across those by
design), only for genuine model unload."""

import io
import unittest

import torch
import torch.nn as nn
from optimum.quanto import freeze

from toolkit.util.quantize import get_qtype, quantize
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


@unittest.skipUnless(torch.cuda.is_available(), "arena teardown needs CUDA")
class ArenaTeardownTests(unittest.TestCase):
    def setUp(self):
        self._ledger_before = pin_manager.pinned_bytes_by_kind()
        pin_manager._LEDGER.clear()

        def _restore():
            pin_manager._LEDGER.clear()
            pin_manager._LEDGER.update(self._ledger_before)

        self.addCleanup(_restore)

    def _offload_ids(self, model):
        return {id(m) for m in model.modules() if isinstance(m, nn.Linear)}

    def test_destroy_detaches_params_and_releases_ledger(self):
        device = torch.device("cuda:0")
        model = _Model(16, 2)
        MemoryManager.attach(
            model, device, _offload_module_ids=self._offload_ids(model),
            use_pinned_arena=True,
        )
        arena = model._mm_weight_arena
        flat_ptr = arena.block_pack(
            arena.arena_block_of(model.blocks[0].a)
        ).host_flat.untyped_storage().data_ptr()
        expected = model.blocks[0].a.weight.detach().clone()

        MemoryManager._destroy_pinned_arena(model)

        # Params are standalone now -- no longer sharing the (now-released)
        # arena flat's storage.
        self.assertNotEqual(
            model.blocks[0].a.weight.untyped_storage().data_ptr(), flat_ptr
        )
        self.assertTrue(torch.equal(model.blocks[0].a.weight.detach().cpu(), expected.cpu()))
        self.assertIsNone(getattr(model, "_mm_weight_arena", None))
        self.assertIsNone(getattr(model.blocks[0].a, "_mm_arena_block", None))
        self.assertEqual(pin_manager.pinned_bytes_by_kind().get("weights", 0), 0)

        MemoryManager.detach(model)

    def test_destroy_is_noop_without_an_arena(self):
        model = _Model(16, 1)
        # No attach() at all -- must not raise.
        MemoryManager._destroy_pinned_arena(model)

    def test_destroy_preserves_requires_grad_and_state_dict(self):
        device = torch.device("cuda:0")
        model = _Model(16, 1)
        MemoryManager.attach(
            model, device, _offload_module_ids=self._offload_ids(model),
            use_pinned_arena=True,
        )
        before = {k: v.detach().clone() for k, v in model.blocks[0].state_dict().items()}
        MemoryManager._destroy_pinned_arena(model)

        buffer = io.BytesIO()
        torch.save(model.blocks[0].state_dict(), buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        for key, value in before.items():
            self.assertTrue(torch.equal(value.cpu(), loaded[key].cpu()), key)
        self.assertFalse(model.blocks[0].a.weight.requires_grad)

        MemoryManager.detach(model)

    def test_destroy_handles_quantized_wrapper_blocks(self):
        device = torch.device("cuda:0")

        class _QModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList(
                    [nn.Linear(8, 4, bias=False).to(torch.bfloat16)]
                )

            def forward(self, x):
                return self.blocks[0](x)

        model = _QModel()
        quantize(model, weights=get_qtype("qfloat8"))
        freeze(model)
        model.blocks[0].weight.requires_grad_(False)
        expected = model.blocks[0].weight.data.dequantize().clone()

        MemoryManager.attach(
            model, device, _offload_module_ids={id(model.blocks[0])},
            use_pinned_arena=True,
        )
        arena = model._mm_weight_arena
        flat_ptr = arena.block_pack(
            arena.arena_block_of(model.blocks[0])
        ).host_flat.untyped_storage().data_ptr()

        MemoryManager._destroy_pinned_arena(model)

        self.assertNotEqual(
            model.blocks[0].weight.data._data.untyped_storage().data_ptr(), flat_ptr
        )
        torch.testing.assert_close(
            model.blocks[0].weight.data.dequantize().cpu(), expected.cpu()
        )

        MemoryManager.detach(model)


if __name__ == "__main__":
    unittest.main()
