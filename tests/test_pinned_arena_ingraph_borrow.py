"""Ticket 534ea49 Slice 4 / 763bb75: ingraph packs must be able to borrow an
already-pinned arena flat instead of allocating (and separately pinning) a
fresh one over the same weights. Only actual module identity and live
storage are trusted -- never the caller's naming convention or a cached
layout -- so a stale/rebuilt block or a genuinely different module set fails
closed to None (caller falls back to an owned pack) rather than ever
packing mixed storage."""

import unittest
from unittest import mock

import torch
import torch.nn as nn
from optimum.quanto import freeze

from toolkit.util.quantize import get_qtype, quantize
from toolkit.memory_management import pin_manager
from toolkit.memory_management.ingraph_stream import (
    block_linear_views,
    release_pack,
)
from toolkit.memory_management.pinned_arena import PinnedWeightArena


def _linear(in_f=8, out_f=4, bias=True):
    layer = nn.Linear(in_f, out_f, bias=bias)
    layer.weight.requires_grad_(False)
    if bias:
        layer.bias.requires_grad_(False)
    return layer


class TryBorrowPackTests(unittest.TestCase):
    def test_borrow_succeeds_with_different_naming_than_arena(self):
        a = _linear()
        b = _linear()
        arena = PinnedWeightArena()
        # Arena tags these with its own (full-path-style) names...
        arena.build({"blocks.0": [("blocks.0.attn.wq", a), ("blocks.0.attn.wk", b)]})

        # ...but ingraph asks for the SAME modules under its own short names.
        pack = arena.try_borrow_pack("blocks.0", [("attn.wq", a), ("attn.wk", b)])
        self.assertIsNotNone(pack)
        self.assertFalse(pack.owns_flat)
        self.assertTrue(pack.borrowed_from_arena)
        self.assertIsNone(pack.pin_handle)
        self.assertEqual(
            pack.host_flat.untyped_storage().data_ptr(),
            arena.block_pack("blocks.0").host_flat.untyped_storage().data_ptr(),
        )
        views = block_linear_views(pack.host_flat, pack)
        self.assertTrue(torch.equal(views["attn.wq"].weight, a.weight))
        self.assertTrue(torch.equal(views["attn.wq"].bias, a.bias))
        self.assertTrue(torch.equal(views["attn.wk"].weight, b.weight))

    def test_borrow_returns_none_for_unknown_block(self):
        arena = PinnedWeightArena()
        a = _linear()
        arena.build({"blocks.0": [("lin", a)]})
        self.assertIsNone(arena.try_borrow_pack("blocks.99", [("lin", a)]))

    def test_borrow_returns_none_for_non_arena_module(self):
        arena = PinnedWeightArena()
        member = _linear()
        stranger = _linear()
        arena.build({"blocks.0": [("lin", member)]})
        self.assertIsNone(arena.try_borrow_pack("blocks.0", [("lin", stranger)]))

    def test_borrow_returns_none_after_block_invalidated(self):
        a = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", a)]})
        arena.invalidate_block("blocks.0")
        self.assertIsNone(arena.try_borrow_pack("blocks.0", [("lin", a)]))

    def test_borrow_rejects_entries_without_module(self):
        a = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", a)]})
        self.assertIsNone(
            arena.try_borrow_pack("blocks.0", [("lin", a.weight, a.bias)])
        )

    def test_quantized_block_borrows_correctly(self):
        model = nn.Sequential(nn.Linear(8, 4, bias=False).to(torch.bfloat16))
        quantize(model, weights=get_qtype("qfloat8"))
        freeze(model)
        layer = model[0]
        layer.weight.requires_grad_(False)
        expected = layer.weight.data.dequantize().clone()

        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})
        pack = arena.try_borrow_pack("blocks.0", [("attn.proj", layer)])
        self.assertIsNotNone(pack)
        self.assertEqual(pack.linears[0].kind, "fp8_rowwise")

        views = block_linear_views(pack.host_flat, pack)
        torch.testing.assert_close(views["attn.proj"].materialized_weight(), expected.to(torch.bfloat16))

    def test_release_pack_on_borrowed_pack_never_touches_pin_manager(self):
        a = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", a)]})
        pack = arena.try_borrow_pack("blocks.0", [("attn.proj", a)])
        with mock.patch.object(pin_manager, "release") as released:
            release_pack(pack)
        released.assert_not_called()


if __name__ == "__main__":
    unittest.main()
