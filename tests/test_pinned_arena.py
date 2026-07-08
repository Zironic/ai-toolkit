import io
import unittest
from unittest import mock

import torch
import torch.nn as nn
from optimum.quanto import freeze

from toolkit.util.quantize import get_qtype, quantize
from toolkit.memory_management import pin_manager
from toolkit.memory_management.pinned_arena import (
    ArenaLayoutError,
    PinnedWeightArena,
)


def _linear(in_f=8, out_f=4, bias=True):
    layer = nn.Linear(in_f, out_f, bias=bias)
    layer.weight.requires_grad_(False)
    if bias:
        layer.bias.requires_grad_(False)
    return layer


class ArenaBuildTests(unittest.TestCase):
    def test_build_repoints_params_into_one_flat_per_block(self):
        a = _linear()
        b = _linear()
        arena = PinnedWeightArena()
        stats = arena.build({"blocks.0": [("lin_a", a), ("lin_b", b)]})

        self.assertEqual(stats.blocks, 1)
        flat_ptr = arena.block_pack("blocks.0").host_flat.untyped_storage().data_ptr()
        self.assertEqual(a.weight.untyped_storage().data_ptr(), flat_ptr)
        self.assertEqual(b.weight.untyped_storage().data_ptr(), flat_ptr)
        self.assertEqual(a.bias.untyped_storage().data_ptr(), flat_ptr)

    def test_build_tags_modules_for_membership(self):
        layer = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})
        self.assertEqual(arena.arena_block_of(layer), "blocks.0")
        self.assertTrue(arena.is_current(layer))

    def test_non_arena_module_reports_no_membership(self):
        member = _linear()
        stranger = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", member)]})
        self.assertIsNone(arena.arena_block_of(stranger))
        self.assertFalse(arena.is_current(stranger))

    def test_state_dict_round_trip_after_repoint(self):
        layer = _linear()
        expected = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})

        buffer = io.BytesIO()
        torch.save(layer.state_dict(), buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        for key, value in expected.items():
            self.assertTrue(torch.equal(value, loaded[key]), key)

    def test_quantized_wrapper_block_repoints_and_preserves_values(self):
        model = nn.Sequential(nn.Linear(8, 4, bias=False).to(torch.bfloat16))
        quantize(model, weights=get_qtype("qfloat8"))
        freeze(model)
        layer = model[0]
        layer.weight.requires_grad_(False)
        expected = layer.weight.data.dequantize().clone()

        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})

        flat_ptr = arena.block_pack("blocks.0").host_flat.untyped_storage().data_ptr()
        self.assertEqual(layer.weight.data._data.untyped_storage().data_ptr(), flat_ptr)
        torch.testing.assert_close(layer.weight.data.dequantize(), expected)

    def test_trainable_leaf_is_rejected(self):
        layer = _linear()
        layer.weight.requires_grad_(True)
        arena = PinnedWeightArena()
        with self.assertRaises(ArenaLayoutError):
            arena.build({"blocks.0": [("lin", layer)]})

    def test_entries_without_module_are_rejected(self):
        layer = _linear()
        arena = PinnedWeightArena()
        with self.assertRaises(ArenaLayoutError):
            arena.build({"blocks.0": [("lin", layer.weight, layer.bias)]})

    def test_pageable_fallback_when_pin_refuses(self):
        layer = _linear()
        arena = PinnedWeightArena()
        # Arena flats pin via pin_register (cudaHostRegister, exact DXGI
        # cost) -- refuse that to force the pageable path.
        with mock.patch(
            "toolkit.memory_management.ingraph_stream.pin_manager.pin_register",
            side_effect=lambda nbytes, kind, **kw: pin_manager.PinHandle(
                tensor=torch.empty(nbytes, dtype=torch.uint8), nbytes=nbytes,
                kind=kind, pinned=False, mechanism="register",
            ),
        ):
            stats = arena.build({"blocks.0": [("lin", layer)]})
        self.assertEqual(stats.pageable_blocks, 1)
        self.assertFalse(arena.block_pack("blocks.0").pinned)
        # Still repointed even though pageable.
        self.assertEqual(
            layer.weight.untyped_storage().data_ptr(),
            arena.block_pack("blocks.0").host_flat.untyped_storage().data_ptr(),
        )
        self.assertFalse(arena.has_all_blocks_pinned())


class ArenaRebuildInvalidateTests(unittest.TestCase):
    def test_rebuild_bumps_generation_and_releases_previous_pack(self):
        layer = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})
        first_gen = arena._generation["blocks.0"]
        old_pack = arena.block_pack("blocks.0")
        old_handle = old_pack.pin_handle

        with mock.patch.object(pin_manager, "release") as released:
            arena.build({"blocks.0": [("lin", layer)]})
        released.assert_called_once_with(old_handle)
        self.assertEqual(arena._generation["blocks.0"], first_gen + 1)
        self.assertTrue(arena.is_current(layer))

    def test_invalidate_block_makes_module_stale(self):
        layer = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})
        self.assertTrue(arena.is_current(layer))
        arena.invalidate_block("blocks.0")
        self.assertFalse(arena.is_current(layer))
        # Membership itself is unaffected -- only currency.
        self.assertEqual(arena.arena_block_of(layer), "blocks.0")


class ArenaRestoreViewTests(unittest.TestCase):
    def test_restore_view_after_simulated_promotion(self):
        layer = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})
        original_weight = layer.weight.detach().clone()
        original_bias = layer.bias.detach().clone()

        # Simulate promotion: replace params with standalone (non-arena)
        # tensors, as promote_to_resident would when moving the layer
        # off the arena view.
        layer.weight = nn.Parameter(layer.weight.detach().clone())
        layer.bias = nn.Parameter(layer.bias.detach().clone())
        flat_ptr = arena.block_pack("blocks.0").host_flat.untyped_storage().data_ptr()
        self.assertNotEqual(layer.weight.untyped_storage().data_ptr(), flat_ptr)

        arena.restore_view(layer, "weight")
        arena.restore_view(layer, "bias")

        self.assertEqual(layer.weight.untyped_storage().data_ptr(), flat_ptr)
        self.assertEqual(layer.bias.untyped_storage().data_ptr(), flat_ptr)
        torch.testing.assert_close(layer.weight.detach(), original_weight)
        torch.testing.assert_close(layer.bias.detach(), original_bias)
        self.assertTrue(arena.is_current(layer))

    def test_restore_view_rejects_non_arena_module(self):
        arena = PinnedWeightArena()
        stranger = _linear()
        with self.assertRaises(KeyError):
            arena.restore_view(stranger, "weight")


class ArenaReleaseTests(unittest.TestCase):
    def test_release_drops_all_blocks_and_releases_packs(self):
        layer = _linear()
        arena = PinnedWeightArena()
        arena.build({"blocks.0": [("lin", layer)]})
        pack = arena.block_pack("blocks.0")
        handle = pack.pin_handle
        with mock.patch.object(pin_manager, "release") as released:
            arena.release()
        released.assert_called_once_with(handle)
        self.assertIsNone(arena.block_pack("blocks.0"))
        self.assertIsNone(arena.arena_block_of(layer))


if __name__ == "__main__":
    unittest.main()
