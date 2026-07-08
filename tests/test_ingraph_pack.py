import io
import unittest
from unittest import mock

import pytest
import torch
import torch.nn as nn
from optimum.quanto import freeze

from toolkit.util.quantize import get_qtype, quantize
from toolkit.memory_management import pin_manager
from toolkit.memory_management.ingraph_stream import (
    block_linear_views,
    pack_block_host,
    release_pack,
)


class InGraphPackTests(unittest.TestCase):
    def _quanto_linear(self):
        model = nn.Sequential(torch.nn.Linear(8, 4, bias=False).to(torch.bfloat16))
        quantize(model, weights=get_qtype("qfloat8"))
        freeze(model)
        return model[0]

    def test_pack_repoints_plain_linear_weights_to_flat_storage(self):
        layer = torch.nn.Linear(8, 4, bias=True)
        before = {k: v.detach().clone() for k, v in layer.state_dict().items()}

        pack = pack_block_host("blocks.0", [("proj", layer)], repoint=True, pin=False)

        self.assertGreater(pack.required_pin_bytes, 0)
        self.assertEqual(1, len(pack.linears))
        flat_storage = pack.host_flat.untyped_storage().data_ptr()
        self.assertEqual(flat_storage, layer.weight.untyped_storage().data_ptr())
        self.assertEqual(flat_storage, layer.bias.untyped_storage().data_ptr())

        after = layer.state_dict()
        self.assertTrue(torch.equal(before["weight"], after["weight"]))
        self.assertTrue(torch.equal(before["bias"], after["bias"]))

    def test_block_linear_views_round_trip(self):
        layer = torch.nn.Linear(8, 4, bias=True)
        pack = pack_block_host("blocks.0", [("proj", layer)], repoint=False, pin=False)
        views = block_linear_views(pack.host_flat, pack)
        weight, bias = views["proj"]
        self.assertTrue(torch.equal(weight, layer.weight))
        self.assertTrue(torch.equal(bias, layer.bias))

    def test_state_dict_save_after_repoint(self):
        layer = torch.nn.Linear(8, 4, bias=True)
        expected = {k: v.detach().clone() for k, v in layer.state_dict().items()}
        pack_block_host("blocks.0", [("proj", layer)], repoint=True, pin=False)

        buffer = io.BytesIO()
        torch.save(layer.state_dict(), buffer)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)

        self.assertTrue(torch.equal(expected["weight"], loaded["weight"]))
        self.assertTrue(torch.equal(expected["bias"], loaded["bias"]))

    def test_quantized_weight_pack_records_qdata_and_scale(self):
        layer = self._quanto_linear()
        expected = layer.weight.data.dequantize().clone()

        pack = pack_block_host("blocks.0", [("proj", layer)], repoint=False, pin=False)
        self.assertEqual(pack.linears[0].kind, "fp8_rowwise")
        self.assertEqual(pack.linears[0].weight.role, "qdata")
        self.assertEqual(pack.linears[0].weight_scale.role, "scale")

        weight, bias = block_linear_views(pack.host_flat, pack)["proj"]
        self.assertIsNone(bias)
        torch.testing.assert_close(weight, expected)

    def test_streamed_linear_uses_native_fp8_path_when_qualified(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("native fp8 scaled_mm requires CUDA SM89+")
        layer = nn.Linear(16, 16, bias=True).to(device="cuda", dtype=torch.bfloat16)
        model = nn.Sequential(layer)
        quantize(model, weights=get_qtype("qfloat8"))
        freeze(model)
        layer = model[0]
        pack = pack_block_host("blocks.0", [("proj", layer)], repoint=False, pin=True)
        view = block_linear_views(pack.host_flat.to("cuda"), pack)["proj"]
        self.assertTrue(pack.linears[0].fp8_qualifies)

        import torch.nn.functional as F
        from toolkit.memory_management.ingraph_stream import streamed_linear

        x = torch.randn(2, 3, 16, device="cuda", dtype=torch.bfloat16)
        expected = F.linear(x, view.materialized_weight().to(x.dtype), view.bias)
        actual = streamed_linear(x, view)
        torch.testing.assert_close(actual, expected, rtol=0.08, atol=0.08)

        compiled = torch.compile(
            lambda inp, qdata, scale, bias: streamed_linear(
                inp,
                type(view)(spec=view.spec, weight=qdata, scale=scale, bias=bias),
            ),
            fullgraph=True,
            dynamic=False,
        )
        compiled_actual = compiled(x, view.weight, view.scale, view.bias)
        torch.testing.assert_close(compiled_actual, actual, rtol=0.08, atol=0.08)

    def test_quantized_repoint_preserves_state_dict(self):
        layer = self._quanto_linear()
        expected = {k: v.detach().clone() for k, v in layer.state_dict().items()}

        pack = pack_block_host("blocks.0", [("proj", layer)], repoint=True, pin=False)

        self.assertEqual(
            pack.host_flat.untyped_storage().data_ptr(),
            layer.weight.data._data.untyped_storage().data_ptr(),
        )
        after = layer.state_dict()
        self.assertEqual(set(expected), set(after))
        for key, value in expected.items():
            self.assertTrue(torch.equal(value, after[key]), key)


class PackHandleOwnershipTests(unittest.TestCase):
    """Slice 0: pack_block_host must carry its own pin grant so callers can
    release it, and never release a grant it doesn't own (borrowed_from_arena
    packs in a later slice). Regression coverage for the 0.40 GiB leak
    (ticket 763bb75): pack_block_host used to allocate via pin_manager.pin_alloc
    but discard the returned PinHandle entirely."""

    def test_pack_carries_pin_handle_and_owns_flat_by_default(self):
        layer = torch.nn.Linear(8, 4, bias=True)
        pack = pack_block_host("blocks.0", [("proj", layer)], repoint=False, pin=True)
        self.assertIsNotNone(pack.pin_handle)
        self.assertTrue(pack.owns_flat)
        self.assertFalse(pack.borrowed_from_arena)

    def test_release_pack_releases_owned_handle(self):
        layer = torch.nn.Linear(8, 4, bias=True)
        pack = pack_block_host("blocks.0", [("proj", layer)], repoint=False, pin=True)
        handle = pack.pin_handle
        with mock.patch.object(pin_manager, "release") as released:
            release_pack(pack)
        released.assert_called_once_with(handle)
        self.assertIsNone(pack.pin_handle)

    def test_release_pack_is_noop_for_borrowed_flat(self):
        layer = torch.nn.Linear(8, 4, bias=True)
        pack = pack_block_host("blocks.0", [("proj", layer)], repoint=False, pin=False)
        pack.owns_flat = False
        pack.borrowed_from_arena = True
        sentinel = object()
        pack.pin_handle = sentinel
        with mock.patch.object(pin_manager, "release") as released:
            release_pack(pack)
        released.assert_not_called()
        self.assertIs(pack.pin_handle, sentinel)

    def test_release_pack_handles_none(self):
        release_pack(None)

    def test_build_failure_after_flat_alloc_releases_handle(self):
        from toolkit.memory_management import ingraph_stream

        layer = torch.nn.Linear(8, 4, bias=True)
        with mock.patch.object(pin_manager, "release") as released:
            with mock.patch.object(
                ingraph_stream, "LinearSpec", side_effect=RuntimeError("boom")
            ):
                with self.assertRaises(RuntimeError):
                    pack_block_host("blocks.0", [("proj", layer)], repoint=False, pin=True)
        released.assert_called_once()


if __name__ == "__main__":
    unittest.main()
