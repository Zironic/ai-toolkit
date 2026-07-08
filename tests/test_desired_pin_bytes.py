"""Ticket 534ea49 Phase 2 Slice A: the auto/explicit pin-budget sizing must be
scoped to the layers ACTUALLY selected for streaming (offload_ids), never the
whole model. attach_smart_training previously sized auto-pin from
plan["model_bytes"] (the whole model) even when a smart-training plan kept
some blocks resident -- this is the single source of truth both attach() and
attach_smart_training() now share."""

import unittest

import torch
import torch.nn as nn

from toolkit.memory_management.manager import MemoryManager

GIB = 1024 ** 3


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 64, bias=True)   # streamed
        self.b = nn.Linear(64, 64, bias=True)   # streamed
        self.c = nn.Linear(128, 128, bias=True)  # resident (NOT in offload_ids)


class DesiredPinBytesForOffloadIdsTests(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.streamed_ids = {id(self.model.a), id(self.model.b)}

    def _bytes_of(self, *modules):
        total = 0
        for m in modules:
            for p in (m.weight, m.bias):
                total += p.numel() * p.element_size()
        return total

    def test_auto_mode_sums_only_offload_ids_not_whole_model(self):
        expected_streamed = self._bytes_of(self.model.a, self.model.b)
        expected_whole_model = expected_streamed + self._bytes_of(self.model.c)

        got = MemoryManager._desired_pin_bytes_for_offload_ids(
            self.model, self.streamed_ids, None
        )
        self.assertEqual(got, int(expected_streamed * 1.03))
        # The bug this fixes: sizing from the whole model would be strictly
        # larger whenever any layer is kept resident.
        self.assertLess(got, int(expected_whole_model * 1.03))

    def test_negative_pinned_weight_gib_is_treated_as_auto(self):
        got_none = MemoryManager._desired_pin_bytes_for_offload_ids(
            self.model, self.streamed_ids, None
        )
        got_negative = MemoryManager._desired_pin_bytes_for_offload_ids(
            self.model, self.streamed_ids, -1.0
        )
        self.assertEqual(got_none, got_negative)

    def test_explicit_pinned_weight_gib_ignores_offload_ids(self):
        got = MemoryManager._desired_pin_bytes_for_offload_ids(
            self.model, self.streamed_ids, 3.5
        )
        self.assertEqual(got, int(3.5 * GIB))
        # Explicit budget is a flat request -- an empty offload set must not
        # change it.
        got_empty = MemoryManager._desired_pin_bytes_for_offload_ids(
            self.model, set(), 3.5
        )
        self.assertEqual(got_empty, int(3.5 * GIB))

    def test_empty_offload_ids_auto_mode_is_zero(self):
        got = MemoryManager._desired_pin_bytes_for_offload_ids(
            self.model, set(), None
        )
        self.assertEqual(got, 0)

    def test_unparseable_pinned_weight_gib_falls_back_to_auto(self):
        expected_streamed = self._bytes_of(self.model.a, self.model.b)
        got = MemoryManager._desired_pin_bytes_for_offload_ids(
            self.model, self.streamed_ids, "not-a-number"
        )
        self.assertEqual(got, int(expected_streamed * 1.03))

    def test_quantized_weights_are_counted_at_physical_not_logical_bytes(self):
        """Regression (live run): a quanto FP8 weight reports its LOGICAL
        dtype through numel()*element_size() (bf16 = 2 bytes/elem), exactly
        2x its physical 1-byte qdata -- producing a nonsense want=20.42 GiB
        auto-pin request for a ~9.9 GiB model. Physical leaf bytes
        (_tensor_storage_bytes) must be used instead."""
        from optimum.quanto import freeze
        from toolkit.util.quantize import get_qtype, quantize

        model = nn.Sequential(nn.Linear(64, 64, bias=False).to(torch.bfloat16))
        quantize(model, weights=get_qtype("qfloat8"))
        freeze(model)
        layer = model[0]

        logical = layer.weight.numel() * layer.weight.element_size()
        physical = MemoryManager._tensor_storage_bytes(layer.weight.data)
        self.assertLess(physical, logical)  # fp8 qdata+scale < bf16 view

        got = MemoryManager._desired_pin_bytes_for_offload_ids(
            model, {id(layer)}, None
        )
        self.assertEqual(got, int(physical * 1.03))


if __name__ == "__main__":
    unittest.main()
