import unittest
import torch
import torch.nn as nn

from optimum.quanto import freeze

from toolkit.util.quantize import quantize, get_qtype
from toolkit.memory_management.manager_modules import (
    _ensure_cpu_pinned,
    _move_params_to_cpu_and_pin,
    _pin_inner_tensors,
    _profile_bytes,
    _profile_is_pinned,
    _unpin_inner_tensors,
)


class _FakeManager:
    def __init__(self, budget_bytes=1 << 40):
        self.pinned_weight_budget_bytes = budget_bytes
        self.pinned_weight_bytes = 0


def _quanto_linear(d=64):
    model = nn.Sequential(nn.Linear(d, d, bias=False).to(torch.bfloat16))
    quantize(model, weights=get_qtype("qfloat8"))
    freeze(model)
    return model[0]


@unittest.skipUnless(torch.cuda.is_available(), "pinning requires CUDA")
class QuantoPinPathTests(unittest.TestCase):
    """quanto QBytesTensor weights (Krea2's qfloat8 format) must be pinned via
    their inner storage. Wrapper-level pin_memory() dispatches through quanto's
    dequantize fallback: it permanently costs ~2x the fp8 storage in host RAM
    and returns a wrapper whose inner data is still unpinned — that exhausted
    host RAM at attach and crashed startup with a raw cudaErrorMemoryAllocation.

    Disables the live available-RAM floor guard (AI_TOOLKIT_PINNED_WEIGHT_AVAILABLE_
    FLOOR_GIB, see test_pin_budget_caps.py): these tests exercise pin-path
    correctness, not host-RAM headroom, and must not depend on how much real
    RAM happens to be free on the machine running them."""

    def test_ensure_cpu_pinned_pins_inner_storage_in_place(self):
        lin = _quanto_linear()
        w = lin.weight.data
        ref = w.dequantize().clone()
        storage_bytes = _profile_bytes(w)
        logical_bytes = w.numel() * w.element_size()

        out, counted = _ensure_cpu_pinned(w, 1 << 40)

        # Same wrapper mutated in place — no dequantize round-trip copy.
        self.assertIs(out, w)
        self.assertTrue(out._data.is_pinned())
        self.assertTrue(_profile_is_pinned(out))
        # Counted at actual leaf storage size (fp8 + scale), not the logical
        # bf16 size the broken wrapper path reported.
        self.assertEqual(counted, storage_bytes)
        self.assertLess(counted, logical_bytes)
        # Values untouched.
        torch.testing.assert_close(out.dequantize(), ref)

    def test_move_params_pins_quanto_weight_and_counts_budget(self):
        lin = _quanto_linear()
        manager = _FakeManager()

        _move_params_to_cpu_and_pin(lin, manager)

        self.assertTrue(_profile_is_pinned(lin.weight.data))
        self.assertEqual(manager.pinned_weight_bytes, _profile_bytes(lin.weight.data))
        self.assertEqual(getattr(lin, "_mm_pinned_bytes", 0), manager.pinned_weight_bytes)
        # Still a quantized wrapper the streaming forward can use.
        self.assertTrue(hasattr(lin.weight.data, "_data"))
        self.assertEqual(lin.weight.data._data.dtype, torch.float8_e4m3fn)

    def test_zero_budget_pins_nothing(self):
        lin = _quanto_linear()
        manager = _FakeManager(budget_bytes=0)

        _move_params_to_cpu_and_pin(lin, manager)

        self.assertFalse(_profile_is_pinned(lin.weight.data))
        self.assertEqual(manager.pinned_weight_bytes, 0)

    def test_unpin_round_trip(self):
        lin = _quanto_linear()
        w = lin.weight.data
        pinned = _pin_inner_tensors(w, 1 << 40)
        self.assertGreater(pinned, 0)

        self.assertTrue(_unpin_inner_tensors(w))
        self.assertFalse(_profile_is_pinned(w))


if __name__ == "__main__":
    unittest.main()
