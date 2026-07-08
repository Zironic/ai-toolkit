"""_dequantize_to must not transit fp32 for TorchAO fp8 weights (git-bug 1895607).

TorchAO's dequantize() allocates ~5x the output bytes internally (fp8 -> fp32
-> mul -> cast) even with output_dtype=bf16. The fast path casts fp8 straight
to the compute dtype and applies the row scales in place, allocating only the
output. Under the WDDM hard allocator cap the difference is an OOM-demote
spiral vs a clean forward when fp8-native sampling is off.
"""

import unittest

import torch

from toolkit.memory_management import manager_modules as mm


def _quantized_linear(out_features=4096, in_features=1536):
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    layer = torch.nn.Linear(out_features=out_features, in_features=in_features).to(
        "cuda", torch.bfloat16
    )
    quantize_(layer, Float8WeightOnlyConfig())
    return layer.weight


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class FastFp8DequantTests(unittest.TestCase):
    def setUp(self):
        self._verified = mm._REUSE_VERIFIED
        mm._REUSE_VERIFIED = None
        self.addCleanup(lambda: setattr(mm, "_REUSE_VERIFIED", self._verified))

    def test_matches_reference_dequant(self):
        weight = _quantized_linear()
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            fast = mm._dequantize_to(weight, dtype)
            reference = mm._reference_dequantize_to(weight, dtype)
            self.assertEqual(fast.dtype, dtype)
            self.assertEqual(fast.shape, reference.shape)
            torch.testing.assert_close(
                fast.float(), reference.float(), rtol=1e-2, atol=1e-2
            )

    def test_allocates_only_the_output(self):
        weight = _quantized_linear()
        out_bytes = weight.shape[0] * weight.shape[1] * 2  # bf16 output
        # Warm up (first call pays the one-time verification allocation).
        mm._dequantize_to(weight, torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        result = mm._dequantize_to(weight, torch.bfloat16)
        torch.cuda.synchronize()
        transient = torch.cuda.max_memory_allocated() - base
        self.assertIsNotNone(result)
        # Reference path would take ~5x the output; require at most ~1.2x
        # (allocator rounding slack only).
        self.assertLessEqual(transient, int(out_bytes * 1.2))

    def test_non_fp8_falls_back_to_reference(self):
        plain = torch.randn(8, 8, device="cuda", dtype=torch.float32)

        class FakeQuantized:
            # no qdata attribute -> fast path must decline
            def dequantize(self, output_dtype=None):
                return plain.to(output_dtype or torch.float32)

        out = mm._dequantize_to(FakeQuantized(), torch.bfloat16)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out.float(), plain.float(), rtol=1e-2, atol=1e-2)

    def test_disabled_verification_falls_back(self):
        weight = _quantized_linear(out_features=64, in_features=64)
        mm._REUSE_VERIFIED = False
        out = mm._dequantize_to(weight, torch.bfloat16)
        reference = mm._reference_dequantize_to(weight, torch.bfloat16)
        torch.testing.assert_close(
            out.float(), reference.float(), rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
