"""_dequantize_to must not transit fp32 for TorchAO fp8 weights (git-bug 1895607).

TorchAO's dequantize() allocates ~5x the output bytes internally (fp8 -> fp32
-> mul -> cast) even with output_dtype=bf16. The fast path casts fp8 straight
to the compute dtype and applies the row scales in place, allocating only the
output. Under the WDDM hard allocator cap the difference is an OOM-demote
spiral vs a clean forward when fp8-native sampling is off.
"""

import unittest
from unittest import mock

import pytest
import torch

from toolkit.quantization import fp8_linear as fp8
from toolkit.quantization.storage import linear_storage_binding


def _current_backend_weights():
    from optimum.quanto import qfloat8
    from optimum.quanto.tensor.qweight import quantize_weight
    from torchao.quantization import Float8Tensor

    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    return (
        Float8Tensor.from_hp(weight),
        quantize_weight(weight, qfloat8, axis=0),
    )


def test_torchao_and_quanto_normalize_to_rowwise_fp8_semantics():
    torchao_weight, quanto_weight = _current_backend_weights()
    declarations = tuple(
        fp8.declare_fp8_linear(weight)
        for weight in (torchao_weight, quanto_weight)
    )

    for declaration in declarations:
        assert declaration is not None
        assert declaration.spec.weight_dtype == torch.float8_e4m3fn
        assert declaration.spec.activation_dtype == torch.float8_e4m3fn
        assert declaration.spec.scale_granularity == "output_row"
        assert not declaration.spec.has_zero_point
        assert declaration.spec.weight_layout == "out_in"
        assert declaration.spec.execution_variant == "scaled_mm_dynamic_activation"

    for weight in (torchao_weight, quanto_weight):
        binding = linear_storage_binding(weight)
        assert tuple(item.name for item in binding.tensors) == ("qdata", "scale")
        operation, tensors = fp8.bind_parameter_operation(
            weight,
            device="cpu",
        )
        assert operation.format_key == "rowwise_fp8"
        assert not operation.native
        torch.testing.assert_close(
            operation.materialize(tensors),
            weight.dequantize().to(torch.bfloat16),
        )


def test_e5m2_is_declared_but_materializes_instead_of_binding_native():
    from optimum.quanto import qfloat8_e5m2
    from optimum.quanto.tensor.qweight import quantize_weight

    weight = quantize_weight(
        torch.randn(32, 64, dtype=torch.bfloat16),
        qfloat8_e5m2,
        axis=0,
    )
    declaration = fp8.declare_fp8_linear(weight)
    assert declaration.spec.weight_dtype == torch.float8_e5m2
    with mock.patch.object(fp8, "native_device_supported", return_value=True):
        operation, tensors = fp8.bind_parameter_operation(weight, device="cuda")
    assert not operation.native
    assert operation.format_key == "fp8"
    torch.testing.assert_close(
        operation.materialize(tensors),
        weight.dequantize().to(torch.bfloat16),
    )


def test_opaque_two_leaf_tuple_is_not_inferred_as_rowwise_fp8():
    qdata = torch.zeros((32, 64), dtype=torch.float8_e4m3fn)
    scale = torch.ones(32, dtype=torch.float32)
    with pytest.raises(ValueError, match="unsupported_linear_storage_operation"):
        fp8.bind_storage_operation(
            (qdata, scale),
            device="cpu",
            weight_leaf_count=2,
            execution_key=("unknown_backend",),
        )


def test_native_and_fallback_bindings_share_the_same_storage_tuple():
    qdata = torch.zeros((32, 64), dtype=torch.float8_e4m3fn)
    scale = torch.ones(32, dtype=torch.float32)
    bias = torch.zeros(32, dtype=torch.bfloat16)
    with mock.patch.object(fp8, "native_device_supported", return_value=True):
        native = fp8.bind_rowwise_fp8(
            qdata,
            scale,
            device="cuda",
            has_bias=True,
        )
    with mock.patch.object(fp8, "native_device_supported", return_value=False):
        fallback = fp8.bind_rowwise_fp8(
            qdata,
            scale,
            device="cuda",
            has_bias=True,
        )

    assert native.native
    assert not fallback.native
    expected = (qdata, scale, bias)
    for binding in (native, fallback):
        actual = binding.explicit_tensors(qdata, bias, scale)
        assert all(got is want for got, want in zip(actual, expected))


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
        self._verified = fp8._REUSE_VERIFIED
        fp8._REUSE_VERIFIED = None
        self.addCleanup(lambda: setattr(fp8, "_REUSE_VERIFIED", self._verified))

    def test_matches_reference_dequant(self):
        weight = _quantized_linear()
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            fast = fp8.dequantize_to(weight, dtype)
            reference = fp8.reference_dequantize_to(weight, dtype)
            self.assertEqual(fast.dtype, dtype)
            self.assertEqual(fast.shape, reference.shape)
            torch.testing.assert_close(
                fast.float(), reference.float(), rtol=1e-2, atol=1e-2
            )

    def test_allocates_only_the_output(self):
        weight = _quantized_linear()
        out_bytes = weight.shape[0] * weight.shape[1] * 2  # bf16 output
        # Warm up (first call pays the one-time verification allocation).
        fp8.dequantize_to(weight, torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        result = fp8.dequantize_to(weight, torch.bfloat16)
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

        out = fp8.dequantize_to(FakeQuantized(), torch.bfloat16)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(out.float(), plain.float(), rtol=1e-2, atol=1e-2)

    def test_disabled_verification_falls_back(self):
        weight = _quantized_linear(out_features=64, in_features=64)
        fp8._REUSE_VERIFIED = False
        out = fp8.dequantize_to(weight, torch.bfloat16)
        reference = fp8.reference_dequantize_to(weight, torch.bfloat16)
        torch.testing.assert_close(
            out.float(), reference.float(), rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
