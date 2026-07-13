"""layer_offloading_fp8_grad_input must actually select the backward path.

The compiled fp8 Linear (_Fp8LinearTrainingFn) used to call the native fp8
grad-input GEMM unconditionally, so the flag was inert whenever fp8 forward was
on: users who disabled it to protect gradient quality still got it. These tests
pin the flag to the branch it chooses.
"""

import unittest
import unittest.mock

import torch

from toolkit.quantization import fp8_linear
from toolkit.quantization.fp8_linear import (
    native_linear_training,
    set_fp8_grad_input_enabled,
)

CUDA_FP8 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 9)
    and hasattr(torch, "_scaled_mm")
)


def _rowwise_fp8_weight(out_features, in_features, device):
    weight = torch.randn(out_features, in_features, device=device) * 0.05
    scale = weight.abs().amax(dim=1, keepdim=True) / torch.finfo(
        torch.float8_e4m3fn
    ).max
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    qdata = (weight / scale).to(torch.float8_e4m3fn)
    # The Linear takes the weight already transposed to (K, N).
    return qdata.t(), scale.reshape(-1).float()


@unittest.skipUnless(CUDA_FP8, "needs an sm89+ CUDA device with _scaled_mm")
class Fp8GradInputFlagTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.addCleanup(set_fp8_grad_input_enabled, False)
        self.calls = []
        real = fp8_linear._grad_input_compute

        def counting(*args, **kwargs):
            self.calls.append(1)
            return real(*args, **kwargs)

        patched = unittest.mock.patch.object(
            fp8_linear, "_grad_input_compute", counting
        )
        patched.start()
        self.addCleanup(patched.stop)

    def _backward(self, enabled):
        set_fp8_grad_input_enabled(enabled)
        torch.manual_seed(0)
        qdata_t, scale_row = _rowwise_fp8_weight(64, 32, self.device)
        x = torch.randn(
            8, 32, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        out = native_linear_training(x, qdata_t, scale_row, None)
        out.sum().backward()
        return x.grad

    def test_enabled_uses_the_native_fp8_grad_input(self):
        grad = self._backward(True)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(torch.isfinite(grad).all())

    def test_disabled_does_not_touch_the_fp8_grad_input(self):
        grad = self._backward(False)
        self.assertEqual(self.calls, [])
        self.assertTrue(torch.isfinite(grad).all())

    def test_both_paths_agree_within_fp8_tolerance(self):
        enabled = self._backward(True).float()
        self.calls.clear()
        disabled = self._backward(False).float()
        # Same weights/inputs (seeded); the fp8 path quantizes grad_out, so it is
        # close but not bitwise equal to the dequantized bf16 matmul.
        torch.testing.assert_close(enabled, disabled, rtol=0.05, atol=0.02)

    def test_disabled_backward_never_materializes_an_fp32_weight(self):
        # The old fallback transited fp32 (4x the fp8 weight) per Linear. Dequant
        # goes straight to the compute dtype now.
        set_fp8_grad_input_enabled(False)
        qdata_t, scale_row = _rowwise_fp8_weight(512, 256, self.device)
        x = torch.randn(
            16, 256, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        out = native_linear_training(x, qdata_t, scale_row, None)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        out.sum().backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - before
        fp8_weight_bytes = 512 * 256
        # Dequant-to-compute-dtype holds at most two bf16 temporaries (the cast
        # and the row-scaled product) = 4x the fp8 weight. Transiting fp32 adds a
        # further 4x buffer on top, so 5x separates the two shapes cleanly.
        self.assertLess(peak, 5 * fp8_weight_bytes)


if __name__ == "__main__":
    unittest.main()
