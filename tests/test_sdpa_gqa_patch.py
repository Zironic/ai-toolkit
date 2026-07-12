import unittest

import torch
import torch.nn.functional as F

import toolkit  # noqa: F401  -- package init applies the patch
from toolkit import sdpa_patch


class SdpaGqaPatchTests(unittest.TestCase):
    """Global native-cuDNN or KV-expansion dispatch for GQA SDPA calls."""

    def test_patch_installed_and_idempotent(self):
        fn = F.scaled_dot_product_attention
        self.assertTrue(getattr(fn, "_aitk_gqa_patch", False))
        sdpa_patch.apply_sdpa_gqa_patch()
        self.assertIs(F.scaled_dot_product_attention, fn)

    def _gqa_tensors(
        self, device, dtype=torch.float32, heads_q=8, heads_kv=2,
        length=64, head_dim=32,
    ):
        torch.manual_seed(0)
        q = torch.randn(2, heads_q, length, head_dim, device=device, dtype=dtype)
        k = torch.randn(2, heads_kv, length, head_dim, device=device, dtype=dtype)
        v = torch.randn(2, heads_kv, length, head_dim, device=device, dtype=dtype)
        return q, k, v

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_gqa_matches_manual_expansion(self):
        q, k, v = self._gqa_tensors("cuda")
        groups = q.shape[1] // k.shape[1]
        expected = F.scaled_dot_product_attention(
            q, k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        )
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_gqa_with_mask_matches_manual_expansion(self):
        q, k, v = self._gqa_tensors("cuda")
        mask = torch.ones(2, 1, 1, 64, device="cuda", dtype=torch.bool)
        mask[..., -5:] = False
        groups = q.shape[1] // k.shape[1]
        expected = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(groups, dim=1),
            v.repeat_interleave(groups, dim=1),
            attn_mask=mask,
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        torch.testing.assert_close(out, expected, rtol=2e-3, atol=2e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_native_cudnn_gqa_is_eligible_and_compiles(self):
        if not sdpa_patch._CUDNN_AVAILABLE:
            self.skipTest("cuDNN SDPA is unavailable")
        q, k, v = self._gqa_tensors(
            "cuda", dtype=torch.bfloat16, heads_q=48, heads_kv=12,
            length=128, head_dim=128,
        )
        mask = torch.ones(2, 1, 1, 128, device="cuda", dtype=torch.bool)
        mask[..., -5:] = False
        self.assertTrue(sdpa_patch.can_use_native_cudnn_gqa(q, k, v, mask))

        def call(q, k, v, mask):
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, enable_gqa=True
            )

        eager = call(q, k, v, mask)
        compiled = torch.compile(call, dynamic=True)(q, k, v, mask)
        torch.testing.assert_close(compiled, eager, rtol=2e-3, atol=2e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_expansion_fallback_is_efficient_eligible(self):
        """The retained fallback must still qualify for efficient SDPA."""
        if sdpa_patch._FLASH_AVAILABLE:
            self.skipTest("build has Flash; unmasked GQA stays native")
        q, k, v = self._gqa_tensors("cuda", dtype=torch.bfloat16, length=128)
        groups = q.shape[1] // k.shape[1]
        k_x = k.repeat_interleave(groups, dim=1)
        v_x = v.repeat_interleave(groups, dim=1)
        bc = torch.backends.cuda
        self.assertFalse(
            bc.can_use_efficient_attention(
                bc.SDPAParams(q, k, v, None, 0.0, False, True), False
            )
        )
        self.assertTrue(
            bc.can_use_efficient_attention(
                bc.SDPAParams(q, k_x, v_x, None, 0.0, False, False), False
            )
        )

    def test_cpu_passthrough_unchanged(self):
        q, k, v = self._gqa_tensors("cpu")
        out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        groups = q.shape[1] // k.shape[1]
        expected = F.scaled_dot_product_attention(
            q, k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        )
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)

    def test_non_gqa_calls_untouched(self):
        q = torch.randn(2, 4, 16, 8)
        out = F.scaled_dot_product_attention(q, q, q)
        self.assertEqual(tuple(out.shape), (2, 4, 16, 8))


if __name__ == "__main__":
    unittest.main()
