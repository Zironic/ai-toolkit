"""Tiled 1-byte transpose op used for the fp8 grad-input B operand."""

import unittest

import torch

from toolkit.quantization.fp8_transpose import column_major

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "needs CUDA")
class TransposeContiguous1ByteTests(unittest.TestCase):
    SHAPES = ((64, 64), (3072, 3072), (12288, 3072), (3072, 12288), (80, 176))

    def test_matches_t_contiguous_bitwise(self):
        for m, n in self.SHAPES:
            with self.subTest(shape=(m, n)):
                x = (torch.randn(m, n, device="cuda") * 0.1).to(torch.float8_e4m3fn)
                got = torch.ops.mm.transpose_contiguous_1byte(x)
                want = x.t().contiguous()
                self.assertEqual(tuple(got.shape), (n, m))
                self.assertTrue(got.is_contiguous())
                self.assertTrue(
                    torch.equal(got.view(torch.uint8), want.view(torch.uint8))
                )

    def test_column_major_operand_is_scaled_mm_ready(self):
        # (M, N) back in the original shape, but column-major: stride (1, M).
        x = (torch.randn(128, 64, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        b = column_major(x)
        self.assertEqual(tuple(b.shape), (128, 64))
        self.assertEqual(b.stride(), (1, 128))
        self.assertTrue(
            torch.equal(b.view(torch.uint8), x.view(torch.uint8))
        )

    def test_handles_non_tile_multiple_shapes(self):
        # 80x176 is not a multiple of the 64 tile: the masked edges must still copy.
        x = (torch.randn(80, 176, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        got = torch.ops.mm.transpose_contiguous_1byte(x)
        self.assertTrue(
            torch.equal(got.view(torch.uint8), x.t().contiguous().view(torch.uint8))
        )

    def test_compiles_without_graph_breaks(self):
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()

        @torch.compile(fullgraph=True, dynamic=False)
        def f(w, a):
            return torch._scaled_mm(
                a,
                column_major(w),
                scale_a=torch.ones((), device="cuda"),
                scale_b=torch.ones((), device="cuda"),
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )

        w = (torch.randn(64, 32, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        a = (torch.randn(16, 64, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        out = f(w, a)
        self.assertEqual(tuple(out.shape), (16, 32))
        breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
        self.assertEqual(breaks, 0)


if __name__ == "__main__":
    unittest.main()
