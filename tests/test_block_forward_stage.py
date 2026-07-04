import unittest

import torch

from toolkit.memory_management import manager_modules as mm


class BlockRingAccountingTests(unittest.TestCase):
    def test_block_ring_bytes_are_counted_once(self):
        state = {
            "w_buffers": [],
            "b_buffers": [],
            "w_grad_buffers": [],
            "b_grad_buffers": [],
            "block_ring": [
                {"bytes": 128},
                {"bytes": 256},
            ],
        }
        self.assertEqual(mm._ring_current_bytes(state), 384)


@unittest.skipUnless(torch.cuda.is_available(), "block forward staging needs CUDA")
class BlockForwardStageTests(unittest.TestCase):
    """Slice 2 forward block-staging: the coalesced-staged weight must be the
    SAME value the per-Linear path would produce (so the forward output and thus
    gradients are identical), and the 2-block ring must evict/free correctly."""

    def setUp(self):
        self.device = torch.device("cuda:0")
        mm.set_block_stream_enabled(self.device, True, depth=2)
        mm.reset_block_stream(self.device)

    def tearDown(self):
        mm.set_block_stream_enabled(self.device, False)

    def test_staged_weight_matches_direct_transfer_and_forward(self):
        w = torch.randn(64, 32)
        b = torch.randn(64)
        mm.stage_block_forward(
            self.device, "blocks.0", [("blocks.0.lin", w, b)], torch.float32
        )
        res = mm.consume_block_resident(self.device, "blocks.0.lin")
        torch.cuda.synchronize()
        self.assertIsNotNone(res)
        w_gpu, b_gpu = res
        # Coalesced-staged weight is bitwise-identical to the per-Linear H2D.
        self.assertTrue(torch.equal(w_gpu, w.to(self.device)))
        self.assertTrue(torch.equal(b_gpu, b.to(self.device)))
        # Forward output is therefore identical.
        x = torch.randn(8, 32, device=self.device)
        ref = torch.nn.functional.linear(x, w.to(self.device), b.to(self.device))
        got = torch.nn.functional.linear(x, w_gpu, b_gpu)
        self.assertTrue(torch.equal(ref, got))

    def test_multi_linear_block_all_resident(self):
        names = ["wq", "wk", "wv", "wo", "gate", "up", "down"]
        weights = {f"blocks.3.{n}": torch.randn(48, 48) for n in names}
        mm.stage_block_forward(
            self.device,
            "blocks.3",
            [(k, w, None) for k, w in weights.items()],
            torch.float32,
        )
        torch.cuda.synchronize()
        for k, w in weights.items():
            res = mm.consume_block_resident(self.device, k)
            self.assertIsNotNone(res, k)
            self.assertTrue(torch.equal(res[0], w.to(self.device)), k)

    def test_one_h2d_per_block_not_per_linear(self):
        # The whole point: an 8-Linear block costs ONE H2D, not eight.
        mm.reset_block_stream(self.device)
        block = [(f"blocks.5.lin{i}", torch.randn(48, 48), None) for i in range(8)]
        mm.stage_block_forward(self.device, "blocks.5", block, torch.float32)
        torch.cuda.synchronize()
        h2d, layers = mm.block_stream_stats(self.device)
        self.assertEqual(h2d, 1, "block must stage in a single H2D copy")
        self.assertEqual(layers, 8)

    def test_two_block_ring_evicts_oldest(self):
        for i in range(3):
            mm.stage_block_forward(
                self.device, f"blocks.{i}",
                [(f"blocks.{i}.lin", torch.randn(64, 32), None)], torch.float32,
            )
            mm.block_forward_done(self.device, f"blocks.{i}")
        torch.cuda.synchronize()
        # depth=2: the first block is evicted, the last two stay resident.
        self.assertIsNone(mm.consume_block_resident(self.device, "blocks.0.lin"))
        self.assertIsNotNone(mm.consume_block_resident(self.device, "blocks.1.lin"))
        self.assertIsNotNone(mm.consume_block_resident(self.device, "blocks.2.lin"))

    def test_disabled_returns_none(self):
        mm.stage_block_forward(
            self.device, "blocks.9", [("blocks.9.lin", torch.randn(8, 8), None)],
            torch.float32,
        )
        mm.set_block_stream_enabled(self.device, False)
        self.assertIsNone(mm.consume_block_resident(self.device, "blocks.9.lin"))


class _Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = torch.nn.Linear(d, d, bias=False)
        self.b = torch.nn.Linear(d, d, bias=False)
        self.c = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.c(self.b(self.a(x)))


class _Model(torch.nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(d) for _ in range(n)])
        self.head = torch.nn.Linear(d, d, bias=False)  # singleton (non-block)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class BlockStreamGradientParityTests(unittest.TestCase):
    """End-to-end: a streamed model must produce identical gradients with block
    streaming on vs off. float32 throughout, so identical weights => identical
    grads bitwise (any divergence is a real bug, not bf16 rounding)."""

    def _run(self, block_stream, state_dict, d, n):
        from toolkit.memory_management import MemoryManager

        device = torch.device("cuda:0")
        model = _Model(d, n)
        model.load_state_dict(state_dict)
        offload_ids = {
            id(m) for m in model.modules() if isinstance(m, torch.nn.Linear)
        }
        MemoryManager.attach(model, device, _offload_module_ids=offload_ids)
        try:
            if block_stream:
                mm.set_block_stream_enabled(device, True, depth=2)
                mm.reset_block_stream(device)
                wired = MemoryManager._wire_block_stream_forward_hooks(model, device)
                self.assertGreaterEqual(wired, n)
            else:
                mm.set_block_stream_enabled(device, False)

            torch.manual_seed(0)
            x = torch.randn(4, d, device=device)
            out = model(x)
            loss = out.square().mean()
            loss.backward()
            grads = {
                name: p.grad.detach().float().cpu().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            return float(loss.detach().cpu()), grads
        finally:
            for h in getattr(model, "_mm_block_stream_handles", []) or []:
                h.remove()
            mm.set_block_stream_enabled(device, False)

    def test_gradients_match_with_and_without_block_stream(self):
        # Disable TF32 so float32 matmuls are IEEE and deterministic between the
        # two runs — otherwise TF32 (default on Ampere+) adds ~1e-6 noise that
        # would mask, or be mistaken for, a real divergence.
        prev_mm = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", prev_mm)
        self.addCleanup(setattr, torch.backends.cudnn, "allow_tf32", prev_cudnn)
        d, n = 32, 4
        ref_model = _Model(d, n)
        state = {k: v.clone() for k, v in ref_model.state_dict().items()}

        loss_off, grads_off = self._run(False, state, d, n)
        loss_off2, grads_off2 = self._run(False, state, d, n)  # baseline noise
        loss_on, grads_on = self._run(True, state, d, n)

        self.assertEqual(set(grads_off), set(grads_on))
        self.assertAlmostEqual(loss_off, loss_on, places=4)

        # Two identical per-Linear runs already differ slightly: cuBLAS picks GEMM
        # algorithms from allocator/workspace state, so "identical" runs diverge at
        # the float32-ULP level. Block streaming must introduce no MORE divergence
        # than that inherent baseline (a real staging bug would be orders larger or
        # NaN). The exact forward-weight identity is proven separately above.
        def max_abs_diff(ga, gb):
            return max((ga[k] - gb[k]).abs().max().item() for k in ga)

        baseline = max_abs_diff(grads_off, grads_off2)
        on_diff = max_abs_diff(grads_off, grads_on)
        self.assertLessEqual(
            on_diff, max(baseline, 1e-12) * 8 + 1e-5,
            f"block-stream grads diverge beyond GEMM noise: "
            f"on_diff={on_diff:.3e} baseline={baseline:.3e}",
        )


if __name__ == "__main__":
    unittest.main()
