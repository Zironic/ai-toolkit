"""Rung 1 (Phase 4-pre) de-risk at synthetic scale, per INGRAPH_STREAM_PLAN.

Production training on the reference card is keep_last=0 / fully streamed, so
resident-block compile is never exercised at Krea2 scale; these tests carry
Rung 1's three risk items on a synthetic model instead, gating Rung 2:

1. `_fp8_linear_training` (grad-safe native FP8 linear) forward/backward
   numerics, eager and compiled.
2. LoRA in the AOTAutograd joint graph: frozen fp8 base + trainable A/B as
   ordinary graph inputs, fullgraph, gradients flow.
3. checkpoint(use_reentrant=False) x torch.compile over a multi-block trunk:
   zero breaks over fwd+bwd, stable across steps, grad parity vs eager.

Parity note (Phase 0 lesson): compiled-vs-eager bf16 is never bitwise
(Inductor epilogue rounding) -- comparisons use FP8-scale tolerances.
"""

import unittest

import torch

from toolkit.memory_management.manager_modules import _fp8_linear_training

FP8 = torch.finfo(torch.float8_e4m3fn)


def _quantize_rowwise(w: torch.Tensor):
    """(out, in) bf16 -> (qdata fp8 (out, in), scale fp32 (out,))."""
    scale = (w.abs().amax(dim=1).float() / FP8.max).clamp(
        min=torch.finfo(torch.float32).tiny
    )
    qdata = (
        (w.float() / scale[:, None]).clamp(FP8.min, FP8.max).to(torch.float8_e4m3fn)
    )
    return qdata, scale


def _dequant(qdata: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return qdata.float() * scale[:, None]


def _graph_breaks() -> int:
    return sum(torch._dynamo.utils.counters["graph_break"].values())


def _unique_graphs() -> int:
    return int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 9),
    "needs CUDA with native FP8 (_scaled_mm)",
)
class Fp8LinearTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cuda")
        w = torch.randn(64, 64, device=self.device, dtype=torch.bfloat16) * 0.05
        self.qdata, self.scale = _quantize_rowwise(w)
        self.qdata_t = self.qdata.t()
        self.w_ref = _dequant(self.qdata, self.scale).to(torch.bfloat16)

    def test_forward_backward_parity_eager(self):
        x = torch.randn(
            4, 16, 64, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        out = _fp8_linear_training(x, self.qdata_t, self.scale, None)
        ref = x.detach() @ self.w_ref.t()
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=8e-2)

        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        ref_grad = grad_out @ self.w_ref
        torch.testing.assert_close(x.grad, ref_grad, rtol=5e-2, atol=8e-2)

    def test_compiles_fullgraph_zero_breaks(self):
        torch._dynamo.reset()
        compiled = torch.compile(
            _fp8_linear_training, fullgraph=True, dynamic=False
        )
        x = torch.randn(
            4, 16, 64, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        breaks_before = _graph_breaks()
        out = compiled(x, self.qdata_t, self.scale, None)
        out.square().mean().backward()
        self.assertEqual(_graph_breaks(), breaks_before)
        self.assertIsNotNone(x.grad)
        ref = x.detach() @ self.w_ref.t()
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=8e-2)


def _lora_block(x, qdata_t, scale_row, A, B):
    """Frozen fp8 base + compile-clean LoRA fast path + nonlinearity.

    Mirrors _memory_management_compile_fast_lora_forward: LoRA computed in the
    adapter dtype (fp32, as network.force_to sets), scaled, cast back."""
    base = _fp8_linear_training(x, qdata_t, scale_row, None)
    lora = (x.to(A.dtype) @ A.t() @ B.t()) * 2.0
    return torch.nn.functional.silu(base + lora.to(base.dtype))


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 9),
    "needs CUDA with native FP8 (_scaled_mm)",
)
class LoraJointGraphTests(unittest.TestCase):
    N_BLOCKS = 2
    RANK = 8

    def setUp(self):
        torch.manual_seed(11)
        self.device = torch.device("cuda")
        self.packs = []
        self.params = []
        for _ in range(self.N_BLOCKS):
            w = torch.randn(64, 64, device=self.device, dtype=torch.bfloat16) * 0.05
            qdata, scale = _quantize_rowwise(w)
            self.packs.append((qdata.t(), scale))
            A = torch.randn(
                self.RANK, 64, device=self.device, dtype=torch.float32
            ).mul_(0.01).requires_grad_(True)
            B = torch.zeros(
                64, self.RANK, device=self.device, dtype=torch.float32
            ).requires_grad_(True)
            # B starts at zero like a real LoRA; nudge it so grads flow to A too.
            with torch.no_grad():
                B.add_(torch.randn_like(B) * 0.01)
            self.params.append((A, B))

    def _trunk(self, x, use_checkpoint):
        for (qdata_t, scale), (A, B) in zip(self.packs, self.params):
            if use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(
                    _lora_block, x, qdata_t, scale, A, B, use_reentrant=False
                )
            else:
                x = _lora_block(x, qdata_t, scale, A, B)
        return x

    def _zero_grads(self):
        for A, B in self.params:
            A.grad = None
            B.grad = None

    def _collect_grads(self):
        return [
            (A.grad.detach().clone(), B.grad.detach().clone())
            for A, B in self.params
        ]

    def test_lora_grads_flow_in_compiled_joint_graph(self):
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda x: self._trunk(x, use_checkpoint=False),
            fullgraph=True,
            dynamic=False,
        )
        x = torch.randn(4, 16, 64, device=self.device, dtype=torch.bfloat16)

        breaks_before = _graph_breaks()
        self._zero_grads()
        compiled(x).square().mean().backward()
        self.assertEqual(_graph_breaks(), breaks_before)
        compiled_grads = self._collect_grads()

        self._zero_grads()
        self._trunk(x, use_checkpoint=False).square().mean().backward()
        eager_grads = self._collect_grads()

        for (ca, cb), (ea, eb) in zip(compiled_grads, eager_grads):
            self.assertGreater(ca.abs().sum().item(), 0.0)
            self.assertGreater(cb.abs().sum().item(), 0.0)
            torch.testing.assert_close(ca, ea, rtol=3e-2, atol=3e-4)
            torch.testing.assert_close(cb, eb, rtol=3e-2, atol=3e-4)

    def test_checkpoint_x_compile_zero_breaks_and_stable(self):
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda x: self._trunk(x, use_checkpoint=True),
            fullgraph=True,
            dynamic=False,
        )
        x = torch.randn(4, 16, 64, device=self.device, dtype=torch.bfloat16)

        breaks_before = _graph_breaks()
        self._zero_grads()
        compiled(x).square().mean().backward()
        self.assertEqual(_graph_breaks(), breaks_before)
        graphs_after_first = _unique_graphs()
        first_grads = self._collect_grads()

        # Repeat steps: zero recompiles, grads keep flowing.
        for _ in range(2):
            self._zero_grads()
            compiled(x).square().mean().backward()
        self.assertEqual(_unique_graphs(), graphs_after_first)

        # Checkpointed grads match the uncheckpointed eager reference.
        self._zero_grads()
        self._trunk(x, use_checkpoint=False).square().mean().backward()
        eager_grads = self._collect_grads()
        for (ca, cb), (ea, eb) in zip(first_grads, eager_grads):
            torch.testing.assert_close(ca, ea, rtol=3e-2, atol=3e-4)
            torch.testing.assert_close(cb, eb, rtol=3e-2, atol=3e-4)


if __name__ == "__main__":
    unittest.main()
