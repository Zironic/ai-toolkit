"""Phase 4a S2: training leaves path -- fp8 grad-safe base + LoRA entries.

End-to-end synthetic version of the streamed training block: real quanto-fp8
packs fetched through the ticket ops inside non-reentrant checkpoint, base
computed by the grad-safe fp8 Function, LoRA A/B as ordinary trainable graph
inputs. Verifies loss/grad parity vs a dequant eager reference, that no base
weight is saved for backward (eager saved-tensor probe), and the compiled
fullgraph variant with the ordering pass.
"""

import unittest

import torch

from toolkit.memory_management import ingraph_stream
from toolkit.memory_management.ingraph_stream import (
    LoraEntry,
    block_linear_views,
    free_on_backward,
    pack_block_host,
    streamed_linear,
)
from toolkit.util.quantize import get_qtype, quantize

N_IN = 64
RANK = 8


def _quantized_linear(seed):
    from optimum.quanto import freeze

    torch.manual_seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(N_IN, N_IN, bias=True).to(
            device="cuda", dtype=torch.bfloat16
        )
    )
    quantize(model, weights=get_qtype("qfloat8"))
    freeze(model)
    return model[0]


def _block_fn(x, host, pack, lora):
    compiling = torch.compiler.is_compiling()
    if compiling:
        token = torch.ops.mm.fetch_start_after(host, x)
    else:
        token = torch.ops.mm.fetch_start(host)
    flat = torch.ops.mm.fetch_wait(token, host.numel())
    views = block_linear_views(flat, pack)
    if torch.is_grad_enabled():
        x = free_on_backward(x, token)
    out = torch.nn.functional.silu(
        streamed_linear(x, views["proj"], training=True, lora=lora)
    )
    if not ingraph_stream.in_recompute():
        if compiling:
            torch.ops.mm.fetch_free_after(token, out)
        else:
            torch.ops.mm.fetch_free(token)
    return out


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 9),
    "needs CUDA with native FP8 (_scaled_mm)",
)
class TrainingLeavesTests(unittest.TestCase):
    N_BLOCKS = 3

    def setUp(self):
        ingraph_stream.configure_fetch_runtime(depth=2)
        self.addCleanup(lambda: ingraph_stream.configure_fetch_runtime(depth=2))
        self.device = torch.device("cuda")
        self.layers = [_quantized_linear(100 + i) for i in range(self.N_BLOCKS)]
        self.packs = [
            pack_block_host(f"blocks.{i}", [("proj", layer)], repoint=False, pin=True)
            for i, layer in enumerate(self.layers)
        ]
        for pack in self.packs:
            self.assertTrue(pack.pinned)
            self.assertTrue(pack.linears[0].fp8_qualifies)
        torch.manual_seed(5)
        self.loras = [
            LoraEntry(
                a=(torch.randn(RANK, N_IN, device=self.device) * 0.02).requires_grad_(True),
                b=(torch.randn(N_IN, RANK, device=self.device) * 0.02).requires_grad_(True),
                scale=2.0,
            )
            for _ in range(self.N_BLOCKS)
        ]

    def _trunk(self, x):
        context_fn = (
            ingraph_stream.compiled_checkpoint_context
            if torch.compiler.is_compiling()
            else ingraph_stream.checkpoint_recompute_context
        )
        for layer, pack, lora in zip(self.layers, self.packs, self.loras):
            x = torch.utils.checkpoint.checkpoint(
                _block_fn,
                x,
                pack.host_flat,
                pack,
                lora,
                use_reentrant=False,
                context_fn=context_fn,
            )
        return x

    def _reference(self, x):
        # Dequant eager reference with the same LoRA math.
        for layer, lora in zip(self.layers, self.loras):
            w = layer.weight.data.dequantize().to(self.device, torch.bfloat16)
            b = layer.bias.data.to(self.device, torch.bfloat16)
            base = torch.nn.functional.linear(x, w, b)
            lora_out = (x.to(lora.a.dtype) @ lora.a.t() @ lora.b.t()) * lora.scale
            x = torch.nn.functional.silu(base + lora_out.to(base.dtype))
        return x

    def _zero_grads(self, x):
        x.grad = None
        for lora in self.loras:
            lora.a.grad = None
            lora.b.grad = None

    def test_eager_training_parity_and_no_weight_saved(self):
        x = torch.randn(
            4, 16, N_IN, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        saved_shapes = []

        def pack_hook(t):
            saved_shapes.append((tuple(t.shape), t.dtype))
            return t

        self._zero_grads(x)
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
            out = self._trunk(x)
        loss = out.square().mean()
        loss.backward()

        # Parity within FP8 activation-quantization tolerance.
        ref = self._reference(x.detach())
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=8e-2)

        # Gradients flow to input and every LoRA leaf.
        self.assertIsNotNone(x.grad)
        for lora in self.loras:
            self.assertGreater(lora.a.grad.abs().sum().item(), 0.0)
            self.assertGreater(lora.b.grad.abs().sum().item(), 0.0)

        # No full base weight among saved-for-backward: the only 2D
        # (N_IN, N_IN) tensors in the model are the fp8 qdata leaves.
        for shape, dtype in saved_shapes:
            self.assertFalse(
                shape == (N_IN, N_IN) and dtype == torch.float8_e4m3fn,
                f"base fp8 weight saved for backward: {shape} {dtype}",
            )

    def test_lora_grads_match_reference(self):
        x = torch.randn(
            4, 16, N_IN, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        grad_out_seed = torch.Generator(device="cpu").manual_seed(9)

        self._zero_grads(x)
        out = self._trunk(x)
        g = torch.randn(out.shape, generator=grad_out_seed).to(out)
        out.backward(g)
        got = [
            (lora.a.grad.clone(), lora.b.grad.clone()) for lora in self.loras
        ]

        self._zero_grads(x)
        ref_out = self._reference(x)
        ref_out.backward(g.to(ref_out.dtype))
        for (ga, gb), lora in zip(got, self.loras):
            torch.testing.assert_close(ga, lora.a.grad, rtol=8e-2, atol=8e-2)
            torch.testing.assert_close(gb, lora.b.grad, rtol=8e-2, atol=8e-2)

    def test_compiled_training_zero_breaks_and_parity(self):
        from toolkit.memory_management import ingraph_stream_scheduling

        ingraph_stream_scheduling.install_ordering_pass()
        torch._dynamo.reset()
        compiled = torch.compile(self._trunk, fullgraph=True, dynamic=False)
        x = torch.randn(
            4, 16, N_IN, device=self.device, dtype=torch.bfloat16, requires_grad=True
        )
        breaks_before = sum(
            torch._dynamo.utils.counters["graph_break"].values()
        )
        self._zero_grads(x)
        out = compiled(x)
        out.square().mean().backward()
        self.assertEqual(
            sum(torch._dynamo.utils.counters["graph_break"].values()),
            breaks_before,
        )
        ref = self._reference(x.detach())
        torch.testing.assert_close(out, ref, rtol=5e-2, atol=8e-2)
        for lora in self.loras:
            self.assertGreater(lora.a.grad.abs().sum().item(), 0.0)
            self.assertGreater(lora.b.grad.abs().sum().item(), 0.0)

        # Repeat steps: stable, grads keep flowing.
        graphs = int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0))
        for _ in range(3):
            self._zero_grads(x)
            compiled(x).square().mean().backward()
        self.assertEqual(
            int(torch._dynamo.utils.counters["stats"].get("unique_graphs", 0)),
            graphs,
        )


if __name__ == "__main__":
    unittest.main()
