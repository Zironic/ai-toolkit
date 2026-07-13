"""Phase 4a S1: fetch ops under checkpoint + autograd (INGRAPH_PHASE4A plan).

Proves the training-mode free discipline: `free_on_backward` anchors each
ticket's free event to the consuming block's backward, while the no-grad
first pass of non-reentrant checkpoint keeps the sampling-style forward
free -- so the depth-K ring neither deadlocks in forward nor recycles a
buffer under backward kernels that still read it.

Weights are random signed permutation matrices (exact in fp32 matmul), so
forward AND gradient chains are bitwise-comparable -- cross-stream buffer
corruption cannot hide inside a tolerance.
"""

import unittest

import torch

from toolkit.memory_management import ingraph_stream
from toolkit.memory_management.ingraph_stream import free_on_backward

K = 32  # matrix side; K*K*4 bytes per block pack


def _signed_permutation(k: int, generator: torch.Generator) -> torch.Tensor:
    perm = torch.randperm(k, generator=generator)
    signs = torch.randint(0, 2, (k,), generator=generator) * 2 - 1
    m = torch.zeros(k, k)
    m[torch.arange(k), perm] = signs.float()
    return m


def _pinned_host(w: torch.Tensor) -> torch.Tensor:
    flat = torch.empty(w.numel() * 4, dtype=torch.uint8, pin_memory=True)
    flat.view(torch.float32).copy_(w.reshape(-1))
    return flat


def _streamed_block(x: torch.Tensor, host: torch.Tensor) -> torch.Tensor:
    # Guarded (_after) op variants exist for compiled graphs: their declared
    # guard mutation gives ordering/DCE protection and is functionalized away.
    # Grad-mode eager must use the plain ops -- the guard's version bump trips
    # autograd's saved-tensor version checks (program order suffices there).
    compiling = torch.compiler.is_compiling()
    if compiling:
        token = torch.ops.mm.fetch_start_after(host, x)
    else:
        token = torch.ops.mm.fetch_start(host)
    flat = torch.ops.mm.fetch_wait(token, host.numel())
    w = flat.view(torch.float32).reshape(K, K)
    if torch.is_grad_enabled():
        # Saved-token swap: checkpoint replaces this token with the recompute
        # generation's, so backward frees the ticket it actually read.
        x = free_on_backward(x, token)
    out = x @ w
    if not ingraph_stream.in_recompute():
        # First-pass generation: its views are dropped by the checkpoint
        # hooks, so forward free is safe and keeps the ring draining.
        if compiling:
            torch.ops.mm.fetch_free_after(token, out)
        else:
            torch.ops.mm.fetch_free(token)
    return out


def _trunk(x: torch.Tensor, hosts) -> torch.Tensor:
    if torch.is_grad_enabled() and x.requires_grad:
        # Select at the traced call site: the HOP invokes context_fn outside
        # the Dynamo frame, where is_compiling() would read False.
        context_fn = (
            ingraph_stream.compiled_checkpoint_context
            if torch.compiler.is_compiling()
            else ingraph_stream.checkpoint_recompute_context
        )
        for host in hosts:
            x = torch.utils.checkpoint.checkpoint(
                _streamed_block,
                x,
                host,
                use_reentrant=False,
                context_fn=context_fn,
            )
        return x
    for host in hosts:
        x = _streamed_block(x, host)
    return x


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class FreeOnBackwardTests(unittest.TestCase):
    N_BLOCKS = 6  # > depth so the ring must recycle in fwd AND bwd

    def setUp(self):
        gen = torch.Generator().manual_seed(23)
        self.weights = [
            _signed_permutation(K, gen) for _ in range(self.N_BLOCKS)
        ]
        self.hosts = [_pinned_host(w) for w in self.weights]
        self.device = torch.device("cuda")
        ingraph_stream.configure_fetch_runtime(depth=2)
        ingraph_stream.reset_fetch_stats()
        self.addCleanup(lambda: ingraph_stream.configure_fetch_runtime(depth=2))

    def _reference(self, x, grad_out):
        w_all = torch.eye(K)
        for w in self.weights:
            w_all = w_all @ w
        w_all = w_all.to(self.device)
        return x @ w_all, grad_out @ w_all.t()

    def test_eager_checkpoint_fwd_bwd_exact_and_refetches(self):
        x = torch.randn(
            8, K, device=self.device, requires_grad=True
        )
        stats0 = int(ingraph_stream.fetch_stats()["fetches"])
        out = _trunk(x, self.hosts)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        ref_out, ref_grad = self._reference(x.detach(), grad_out)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, ref_grad, rtol=0, atol=0)
        # Backward re-fetch falls out of checkpoint recompute: one fetch per
        # block in forward plus one per block in recompute.
        fetches = int(ingraph_stream.fetch_stats()["fetches"]) - stats0
        self.assertEqual(fetches, 2 * self.N_BLOCKS)

    def test_compiled_checkpoint_fwd_bwd_zero_breaks(self):
        # Requires the post-grad ordering pass: without it, Inductor hoists
        # backward re-fetches (data-dep only on saved boundary activations)
        # above frees and trips the depth guard. The pass rewrites backward
        # fetch_start_after nodes to fetch_start_gated, gated on the previous
        # free's token output -- real dataflow no scheduling stage can drop.
        from toolkit.memory_management import ingraph_stream_scheduling

        ingraph_stream_scheduling.install_ordering_pass()
        torch._dynamo.reset()
        compiled = torch.compile(
            lambda x: _trunk(x, self.hosts), fullgraph=True, dynamic=False
        )
        x = torch.randn(8, K, device=self.device, requires_grad=True)
        breaks_before = sum(
            torch._dynamo.utils.counters["graph_break"].values()
        )
        out = compiled(x)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        self.assertEqual(
            sum(torch._dynamo.utils.counters["graph_break"].values()),
            breaks_before,
        )
        ref_out, ref_grad = self._reference(x.detach(), grad_out)
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, ref_grad, rtol=0, atol=0)

    def test_backward_buffer_reuse_hammer(self):
        # Many passes at depth 2: any premature recycle (a fetch overwriting a
        # buffer a backward still reads) breaks bitwise equality of grads.
        x = torch.randn(8, K, device=self.device, requires_grad=True)
        grad_out = torch.randn(8, K, device=self.device)
        _, ref_grad = self._reference(x.detach(), grad_out)
        for _ in range(25):
            if x.grad is not None:
                x.grad = None
            out = _trunk(x, self.hosts)
            out.backward(grad_out)
            torch.testing.assert_close(x.grad, ref_grad, rtol=0, atol=0)

    def test_no_grad_path_unchanged(self):
        # Sampling-style: forward free, no autograd nodes, ring drains.
        x = torch.randn(8, K, device=self.device)
        with torch.no_grad():
            out = _trunk(x, self.hosts)
        ref_out, _ = self._reference(x, torch.zeros_like(x))
        torch.testing.assert_close(out, ref_out, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

import pytest

pytestmark = pytest.mark.process_isolated
