"""A leaf's residency must not change what the trunk computes.

The in-graph trunk feeds `streamed_linear_tensors` either a view into a fetched
flat buffer (streamed leaf) or the Linear's own Parameter (resident leaf). Those
are different tensors -- different storages, different base pointers, different
alignment -- carrying the same bytes. If the fp8 kernel is at all sensitive to
that, every residency decision silently perturbs training, and the planner is
free to change residency run to run.

This pins the primitive down on the real hardware path. It is the unit-level
statement of what the GPU smoke shows end to end: shifting block leaves between
streamed and resident (84 resident vs 56 resident) leaves the step-0 loss
bit-identical at 6.5984.
"""

import unittest

import torch

from toolkit.quantization.fp8_linear import bind_rowwise_fp8, bind_storage_operation
from toolkit.quantization.storage import linear_storage_binding

from toolkit.memory_management.ingraph_stream import (
    resident_linear_tensors,
    streamed_linear_tensors,
)


def _flat_view_of(weight: torch.Tensor, offset: int = 256) -> torch.Tensor:
    """The same bytes, but as a view into a larger buffer -- what a pack does."""
    nbytes = weight.numel() * weight.element_size()
    flat = torch.empty(nbytes + offset, dtype=torch.uint8, device=weight.device)
    view = flat[offset : offset + nbytes].view(weight.dtype).reshape(weight.shape)
    view.copy_(weight)
    return view


@unittest.skipUnless(torch.cuda.is_available(), "fp8 _scaled_mm needs CUDA")
class Fp8LeafPlacementIsBitwiseInvariantTests(unittest.TestCase):
    def _weights(self, out_f, in_f):
        device = torch.device("cuda:0")
        torch.manual_seed(0)
        qdata = (torch.randn(out_f, in_f, device=device) / 8).to(torch.float8_e4m3fn)
        scale = (torch.rand(out_f, device=device).abs() + 0.5).to(torch.float32)
        return qdata, scale

    def test_streamed_view_and_resident_tensor_agree_bitwise(self):
        # Krea2 block shapes at the smoke's training resolution.
        out_f, in_f, tokens = 1536, 6144, 1280
        qdata, scale = self._weights(out_f, in_f)
        x = torch.randn(tokens, in_f, device=qdata.device, dtype=torch.bfloat16)

        view = _flat_view_of(qdata)
        self.assertTrue(torch.equal(view.view(torch.uint8), qdata.view(torch.uint8)))
        self.assertNotEqual(view.data_ptr(), qdata.data_ptr())

        resident = streamed_linear_tensors(
            x,
            qdata,
            None,
            scale,
            operation=bind_rowwise_fp8(
                qdata, scale, device=qdata.device, has_bias=False
            ),
            training=True,
        )
        streamed = streamed_linear_tensors(
            x,
            view,
            None,
            scale,
            operation=bind_rowwise_fp8(
                view, scale, device=view.device, has_bias=False
            ),
            training=True,
        )
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(resident, streamed),
            "fp8 _scaled_mm result depends on weight placement; every residency "
            "change would perturb training",
        )

    def test_mixed_leaf_sources_match_all_streamed(self):
        """The trunk splices resident and streamed leaves back into one block
        call. Chaining them must equal sourcing every leaf from the flat -- any
        drift compounds across the 28 blocks of the real trunk."""
        dim, tokens = 1024, 256  # square so the leaves chain
        x = torch.randn(tokens, dim, device="cuda:0", dtype=torch.bfloat16)

        # Distinct weights per leaf, so a swapped source cannot pass by accident.
        leaves = []
        for i in range(3):
            torch.manual_seed(i)
            qdata = (torch.randn(dim, dim, device="cuda:0") / 8).to(torch.float8_e4m3fn)
            scale = (torch.rand(dim, device="cuda:0").abs() + 0.5).to(torch.float32)
            leaves.append((qdata, scale))

        all_streamed = [(_flat_view_of(q), None, s) for q, s in leaves]
        # Leaf 1 resident, leaves 0 and 2 streamed: a partially-resident block.
        mixed = [all_streamed[0], (leaves[1][0], None, leaves[1][1]), all_streamed[2]]
        self.assertIsNot(mixed[1][0], all_streamed[1][0])

        def run(triples):
            out = x
            for weight, bias, scale in triples:
                out = streamed_linear_tensors(
                    out,
                    weight,
                    bias,
                    scale,
                    operation=bind_rowwise_fp8(
                        weight, scale, device=weight.device, has_bias=False
                    ),
                    training=True,
                )
            return out

        torch.cuda.synchronize()
        self.assertTrue(torch.equal(run(all_streamed), run(mixed)))


class ResidentLinearTensorsTests(unittest.TestCase):
    def test_float_linear_round_trips_through_the_streamed_math(self):
        linear = torch.nn.Linear(16, 8, bias=True).eval()
        x = torch.randn(4, 16)
        tensors = resident_linear_tensors(linear)
        operation = bind_storage_operation(
            tensors,
            device="cpu",
            weight_leaf_count=1,
            execution_key=linear_storage_binding(
                linear.weight,
                linear.bias,
            ).execution_key,
        )
        weight, bias, scale = operation.functional_components(tensors)
        self.assertIsNone(scale)
        torch.testing.assert_close(
            streamed_linear_tensors(
                x, weight, bias, scale, operation=operation
            ),
            linear(x),
        )


if __name__ == "__main__":
    unittest.main()
