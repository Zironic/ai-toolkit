import unittest

import torch

from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from toolkit.memory_management.ingraph_stream import (
    BlockLeafPlan,
    block_tensor_views,
    pack_block_host,
)


class InGraphSamplerTests(unittest.TestCase):
    def _model(self):
        torch.manual_seed(123)
        return SingleStreamDiT(
            SingleMMDiTConfig(
                features=32,
                tdim=16,
                txtdim=32,
                heads=4,
                multiplier=1,
                layers=1,
                patch=1,
                channels=4,
                txtheads=4,
                txtkvheads=4,
            )
        ).eval()

    def test_functional_block_path_matches_module_linears(self):
        model = self._model()
        block = model.blocks[0]
        pack = pack_block_host(
            "blocks.0",
            model._block_linear_entries(block),
            repoint=False,
            pin=False,
        )
        leaves = model._nest_block_leaves(pack.host_flat, pack)

        x = torch.randn(1, 5, 32)
        vec = torch.randn(1, 192)
        pos = torch.zeros(1, 5, 3)
        freqs = model.posemb(pos)
        mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)

        with torch.no_grad():
            eager = block(x, vec, freqs, mask)
            functional = block(x, vec, freqs, mask, leaves=leaves)
            positional = block.forward_streamed(
                x,
                vec,
                freqs,
                mask,
                block_tensor_views(pack.host_flat, pack),
                pack.fp8_flags,
            )

        torch.testing.assert_close(functional, eager, rtol=0, atol=0)
        torch.testing.assert_close(positional, eager, rtol=0, atol=0)

    def test_partial_ingraph_sample_factory_binds_instance(self):
        model = self._model()
        block = model.blocks[0]
        pack = pack_block_host(
            "blocks.0",
            model._block_linear_entries(block),
            repoint=False,
            pin=False,
        )
        model._ingraph_sampling_plans = {
            0: BlockLeafPlan(
                block_key="blocks.0",
                pack=pack,
                sources=tuple((True, i) for i in range(len(pack.fp8_flags))),
                resident_args=(),
                fp8_flags=pack.fp8_flags,
            )
        }
        model._ingraph_sampling_packs = {0: pack}

        fn = model._make_ingraph_sample_block_fn(0)

        self.assertTrue(callable(fn))

    def test_positional_block_path_compiles_fullgraph(self):
        model = self._model()
        block = model.blocks[0]
        pack = pack_block_host(
            "blocks.0",
            model._block_linear_entries(block),
            repoint=False,
            pin=False,
        )
        leaf_args = block_tensor_views(pack.host_flat, pack)

        x = torch.randn(1, 5, 32)
        vec = torch.randn(1, 192)
        pos = torch.zeros(1, 5, 3)
        freqs = model.posemb(pos)
        mask = torch.ones(1, 1, 5, 5, dtype=torch.bool)

        def fn(x, vec, freqs, mask):
            return block.forward_streamed(
                x,
                vec,
                freqs,
                mask,
                leaf_args,
                pack.fp8_flags,
            )

        torch._dynamo.reset()
        compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="eager")
        with torch.no_grad():
            eager = fn(x, vec, freqs, mask)
            got = compiled(x, vec, freqs, mask)
        torch.testing.assert_close(got, eager, rtol=0, atol=0)
        self.assertEqual(sum(torch._dynamo.utils.counters["graph_break"].values()), 0)


if __name__ == "__main__":
    unittest.main()
