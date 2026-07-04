import unittest

import torch

from extensions_built_in.diffusion_models.krea2.src.mmdit import (
    SingleMMDiTConfig,
    SingleStreamDiT,
)
from toolkit.memory_management.ingraph_stream import pack_block_host


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

        torch.testing.assert_close(functional, eager, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
