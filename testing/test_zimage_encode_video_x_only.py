import pytest
import torch
import types
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel

class DummyVAE:
    def __init__(self):
        self.config = {"block_out_channels": [1, 2]}
    def to(self, *args, **kwargs):
        return self
    def eval(self):
        pass
    def requires_grad_(self, v):
        pass
    def encode(self, batch):
        b, c, h, w = batch.shape
        return torch.randn(b, 16, max(1, h//16), max(1, w//16))


def test_encode_rejects_pil_or_numpy_inputs():
    z = ZImageModel.__new__(ZImageModel)
    z.vae = DummyVAE()
    # non-tensor input should raise
    with pytest.raises(RuntimeError):
        z.encode_control_images([object()])


def test_encode_rejects_tiling_flag():
    z = ZImageModel.__new__(ZImageModel)
    z.vae = DummyVAE()
    with pytest.raises(NotImplementedError):
        z.encode_control_images([torch.randint(0,255,(3,512,512), dtype=torch.uint8)], tile=True)
