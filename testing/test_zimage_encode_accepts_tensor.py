import torch
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
import types

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
        # Simulate encoding: return batch downsampled by 16 in spatial dims and with 16 channels
        b, c, h, w = batch.shape
        return torch.randn(b, 16, max(1, h//16), max(1, w//16))


def test_encode_control_images_accepts_tensors(monkeypatch):
    # Create instance without running full constructor (avoid heavy init)
    z = ZImageModel.__new__(ZImageModel)
    z.vae = DummyVAE()
    # transformer used only for control_in_dim in assembly
    z.transformer = types.SimpleNamespace(control_in_dim=33)
    imgs = [torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8), torch.randint(0,255,(3,512,512), dtype=torch.uint8)]
    latents = z.encode_control_images(imgs, tile=False)
    assert isinstance(latents, torch.Tensor)
    assert latents.shape[0] == 2

