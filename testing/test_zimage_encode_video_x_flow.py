import torch
import types
import pytest
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel

class DummyVAEOut:
    def __init__(self, lat):
        self.latents = lat

class DummyVAEOutDist:
    def __init__(self, lat):
        class Dist:
            def __init__(self, lat):
                self._lat = lat
            def mode(self):
                return self._lat
        self.latent_dist = Dist(lat)

class DummyVAE:
    def __init__(self):
        self.config = types.SimpleNamespace(block_out_channels=[1,2], shift_factor=0.1, scaling_factor=0.5)
    def to(self, *args, **kwargs):
        return self
    def eval(self):
        pass
    def requires_grad_(self, v):
        pass
    def encode(self, batch):
        # return a ModelOutput-like object with latents attr
        b, c, h, w = batch.shape
        lat = torch.randn(b, 16, max(1, h//16), max(1, w//16))
        return DummyVAEOut(lat)

class DummyProc:
    def preprocess(self, images, height=None, width=None):
        # Return pixel_values tensor shaped [B,C,H,W]
        if isinstance(images, list):
            imgs = images
        else:
            imgs = [images]
        out = torch.randn(len(imgs), 3, height or 512, width or 512)
        return {'pixel_values': out}


def test_video_x_flow_with_modeloutput_like():
    z = ZImageModel.__new__(ZImageModel)
    z.vae = DummyVAE()
    z.image_processor = DummyProc()
    lat = z.encode_control_images([object(), object()], height=512, width=512)
    assert isinstance(lat, torch.Tensor)


def test_video_x_flow_with_dist_like():
    class VAE2(DummyVAE):
        def encode(self, batch):
            b, c, h, w = batch.shape
            lat = torch.randn(b, 16, max(1, h//16), max(1, w//16))
            return DummyVAEOutDist(lat)
    z = ZImageModel.__new__(ZImageModel)
    z.vae = VAE2()
    z.image_processor = DummyProc()
    lat = z.encode_control_images([object()], height=512, width=512)
    assert isinstance(lat, torch.Tensor)
    # check scaling applied roughly
    # cannot exactly predict, but shape must match
    assert lat.ndim == 4
