import types
import torch
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


class DummyVAE:
    def __init__(self):
        self.config = {"block_out_channels": [16, 32]}
        self._param = torch.nn.Parameter(torch.zeros(1))
        self.eval_called = False
        self.grad_flag = True

    def parameters(self):
        yield self._param

    def to(self, *args, **kwargs):
        # pretend to move device; record call
        self.moved = True
        return self

    def eval(self):
        self.eval_called = True

    def requires_grad_(self, v):
        self.grad_flag = v

    def encode(self, batch):
        # ensure dims are divisible by VAE scale
        _, _, h, w = batch.shape
        assert h % 2 == 0 and w % 2 == 0, f"Expected divisible by 2, got {h}x{w}"
        # return simple latents
        b = batch.shape[0]
        return torch.randn(b, 16, max(1, h//16), max(1, w//16))


def test_control_encode_sets_eval_and_resizes():
    z = ZImageModel.__new__(ZImageModel)
    z.vae = DummyVAE()
    z.transformer = types.SimpleNamespace(control_in_dim=33)
    imgs = [torch.randint(0, 255, (3, 65, 65), dtype=torch.uint8)]
    latents = z.encode_control_images(imgs, tile=False)
    assert isinstance(latents, torch.Tensor)
    # ensure VAE flags updated
    assert z.vae.eval_called
    assert z.vae.grad_flag is False
    assert latents.shape[0] == 1
