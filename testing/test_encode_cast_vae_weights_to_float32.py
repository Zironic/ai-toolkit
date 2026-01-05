import torch
import types
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


class DummyVAEWithBFloatOnCPU:
    def __init__(self):
        # parameter initially bfloat16 on CPU
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.config = {"block_out_channels": [1, 2]}
        self.cast_done = False

    def parameters(self):
        yield self._param

    def to(self, *args, **kwargs):
        # Simulate casting by converting param dtype (record cast)
        try:
            if 'dtype' in kwargs and kwargs['dtype'] == torch.float32:
                self._param.data = self._param.data.to(torch.float32)
                self.cast_done = True
        except Exception:
            pass
        return self

    def eval(self):
        pass

    def requires_grad_(self, v):
        pass

    def encode(self, batch):
        # Assert param dtype is float32 (we expect prior casting)
        assert self._param.dtype == torch.float32, f"VAE weights not casted to float32: {self._param.dtype}"
        b, c, h, w = batch.shape
        return torch.randn(b, 16, max(1, h//16), max(1, w//16))


def test_vae_weights_cast_to_float32_on_fallback():
    z = ZImageModel.__new__(ZImageModel)
    z.vae = DummyVAEWithBFloatOnCPU()
    z.transformer = types.SimpleNamespace(control_in_dim=33)
    imgs = [torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8) for _ in range(2)]
    # Run encode; should cast VAE weights to float32 and succeed
    latents = z.encode_control_images(imgs, tile=False)
    assert isinstance(latents, torch.Tensor)
    assert latents.shape[0] == 2
    assert z.vae.cast_done
