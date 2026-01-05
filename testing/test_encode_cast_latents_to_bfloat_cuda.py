import torch
import types
import pytest
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA to test bfloat16 on GPU")
def test_latents_cast_to_bfloat16_on_cuda():
    z = ZImageModel.__new__(ZImageModel)
    # create a VAE param with bfloat16 on cuda
    param = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16, device='cuda'))
    class DummyVAE:
        def __init__(self):
            self.config = {"block_out_channels": [1, 2]}
            self._param = param
        def parameters(self):
            yield self._param
        def eval(self):
            pass
        def requires_grad_(self, v):
            pass
        def encode(self, batch):
            # return float32 latents initially
            b,c,h,w = batch.shape
            return torch.randn(b,16, max(1,h//16), max(1,w//16), device=batch.device, dtype=torch.float32)
    z.vae = DummyVAE()
    z.transformer = types.SimpleNamespace(control_in_dim=33)
    imgs = [torch.randint(0,255,(3,64,64), dtype=torch.uint8).to('cuda')]
    latents = z.encode_control_images(imgs, tile=False)
    assert latents.dtype == torch.bfloat16
