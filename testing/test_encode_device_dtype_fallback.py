import torch
import types
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


class DummyVAEWithBFloatOnCPU:
    def __init__(self):
        # simulate a VAE with a parameter on CPU bfloat16 (unsupported for many ops)
        self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.config = {"block_out_channels": [1, 2]}

    def parameters(self):
        yield self._param

    def buffers(self):
        if False:
            yield None

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return None

    def requires_grad_(self, v):
        pass

    def encode(self, batch):
        # ensure batch dtype is float32 to indicate fallback was used
        assert batch.dtype == torch.float32, f"Expected float32 batch but got {batch.dtype}"
        b, c, h, w = batch.shape
        return torch.randn(b, 16, max(1, h//16), max(1, w//16))


def test_encode_falls_back_to_float32_when_bfloat16_cpu_param():
    z = ZImageModel.__new__(ZImageModel)
    z.vae = DummyVAEWithBFloatOnCPU()
    # transformer used only for control_in_dim in assembly
    z.transformer = types.SimpleNamespace(control_in_dim=33)
    imgs = [torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8) for _ in range(2)]
    latents = z.encode_control_images(imgs, tile=False)
    assert isinstance(latents, torch.Tensor)
    assert latents.shape[0] == 2
    # When we fell back from bfloat16 CPU inputs/weights, we expect the final latents to be float32
    assert latents.dtype == torch.float32
