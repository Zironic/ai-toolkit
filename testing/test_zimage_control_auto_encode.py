import torch
from types import SimpleNamespace
from toolkit.stable_diffusion_model import StableDiffusion


def test_predict_noise_zimage_auto_encodes_control_images():
    sd = SimpleNamespace()

    # Fake unet returns .sample
    class FakeUnet:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
            class R: pass
            R.sample = torch.zeros_like(latents)
            # Ensure control_context is latents-shaped
            if isinstance(control_context, torch.Tensor):
                h, w = int(control_context.shape[-2]), int(control_context.shape[-1])
                assert h == 48 and w == 48
            return R

    sd.unet = FakeUnet()

    # fake controlnet that asserts it receives latents with spatial dims 48x48
    class FakeControlNet:
        def __call__(self, sample, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
            if isinstance(control_context, torch.Tensor):
                h, w = int(control_context.shape[-2]), int(control_context.shape[-1])
                assert h == 48 and w == 48
            return torch.zeros((1, 4, 48, 48))

    fakecn = FakeControlNet()

    # encode_control_images should be called and return latents sized to 48x48
    called = {'flag': False}

    def fake_encode(imgs, tile=False, tile_size=None, overlap=None):
        called['flag'] = True
        # Return a batched tensor simulating VAE latents
        return torch.zeros((len(imgs), 4, 48, 48))

    sd.encode_control_images = fake_encode

    # minimal text embeddings
    text_embeddings = torch.zeros((1, 1, 16))

    latents = torch.randn((1, 16, 48, 48))
    zimage_control_images = torch.randn((1, 3, 1, 512, 512))

    # Attach model_config with control tiling flags off
    sd.model_config = SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)

    # Bind method
    func = StableDiffusion._predict_noise_zimage
    out = func(sd, latents, text_embeddings, torch.tensor([1.0]), zimage_controlnet=fakecn, zimage_control_images=zimage_control_images, zimage_conditioning_scale=1.0)

    assert called['flag'] is True
    assert out is not None
