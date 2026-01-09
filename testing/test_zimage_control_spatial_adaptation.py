import pytest
try:
    import torch
except Exception:
    pytest.skip("Skipping: PyTorch import failed in this environment", allow_module_level=True)

from types import SimpleNamespace
from toolkit.stable_diffusion_model import StableDiffusion


def test_predict_noise_zimage_rejects_raw_control_images_in_strict_mode():
    # Strict mode does not auto-encode raw pixel images; pre-encoded latents are required.
    sd = SimpleNamespace()

    # Minimal unet placeholder
    class FakeUnet:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
            class R: pass
            R.sample = torch.zeros_like(latents)
            return R

    sd.unet = FakeUnet()

    latents = torch.randn((1, 16, 48, 48))
    # Raw pixel images with 3 channels should be rejected by strict logic
    zimage_control_images = torch.randn((1, 3, 1, 512, 512))

    text_embeddings = torch.zeros((1, 1, 16))

    func = StableDiffusion._predict_noise_zimage
    with pytest.raises(RuntimeError, match="Illegal `control_context` channel count"):
        func(sd, latents, text_embeddings, torch.tensor([1.0]), zimage_controlnet=SimpleNamespace(), zimage_control_images=zimage_control_images, zimage_conditioning_scale=1.0)


def test_predict_noise_zimage_rejects_wrong_channel_control_latents_in_strict_mode():
    """Strict mode rejects pre-encoded control latents whose channel count is not 33."""
    sd = SimpleNamespace()

    class FakeUnet:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
            class R: pass
            R.sample = torch.zeros_like(latents)
            return R

    sd.unet = FakeUnet()

    latents = torch.randn((1, 16, 64, 64))
    # Encoded latents with 16 channels should be rejected under strict rules
    zimage_control_latents = torch.randn((1, 16, 60, 60))

    func = StableDiffusion._predict_noise_zimage
    with pytest.raises(RuntimeError, match="Illegal `control_context` channel count"):
        func(sd, latents, torch.zeros((1,1,16)), torch.tensor([1.0]), zimage_controlnet=SimpleNamespace(), zimage_control_images=zimage_control_latents, zimage_conditioning_scale=1.0)

