import torch
from types import SimpleNamespace
import pytest
from toolkit.stable_diffusion_model import StableDiffusion


def test_predict_noise_zimage_raises_on_resize_failure(monkeypatch):
    sd = SimpleNamespace()

    class FakeUnet:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
            class R: pass
            R.sample = torch.zeros_like(latents)
            return R

    sd.unet = FakeUnet()

    # latents (noisy) have 64x64 spatial dims
    latents = torch.randn((1, 16, 64, 64))

    # control latents encoded at 60x60 (mismatch)
    zimage_control_latents = torch.randn((1, 4, 60, 60))

    class FakeControlNet:
        def __call__(self, sample, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
            return torch.zeros((1, 4, sample.shape[2], sample.shape[3]))

    fakecn = FakeControlNet()

    text_embeddings = torch.zeros((1, 1, 16))

    # Monkeypatch interpolate to raise an error to simulate a failure scenario
    import torch.nn.functional as F

    def bad_interpolate(*args, **kwargs):
        raise ValueError("interpolate failure simulated")

    monkeypatch.setattr('torch.nn.functional.interpolate', bad_interpolate)

    func = StableDiffusion._predict_noise_zimage

    with pytest.raises(RuntimeError) as exc:
        func(sd, latents, text_embeddings, torch.tensor([1.0]), zimage_controlnet=fakecn, zimage_control_images=zimage_control_latents, zimage_conditioning_scale=1.0)

    assert 'Z-Image control latent resize failed' in str(exc.value)
