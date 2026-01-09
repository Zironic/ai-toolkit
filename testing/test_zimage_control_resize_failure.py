import torch
from types import SimpleNamespace
import pytest
from toolkit.stable_diffusion_model import StableDiffusion


def test_predict_noise_zimage_raises_on_bad_control_latents_in_strict_mode(monkeypatch):
    sd = SimpleNamespace()

    class FakeUnet:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, control_context=None, **kwargs):
            class R: pass
            R.sample = torch.zeros_like(latents)
            return R

    sd.unet = FakeUnet()

    # latents (noisy) have 64x64 spatial dims
    latents = torch.randn((1, 16, 64, 64))

    # control latents encoded at 60x60 with wrong channel count (16) should be rejected
    zimage_control_latents = torch.randn((1, 16, 60, 60))

    fakecn = lambda *a, **k: torch.zeros((1, 4, a[0].shape[2], a[0].shape[3]))

    func = StableDiffusion._predict_noise_zimage

    with pytest.raises(RuntimeError, match="Illegal `control_context` channel count"):
        func(sd, latents, torch.zeros((1,1,16)), torch.tensor([1.0]), zimage_controlnet=fakecn, zimage_control_images=zimage_control_latents, zimage_conditioning_scale=1.0)
