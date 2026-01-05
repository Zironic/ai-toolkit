import torch
from types import SimpleNamespace
from toolkit.stable_diffusion_model import StableDiffusion
import pytest


def test_auto_encode_requires_encoder():
    sd = SimpleNamespace()
    sd.unet = lambda *a, **k: None
    latents = torch.randn((1, 16, 48, 48))
    text_embeddings = torch.zeros((1, 1, 16))
    zimage_control_images = torch.randn((1, 3, 1, 512, 512))

    # No sd.encode_control_images present
    with pytest.raises(RuntimeError):
        StableDiffusion._predict_noise_zimage(sd, latents, text_embeddings, torch.tensor([1.0]), zimage_controlnet=lambda *a, **k: None, zimage_control_images=zimage_control_images)
