import pytest
import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class FakeInner:
    def __init__(self):
        self.name_or_path = "fake/adapter"
        self.control_in_dim = 4

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0):
        # should not be reached in this test
        return None


def test_wrapper_rejects_raw_pixel_images():
    inner = FakeInner()
    wrapper = VideoXControlnetWrapper(inner)

    latents = torch.randn((1, 33, 1, 32, 32))  # small latent tensor
    # raw pixel images: [B, C, F, H, W]
    pixels = torch.randn((1, 3, 1, 512, 512))

    with pytest.raises(RuntimeError) as exc:
        wrapper(latents, torch.tensor([1.0]), pixels, conditioning_scale=1.0)

    msg = str(exc.value)
    assert "received raw pixel images" or "VideoXControlnetWrapper received raw pixel images" in msg or "VideoXControlnetWrapper" in msg
    assert "fake/adapter" in msg
    assert "control_in_dim=4" in msg or "control_in_dim" in msg
