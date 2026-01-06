import pytest
import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class FakeInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # create a conv_in like object with weight expecting 4 in-ch
        self.conv_in = type('C', (), {})()
        self.conv_in.weight = torch.zeros((8, 4, 3, 3))
        self.called_with = None

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        # record the control_context shape
        self.called_with = control_context
        # return a dummy tensor
        return torch.zeros((latents.shape[0], 4, latents.shape[2], latents.shape[3]))


def test_wrapper_rejects_raw_pixel_images_when_adapter_expects_latents():
    inner = FakeInner()
    wrapper = VideoXControlnetWrapper(inner)

    # control_context looks like a raw pixel image [B,C,H,W]
    ctrl = torch.zeros((1, 3, 512, 512))
    lat = torch.zeros((1, 16, 64, 64))

    with pytest.raises(RuntimeError) as exc:
        wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    msg = str(exc.value)
    assert 'raw pixel' in msg or 'received raw pixel images' in msg


def test_wrapper_rejects_small_latents_with_incompatible_channels():
    inner = FakeInner()
    wrapper = VideoXControlnetWrapper(inner)

    # Small spatial dims, likely latents, but channels don't match expected (3 vs 4)
    ctrl = torch.zeros((1, 3, 8, 8))
    lat = torch.zeros((1, 16, 64, 64))

    with pytest.raises(RuntimeError) as exc:
        wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    msg = str(exc.value)
    assert 'do not match expected_in' in msg or 'will not adapt control images' in msg
