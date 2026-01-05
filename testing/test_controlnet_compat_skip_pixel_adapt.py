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


def test_wrapper_skips_pixel_image_adaptation():
    inner = FakeInner()
    wrapper = VideoXControlnetWrapper(inner)

    # control_context looks like a raw pixel image [B,C,H,W]
    ctrl = torch.zeros((1, 3, 512, 512))
    lat = torch.zeros((1, 16, 64, 64))
    out = wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    # inner should have been called with original 3-channel control image (not padded to 4)
    assert isinstance(inner.called_with, torch.Tensor)
    assert inner.called_with.shape[1] == 3


def test_wrapper_adapts_when_not_pixel_images():
    inner = FakeInner()
    wrapper = VideoXControlnetWrapper(inner)

    # Simulate a control_context that is small spatial dims (likely latent)
    # Under the new policy we no longer adapt based on inner.conv_in; instead
    # we defer deterministic assembly/adaptation to the Z-Image pipeline. Thus
    # we expect the inner to receive the original channels when no explicit
    # `control_in_dim` is provided on the adapter.
    ctrl = torch.zeros((1, 3, 8, 8))
    lat = torch.zeros((1, 16, 64, 64))
    out = wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    assert isinstance(inner.called_with, torch.Tensor)
    assert inner.called_with.shape[1] == 3
