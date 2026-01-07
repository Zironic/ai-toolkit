import torch
import pytest

import toolkit.control_channels as cc
from toolkit.controlnet_compat import VideoXControlnetWrapper


class GoodInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, latents, timestep, control_context=None, conditioning_scale=1.0, *args, **kwargs):
        # simple echo behavior for testing
        return latents * 0.0


def test_wrapper_does_not_call_adapt_control_images(monkeypatch):
    inner = GoodInner()
    w = VideoXControlnetWrapper(inner)

    # Monkeypatch adapt_control_images to raise if called
    def boom(*a, **k):
        raise RuntimeError("adapt_control_images should not be called by wrapper")
    monkeypatch.setattr(cc, 'adapt_control_images', boom)

    lat = torch.zeros((1, 4, 16, 16))
    # Create a pre-assembled 33-channel control_context and ensure wrapper forwards
    ctrl33 = torch.zeros((1, 33, 16, 16))
    out = w.forward(lat, torch.tensor([1.0]), ctrl33)
    assert isinstance(out, torch.Tensor)


def test_wrapper_rejects_4_channel_control_context():
    inner = GoodInner()
    w = VideoXControlnetWrapper(inner)
    lat = torch.zeros((1, 4, 16, 16))
    ctrl4 = torch.zeros((1, 4, 64, 64))
    with pytest.raises(RuntimeError) as excinfo:
        w.forward(lat, torch.tensor([1.0]), ctrl4)
    assert 'Received 4-channel base latents' in str(excinfo.value)
