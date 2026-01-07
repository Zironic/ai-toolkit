import torch
import pytest

from toolkit.controlnet_compat import VideoXControlnetWrapper
import toolkit.control_channels as cc


class EchoInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_control = None

    def forward(self, latents, timestep, control_context=None, conditioning_scale=1.0, *args, **kwargs):
        # record and return the control_context so the test can assert its shape
        self.last_control = control_context
        return control_context


def test_wrapper_assembles_16_channel_packed_to_33(monkeypatch):
    inner = EchoInner()
    # Provide adapter identity so wrapper validation passes
    inner.name_or_path = 'zimage_adapter_test'
    inner.control_in_dim = 33
    w = VideoXControlnetWrapper(inner)

    # Guard: ensure adapt_control_images is not called by wrapper
    def boom(*a, **k):
        raise RuntimeError("adapt_control_images should not be called by wrapper")
    monkeypatch.setattr(cc, 'adapt_control_images', boom)

    # Create a packed 16-channel latent tensor (B, C=16, H, W)
    lat = torch.zeros((1, 4, 16, 16))
    packed = torch.randn((1, 16, 16, 16))

    out = w.forward(lat, torch.tensor([1.0]), packed)

    # Wrapper returns the inner output (then restored to original `latents` channel count).
    assert isinstance(out, torch.Tensor)

    # The inner received the assembled 5D control_context [B, 33, 1, H, W]
    assert inner.last_control is not None
    assert isinstance(inner.last_control, torch.Tensor)
    assert inner.last_control.ndim == 5
    assert inner.last_control.shape[1] == 33
    assert inner.last_control.shape[2] == 1
    assert inner.last_control.shape[-2:] == (16, 16)

    # The wrapper restores outputs to the original latents channel count (orig_ch=4)
    assert out.ndim >= 4
    assert out.shape[1] == 4
