import pytest
pytest.importorskip("torch")
from toolkit.controlnet_compat import VideoXControlnetWrapper
import toolkit.control_channels as cc


class EchoInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_control = None
        self.last_kwargs = None

    def forward(self, latents, timestep, control_context=None, conditioning_scale=1.0, *args, **kwargs):
        # record and return the control_context so the test can assert its shape
        self.last_control = control_context
        self.last_kwargs = kwargs
        return control_context


def test_wrapper_assembles_16_channel_packed_to_33(monkeypatch):
    inner = EchoInner()
    # Provide adapter identity so wrapper validation passes
    inner.name_or_path = 'zimage_adapter_test'
    inner.control_in_dim = 33
    w = VideoXControlnetWrapper(inner)

    # Create a packed 16-channel latent tensor (B, C=16, H, W)
    lat = torch.zeros((1, 16, 16, 16))
    packed = torch.randn((1, 16, 16, 16))

    out = w.forward(lat, torch.tensor([1.0]), packed)

    # Wrapper returns the inner output (then restored to original `latents` channel count).
    assert isinstance(out, torch.Tensor)

    # The inner received the assembled control_context (either 4D [B,33,H,W] or 5D [B,33,1,H,W])
    assert inner.last_control is not None
    assert isinstance(inner.last_control, torch.Tensor)
    assert inner.last_control.shape[1] == 33
    assert inner.last_control.shape[-2:] == (16, 16)

    # Ensure adapter kwarg was passed
    assert inner.last_kwargs.get('controlnet') is inner

    # The wrapper restores outputs to the original latents channel count (orig_ch=16)
    assert out.ndim >= 4
    assert out.shape[1] == 16
