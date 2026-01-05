import torch
import pytest
from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyInnerConv4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # a conv with in_channels=4 somewhere in the module tree
        self.some_conv = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        # use a forward that asserts control_context channels == 4
    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        if isinstance(control_context, torch.Tensor):
            assert control_context.shape[1] == 4, f"expected 4 channels, got {control_context.shape[1]}"
        elif isinstance(control_context, (list, tuple)):
            for c in control_context:
                assert c.shape[1] == 4
        return torch.zeros_like(latents)


class DummyInnerConvMixed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # explicit conv_in with 3 channels
        self.conv_in = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # also include another conv with 4 channels somewhere
        self.some_conv = torch.nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        # This inner expects the explicit conv_in behavior (3 channels)
        if isinstance(control_context, torch.Tensor):
            assert control_context.shape[1] == 3, f"expected 3 channels, got {control_context.shape[1]}"
        return torch.zeros_like(latents)


def test_wrapper_rejects_channel_mismatch_when_expected_is_4():
    # Ensure we have the real adapt_control_images in case other tests monkeypatched it
    import importlib, toolkit.control_channels as cc
    importlib.reload(cc)

    inner = DummyInnerConv4()
    inner.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'
    # explicit control_in_dim now required by strict policy
    inner.control_in_dim = 4
    wrapper = VideoXControlnetWrapper(inner)
    # single-frame control context with 3 channels should now be rejected (strict)
    ctrl = torch.randn(1, 3, 64, 64)
    lat = torch.randn(1, 16, 8, 8)
    with pytest.raises(RuntimeError, match=r"VideoXControlnetWrapper received raw pixel images"):
        wrapper(lat, 0.0, ctrl, conditioning_scale=1.0)


def test_wrapper_rejects_channel_mismatch_when_expected_is_3():
    import importlib, toolkit.control_channels as cc
    importlib.reload(cc)

    inner = DummyInnerConvMixed()
    inner.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'
    # explicit control_in_dim now required by strict policy
    inner.control_in_dim = 3
    wrapper = VideoXControlnetWrapper(inner)
    # provide 4-channel control context; wrapper should now be rejected (strict)
    ctrl = torch.randn(1, 4, 64, 64)
    lat = torch.randn(1, 16, 8, 8)
    with pytest.raises(RuntimeError, match=r"VideoXControlnetWrapper received raw pixel images"):
        wrapper(lat, 0.0, ctrl, conditioning_scale=1.0)


def test_wrapper_accepts_33_channel_control_context():
    import importlib, toolkit.control_channels as cc
    importlib.reload(cc)

    class DummyInnerAccept33(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
            # wrapper may call with a list-of-latents (VideoX parity). Accept both.
            if isinstance(control_context, torch.Tensor):
                assert control_context.shape[1] == 33
            if isinstance(latents, list):
                return torch.zeros_like(latents[0])
            return torch.zeros_like(latents)

    inner = DummyInnerAccept33()
    inner.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'
    # wrapper should forward assembled 33-channel control_context unchanged
    wrapper = VideoXControlnetWrapper(inner)
    ctrl = torch.randn(1, 33, 1, 64, 64)
    lat = torch.randn(1, 16, 8, 8)
    # should not raise
    wrapper(lat, 0.0, ctrl, conditioning_scale=1.0)

