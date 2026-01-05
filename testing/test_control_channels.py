import torch
from toolkit.control_channels import (
    adapt_control_images,
    adapt_noisy_latents_for_adapter,
)
import types


class DummyAdapterWithConv:
    def __init__(self, in_ch):
        self.conv_in = torch.nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=3, padding=1)


def test_collapse_5d_squeeze_and_trim():
    inner = DummyAdapterWithConv(3)
    ctrl5 = torch.randn(1, 4, 1, 32, 32)
    adapted, expected = adapt_control_images(ctrl5, inner)
    assert expected == 3
    assert adapted.ndim == 4
    assert adapted.shape[1] == 3


def test_mean_frames_and_trim():
    inner = DummyAdapterWithConv(3)
    ctrl5 = torch.randn(1, 4, 2, 32, 32)
    adapted, expected = adapt_control_images(ctrl5, inner)
    assert adapted.ndim == 4
    assert adapted.shape[1] == 3


def test_preserve_4_when_expected_4():
    inner = DummyAdapterWithConv(4)
    ctrl4 = torch.randn(1, 4, 32, 32)
    adapted, expected = adapt_control_images(ctrl4, inner)
    assert adapted.shape[1] == 4


def test_adapt_noisy_latents_grouped_mean():
    # 16 -> 4 grouped mean
    lat = torch.randn(1, 16, 32, 32)
    out = adapt_noisy_latents_for_adapter(lat, 4)
    assert out.shape[1] == 4


def test_adapt_noisy_latents_pad_and_slice():
    lat = torch.randn(1, 6, 32, 32)
    out = adapt_noisy_latents_for_adapter(lat, 4)
    assert out.shape[1] == 4
    lat2 = torch.randn(1, 4, 32, 32)
    out2 = adapt_noisy_latents_for_adapter(lat2, 6)
    assert out2.shape[1] == 6


def test_zimage_special_case_padding():
    # 3ch single-frame should be padded to 4 for the Z-Image Turbo adapter
    from types import SimpleNamespace
    adapter = SimpleNamespace()
    adapter.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'
    ctrl5 = torch.randn(1, 3, 1, 64, 64)
    adapted, expected = adapt_control_images(ctrl5, adapter)
    assert expected == 4
    assert adapted.shape[1] == 4


def test_zimage_special_case_prefers_conv_in_when_present():
    # If the adapter exposes a conv with in_channels==3, prefer 3 instead of forcing 4
    from types import SimpleNamespace
    adapter = SimpleNamespace()
    adapter.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'
    adapter.conv_in = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
    ctrl4 = torch.randn(1, 4, 32, 32)
    adapted, expected = adapt_control_images(ctrl4, adapter)
    assert expected == 3
    assert adapted.shape[1] == 3


def test_zimage_control_context_passthrough():
    # If a 33-channel assembled control_context is provided, pass it through
    # unchanged and report expected == 33 even if the adapter config suggests 4.
    from types import SimpleNamespace
    adapter = SimpleNamespace()
    adapter.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'
    # Simulate an adapter that would otherwise prefer 4 channels
    adapter.control_in_dim = 4
    ctrl5 = torch.randn(1, 33, 1, 64, 64)
    adapted, expected = adapt_control_images(ctrl5, adapter)
    assert expected == 33
    assert adapted.ndim == 4
    assert adapted.shape[1] == 33

    # Also accept already-collapsed 4D control_contexts
    ctrl4 = torch.randn(1, 33, 64, 64)
    adapted2, expected2 = adapt_control_images(ctrl4, adapter)
    assert expected2 == 33
    assert adapted2.shape[1] == 33

