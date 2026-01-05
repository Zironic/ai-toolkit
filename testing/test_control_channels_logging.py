import torch
from toolkit.control_channels import adapt_noisy_latents_for_adapter, adapt_control_images
from types import SimpleNamespace


def test_adapt_noisy_latents_prints_grouped_mean(capsys):
    lat = torch.randn(1, 16, 32, 32)
    out = adapt_noisy_latents_for_adapter(lat, 4)
    captured = capsys.readouterr()
    assert "adapt_noisy_latents_for_adapter" in captured.out
    assert "grouped_mean_reduction" in captured.out or "grouped_mean" in captured.out
    assert out.shape[1] == 4


def test_adapt_control_images_prints_trim_or_pad(capsys):
    adapter = SimpleNamespace()
    adapter.name_or_path = 'alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps'
    ctrl5 = torch.randn(1, 3, 1, 64, 64)
    adapted, expected = adapt_control_images(ctrl5, adapter)
    captured = capsys.readouterr()
    assert "adapt_control_images: adapted channels" in captured.out
    assert adapted.shape[1] == expected
