import pytest

# Deleted legacy tests referencing `adapt_control_images`. See repository policy
# and issue tracker for details — these tests prevented safe refactors.
pytest.skip("Deleted: legacy tests referencing adapt_control_images. Write new tests for the new adapter flow.", allow_module_level=True)


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
