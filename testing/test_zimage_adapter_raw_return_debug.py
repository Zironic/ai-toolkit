import pytest
import torch

from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
from types import SimpleNamespace


class DummyControlNetNone:
    def __call__(self, sample, timestep, control_context, conditioning_scale=1.0):
        return None


class DummyControlNetTensor:
    def __call__(self, sample, timestep, control_context, conditioning_scale=1.0):
        # return a tensor shaped like a single down-block residual
        return torch.zeros_like(sample)


def test_zimage_adapter_none_sets_flags_and_no_fallback(monkeypatch):
    sd = ZImageModel(device='cpu', model_config=SimpleNamespace(name_or_path='x'))
    sd.transformer = SimpleNamespace(__call__=lambda *a, **k: a[0])

    lat = torch.zeros((1, 4, 8, 8))
    te = torch.zeros((1, 1, 512))
    t = torch.tensor([1.0])

    # When adapter returns None, behave as a hard failure (adapter must provide hints)
    with pytest.raises(RuntimeError):
        sd._predict_noise_zimage(lat, te, t, zimage_controlnet=DummyControlNetNone(), zimage_control_images=torch.zeros((1, 3, 64, 64)), zimage_conditioning_scale=1.0)

    # Diagnostics should remain unset
    assert getattr(sd, '_last_zimage_control_hints_present', None) is False
    assert getattr(sd, '_last_zimage_control_hints_shapes', None) is None


def test_zimage_adapter_tensor_sets_flags_and_shapes(monkeypatch):
    sd = ZImageModel(device='cpu', model_config=SimpleNamespace(name_or_path='x'))
    sd.transformer = SimpleNamespace(__call__=lambda *a, **k: a[0])

    lat = torch.zeros((1, 4, 8, 8))
    te = torch.zeros((1, 1, 512))
    t = torch.tensor([1.0])

    sd._predict_noise_zimage(lat, te, t, zimage_controlnet=DummyControlNetTensor(), zimage_control_images=torch.zeros((1, 3, 64, 64)), zimage_conditioning_scale=1.0)
    assert getattr(sd, '_last_zimage_control_hints_present', None) is True
    shapes = getattr(sd, '_last_zimage_control_hints_shapes', None)
    # expect at least one shape recorded
    assert shapes is not None and len(shapes) >= 1