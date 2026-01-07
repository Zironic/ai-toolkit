import pytest
import torch

from toolkit.control_channels import adapt_control_images


class DummyAdapter:
    def __init__(self, name=None, control_in_dim=None):
        self.name_or_path = name
        self.control_in_dim = control_in_dim


def test_adapt_control_images_does_not_infer_from_name():
    # Adapter name contains 'zimage' but no explicit control_in_dim.
    adapter = DummyAdapter(name='my_zimage_adapter', control_in_dim=None)
    ctrl = torch.zeros((1, 4, 64, 64))
    out, expected = adapt_control_images(ctrl, adapter)
    # No heuristics: expected must remain None and tensor should be returned unchanged
    assert expected is None
    assert isinstance(out, torch.Tensor)
    assert out.shape == ctrl.shape


def test_adapt_control_images_strict_raises_when_adapter_has_explicit_and_input_is_pixel():
    # Adapter declares an explicit control_in_dim but has no name; pixel-like inputs should error
    adapter = DummyAdapter(name=None, control_in_dim=3)
    ctrl = torch.zeros((1, 4, 64, 64))
    with pytest.raises(RuntimeError) as exc:
        adapt_control_images(ctrl, adapter)
    assert 'looks_like=pixel-image' in str(exc.value) or 'expected 3' in str(exc.value)
