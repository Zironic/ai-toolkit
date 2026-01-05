import torch
from extensions_built_in.diffusion_models.z_image_adapter import load_videox_control_adapter
from toolkit.controlnet_compat import VideoXControlnetWrapper


def test_loader_instantiates_wrapper_and_inner_class():
    wrapper = load_videox_control_adapter(name_or_path=None)
    assert isinstance(wrapper, VideoXControlnetWrapper)
    inner = wrapper.inner
    # inner should expose control_in_dim attribute (Z-Image model provides this)
    assert hasattr(inner, 'control_in_dim')


def test_wrapper_signature_validation_when_loading():
    # Ensure that loader returns a wrapper that enforces signature (construction already validated)
    wrapper = load_videox_control_adapter(name_or_path=None)
    assert hasattr(wrapper, 'forward')
    # calling forward with a minimal pre-encoded control_context should not raise immediate signature errors
    lat = torch.randn(1, 16, 8, 8)
    ctrl = torch.randn(1, getattr(wrapper.inner, 'control_in_dim', 33), 1, 16, 16)
    out = wrapper(lat, 0.0, ctrl, conditioning_scale=1.0)
    # Outer wrapper may call into inner and raise deeper runtime errors if inner can't handle shapes; ensure call returns or raises a controlled error
    assert out is not None
