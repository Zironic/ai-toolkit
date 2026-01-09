import types
import pytest

from toolkit.control_util import prepare_controlnet_adapter


class DummySD:
    """Simple fake SD instance for validation tests."""
    def __init__(self, provide_zimage=False):
        if provide_zimage:
            # provide a callable hook
            def _predict_noise_zimage(*args, **kwargs):
                return None
            self._predict_noise_zimage = _predict_noise_zimage
            def encode_control_images(*args, **kwargs):
                return None
            self.encode_control_images = encode_control_images


def test_prepare_controlnet_adapter_sets_control_in_dim_for_zimage():
    a = types.SimpleNamespace()
    a.name_or_path = 'my-org/zimage-example'
    # no control_in_dim initially
    assert not hasattr(a, 'control_in_dim')
    adapter = prepare_controlnet_adapter(sd=None, adapter_spec_or_obj=a, adapter_config=types.SimpleNamespace(controlnet_mode='zimage'), train_config=None, strict=True, require_zimage_model=False)
    assert getattr(adapter, 'control_in_dim', None) == 33


def test_prepare_controlnet_adapter_requires_sd_predict_noise_zimage():
    a = types.SimpleNamespace()
    a.name_or_path = 'my-org/zimage-example'
    a.control_in_dim = 33
    sd = DummySD(provide_zimage=False)
    with pytest.raises(RuntimeError) as exc:
        prepare_controlnet_adapter(sd=sd, adapter_spec_or_obj=a, adapter_config=types.SimpleNamespace(controlnet_mode='zimage'), train_config=None, strict=True, require_zimage_model=True)
    assert '_predict_noise_zimage' in str(exc.value) or 'Z-Image' in str(exc.value)
