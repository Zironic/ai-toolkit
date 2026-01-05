import types
import pytest
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
from toolkit.config_modules import ModelConfig


def make_minimal_model_config():
    cfg = ModelConfig(name_or_path='dummy')
    cfg.controlnet_enabled = True
    cfg.controlnet_name_or_path = 'some/controlnet'
    return cfg


def test_validate_controlnet_raises_when_none():
    mcfg = make_minimal_model_config()
    zm = ZImageModel(device='cpu', model_config=mcfg)
    # simulate a failed load
    zm.controlnet = None
    with pytest.raises(RuntimeError, match=r"ControlNet load failed: 'controlnet' is None"):
        zm.validate_controlnet('some/controlnet')


def test_validate_controlnet_raises_when_missing_name():
    mcfg = make_minimal_model_config()
    zm = ZImageModel(device='cpu', model_config=mcfg)
    class FakeAdapter:
        pass
    zm.controlnet = FakeAdapter()
    with pytest.raises(RuntimeError, match=r"missing required attribute 'name_or_path'"):
        zm.validate_controlnet('some/controlnet')
