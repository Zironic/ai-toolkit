import pytest
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


def test_get_model_to_train_raises_if_model_not_loaded():
    cfg = ModelConfig(name_or_path='dummy')
    model = ZImageModel('cpu', cfg)
    with pytest.raises(RuntimeError, match="Model not loaded"):
        model.get_model_to_train()
