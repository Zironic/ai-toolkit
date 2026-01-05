import os
import pytest
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


def test_model_config_does_not_expose_allow_dummy_tokenizer():
    # passing the kw should not create a persistent attribute on ModelConfig
    cfg = ModelConfig(name_or_path="nonexistent-model", arch="zimage", allow_dummy_tokenizer=True)
    assert not hasattr(cfg, 'allow_dummy_tokenizer')


def test_env_var_dummy_tokenizer_not_supported(monkeypatch):
    monkeypatch.setenv('ZIMAGE_ALLOW_DUMMY_TOKENIZER', '1')
    cfg = ModelConfig(name_or_path="nonexistent-model", arch="zimage")
    sd = ZImageModel(device="cpu", model_config=cfg, dtype="float32")

    sd.tokenizer = None
    sd.text_encoder = [object()]

    with pytest.raises(Exception):
        sd.get_generation_pipeline()
