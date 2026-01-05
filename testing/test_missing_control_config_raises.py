import pytest
import types
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


def test_load_model_raises_when_control_enabled_but_config_missing(monkeypatch):
    cfg = ModelConfig(name_or_path='some_model', controlnet_enabled=True, controlnet_name_or_path=None, controlnet_file=None)
    model = ZImageModel('cpu', cfg)

    # Patch heavy parts of load_model to avoid HF calls
    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.ZImageTransformer2DModel', lambda *a, **k: object())
    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.ZImagePipeline', lambda **k: object())
    # Avoid network calls when attempting to load tokenizer from HF hub
    monkeypatch.setattr('transformers.AutoTokenizer', types.SimpleNamespace(from_pretrained=lambda *a, **k: None))
    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.AutoTokenizer', types.SimpleNamespace(from_pretrained=lambda *a, **k: None))

    with pytest.raises(RuntimeError):
        model.load_model()
