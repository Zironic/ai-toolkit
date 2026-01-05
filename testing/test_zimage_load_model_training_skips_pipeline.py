import pytest
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


def test_load_model_raises_when_tokenizer_missing(monkeypatch):
    cfg = ModelConfig(name_or_path="nonexistent-model", arch="zimage")
    sd = ZImageModel(device="cpu", model_config=cfg, dtype="float32")

    def fake_from_pretrained(*args, **kwargs):
        raise Exception("tokenizer missing")

    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.AutoTokenizer.from_pretrained', fake_from_pretrained)

    with pytest.raises(Exception):
        sd.load_model()
