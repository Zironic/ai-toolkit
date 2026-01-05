import pytest
import torch
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


class DummyTE:
    def to(self, *args, **kwargs):
        return self


def test_generation_pipeline_requires_tokenizer():
    cfg = ModelConfig(name_or_path="nonexistent-model", arch="zimage")
    sd = ZImageModel(device="cpu", model_config=cfg, dtype="float32")

    # Simulate missing tokenizer but a present text encoder
    sd.tokenizer = None
    sd.text_encoder = [DummyTE()]

    with pytest.raises(RuntimeError) as excinfo:
        sd.get_generation_pipeline()
    msg = str(excinfo.value)
    assert "requires a tokenizer implementing `apply_chat_template`" in msg
