import pytest
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


def test_tokenizer_uses_subfolder_arg(monkeypatch):
    cfg = ModelConfig(name_or_path="base_path", arch="zimage")
    sd = ZImageModel(device="cpu", model_config=cfg, dtype="float32")

    captured = {}

    from extensions_built_in.diffusion_models.z_image.z_image import AutoTokenizer

    def fake_from_pretrained(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        raise Exception("simulated missing tokenizer")

    from extensions_built_in.diffusion_models.z_image.z_image import ZImageTransformer2DModel

    # monkeypatch transformer load to allow tokenizer load to be attempted
    monkeypatch.setattr(ZImageTransformer2DModel, 'from_pretrained', lambda *a, **k: object())
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', fake_from_pretrained)

    with pytest.raises(Exception):
        sd.load_model()

    # ensure the loader called from_pretrained with subfolder='tokenizer'
    assert ('tokenizer' in captured.get('args', [])) or (captured.get('kwargs', {}).get('subfolder') == 'tokenizer')
