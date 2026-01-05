import pytest
import torch
from types import SimpleNamespace
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel, AutoencoderKL, AutoTokenizer, Qwen3ForCausalLM
from toolkit.config_modules import ModelConfig


def test_zimage_loads_vae_and_sets_scale(monkeypatch):
    # Arrange: make AutoencoderKL.from_pretrained return a dummy VAE with minimal API
    class DummyVAE:
        def __init__(self):
            self.config = {"block_out_channels": [32, 64, 128]}
            self.dtype = torch.float32

        def to(self, device, dtype=None):
            return self

        def eval(self):
            return None

        def requires_grad_(self, flag):
            return None

    monkeypatch.setattr(AutoencoderKL, 'from_pretrained', staticmethod(lambda *a, **k: DummyVAE()))

    # Tokenizer/text encoder optional; ensure they fail gracefully
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(Qwen3ForCausalLM, 'from_pretrained', staticmethod(lambda *a, **k: (_ for _ in ()).throw(Exception('no te'))))

    # Make transformer.from_pretrained return a lightweight dummy to avoid network calls
    class DummyTransformer:
        def __init__(self):
            pass
        def state_dict(self):
            return {}
    import extensions_built_in.diffusion_models.z_image.z_image as zmod
    monkeypatch.setattr(zmod.ZImageTransformer2DModel, 'from_pretrained', staticmethod(lambda *a, **k: DummyTransformer()))

    cfg = ModelConfig(name_or_path='dummy_model')
    model = ZImageModel('cpu', cfg)

    # Act
    model.load_model()

    # Assert
    assert hasattr(model, 'vae') and model.vae is not None
    # vae_scale_factor = 2 ** (len(block_out_channels) - 1) -> 2 ** (3 - 1) = 4
    assert model.vae_scale_factor == 4


def test_zimage_handles_readonly_vae_config(monkeypatch):
    # Arrange: make AutoencoderKL.from_pretrained return a VAE whose `config` is a read-only property
    class ReadOnlyVAE:
        def __init__(self):
            self._cfg = {"block_out_channels": [32, 64, 128]}
            self.dtype = torch.float32

        @property
        def config(self):
            return self._cfg

        def to(self, device, dtype=None):
            return self

        def eval(self):
            return None

        def requires_grad_(self, flag):
            return None

    monkeypatch.setattr(AutoencoderKL, 'from_pretrained', staticmethod(lambda *a, **k: ReadOnlyVAE()))

    # Tokenizer/text encoder optional; ensure they fail gracefully
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(Qwen3ForCausalLM, 'from_pretrained', staticmethod(lambda *a, **k: (_ for _ in ()).throw(Exception('no te'))))

    # Make transformer.from_pretrained return a lightweight dummy to avoid network calls
    class DummyTransformer:
        def __init__(self):
            pass
        def state_dict(self):
            return {}
    import extensions_built_in.diffusion_models.z_image.z_image as zmod
    monkeypatch.setattr(zmod.ZImageTransformer2DModel, 'from_pretrained', staticmethod(lambda *a, **k: DummyTransformer()))

    cfg = ModelConfig(name_or_path='dummy_model')
    model = ZImageModel('cpu', cfg)

    # Act
    model.load_model()
    pipeline = model.get_generation_pipeline()

    # Assert: pipeline should have a vae with a `.config` that supports attribute access
    assert hasattr(pipeline, 'vae') and pipeline.vae is not None
    assert hasattr(pipeline.vae, 'config')
    from types import SimpleNamespace
    assert isinstance(pipeline.vae.config, SimpleNamespace)
