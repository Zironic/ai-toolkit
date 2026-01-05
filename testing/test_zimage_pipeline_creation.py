import types
import torch
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
from toolkit.prompt_utils import PromptEmbeds


def test_pipeline_created_on_load_and_get_prompt_embeds(monkeypatch):
    # Monkeypatch heavy components to lightweight fakes
    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.ZImageTransformer2DModel', lambda *a, **k: object())

    class FakeTextEncoder:
        def __init__(self):
            self.device = torch.device('cpu')

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

    class FakePipeline:
        def __init__(self, **k):
            self.transformer = object()
            self.tokenizer = None
            self.text_encoder = FakeTextEncoder()

        def to(self, device):
            # emulate to() behavior
            return self

        def encode_prompt(self, prompt, do_classifier_free_guidance=False, device=None):
            # Return dummy embeddings compatible with PromptEmbeds expectations
            emb = torch.zeros(1, 77, 768)
            return emb, None

    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.ZImagePipeline', lambda **k: FakePipeline(**k))

    # Avoid heavy HF loads by stubbing tokenizer and VAE
    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*a, **k):
            return object()

    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.AutoTokenizer', FakeTokenizer)
    class FakeVAE:
        @staticmethod
        def from_pretrained(*a, **k):
            return object()
    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.AutoencoderKL', FakeVAE)

    cfg = ModelConfig(name_or_path='some_model')
    model = ZImageModel('cpu', cfg)

    # This should not raise
    model.load_model()

    assert hasattr(model, 'pipeline')

    pe = model.get_prompt_embeds("hello")
    assert isinstance(pe, PromptEmbeds)
    assert pe.text_embeds.shape[0] == 1
