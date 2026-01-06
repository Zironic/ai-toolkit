import os
import tempfile
import torch
from toolkit.dataloader_mixins import TextEmbeddingCachingMixin
from toolkit.prompt_utils import PromptEmbeds


class FakeSD:
    def __init__(self):
        self.device_torch = 'cpu'
        self.torch_dtype = torch.float32
        self.encode_control_in_text_embeddings = False
        self.has_multiple_control_images = False

    def encode_prompt(self, prompt, **kwargs):
        # return a simple PromptEmbeds wrapping a tensor [1, seq_len, dim]
        t = torch.randn(1, 8, 768)
        return PromptEmbeds(t)


class FakeDataset(TextEmbeddingCachingMixin):
    def __init__(self, dataset_path, sd, dataset_config):
        # bypass parent __init__ heavy behavior
        self.dataset_path = dataset_path
        self.sd = sd
        self.dataset_config = dataset_config
        self.file_list = []
        self.is_caching_text_embeddings = True


def test_split_prompt_saved(tmp_path):
    sd = FakeSD()
    ds_cfg = type('C', (), {})()
    ds_cfg.split_prompt_enabled = True
    ds_cfg.split_prompt = '[Trigger] style'

    ds = FakeDataset(str(tmp_path), sd, ds_cfg)
    ds.cache_text_embeddings()

    split_path = os.path.join(str(tmp_path), 'split_prompt.safetensors')
    assert os.path.exists(split_path)
    # load and verify
    pe = PromptEmbeds.load(split_path)
    assert hasattr(pe, 'text_embeds')
    assert pe.text_embeds is not None
