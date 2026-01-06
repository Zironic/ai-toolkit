import os
import pytest
import torch
from toolkit.dataloader_mixins import TextEmbeddingCachingMixin


class BadSD:
    def __init__(self):
        self.device_torch = 'cpu'
        self.torch_dtype = torch.float32
        self.encode_control_in_text_embeddings = False
        self.has_multiple_control_images = False

    def encode_prompt(self, prompt, **kwargs):
        raise ValueError("encoder broken")


class FakeDataset(TextEmbeddingCachingMixin):
    def __init__(self, dataset_path, sd, dataset_config):
        # bypass parent __init__ heavy behavior
        self.dataset_path = dataset_path
        self.sd = sd
        self.dataset_config = dataset_config
        self.file_list = []
        self.is_caching_text_embeddings = True


def test_split_prompt_encode_failure_raises(tmp_path):
    sd = BadSD()
    ds_cfg = type('C', (), {})()
    ds_cfg.split_prompt_enabled = True
    ds_cfg.split_prompt = '[Trigger] style'

    ds = FakeDataset(str(tmp_path), sd, ds_cfg)
    with pytest.raises(RuntimeError) as exc:
        ds.cache_text_embeddings()
    assert 'SplitPrompt encoding failed' in str(exc.value)
