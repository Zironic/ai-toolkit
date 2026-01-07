import os
import types
import tempfile
from PIL import Image

import tools.visualize_lora_attention as vla


class FakeTokenizer:
    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 1), (2, 3), (4, 5)]}
    def convert_ids_to_tokens(self, ids):
        return ["a", "b", "c"]
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class FakePipe:
    def __init__(self):
        self.unet = types.SimpleNamespace(set_attn_processor=lambda p: p)
        self.device = 'cpu'
        self.vae = types.SimpleNamespace(sample_size=64)
        self.tokenizer = FakeTokenizer()
    def enable_attention_slicing(self):
        return None
    def __call__(self, prompt, num_inference_steps=20, generator=None):
        return types.SimpleNamespace(images=[Image.new('RGB', (64, 64), color='white')])


def test_tokenizer_fallback(tmp_path, monkeypatch):
    out_dir = str(tmp_path)
    args = types.SimpleNamespace(prompt='a fox', model='unknown-model', out_dir=out_dir, top_tokens=2, layers=None, heads=None, aggregation='avg', lora=None)
    monkeypatch.setattr(vla, 'parse_args', lambda: args)
    # Make AutoTokenizer.from_pretrained raise
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda name: (_ for _ in ()).throw(ValueError('no tokenizer')))
    # return an sd-like object with a pipeline
    monkeypatch.setattr('toolkit.model_utils.load_model_for_inference', lambda name, device, dtype, apply_lora: types.SimpleNamespace(pipeline=FakePipe()))

    # Run main (should use pipeline.tokenizer fallback and not raise)
    vla.main()

    # mapping.json should exist
    assert os.path.exists(os.path.join(out_dir, 'mapping.json'))
