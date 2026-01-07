import os
import types
import tempfile
from PIL import Image

import toolkit.attn_recorder as ar
import tools.visualize_lora_attention as vla


class FakeTokenizer:
    def __init__(self):
        pass

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 1), (2, 3), (4, 5)]}

    def convert_ids_to_tokens(self, ids):
        return ["a", "b", "c"]

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


import torch

class FakeUNet:
    def set_attn_processor(self, proc):
        # ensure recorder is stored and create a fake recorded attention map
        self._rec = proc
        # fake attn record: [B=1, H=2, T=4, S=4]
        proc.records.append({"module": "fake", "block_idx": 0, "attn": torch.ones((1, 2, 4, 4))})


class FakePipe:
    def __init__(self):
        self.unet = FakeUNet()
        self.device = 'cpu'
        self.vae = types.SimpleNamespace(sample_size=64)

    def enable_attention_slicing(self):
        return None

    def __call__(self, prompt, num_inference_steps=20, generator=None):
        return types.SimpleNamespace(images=[Image.new('RGB', (64, 64), color='white')])


def test_cli_runs_and_writes(tmp_path, monkeypatch):
    out_dir = str(tmp_path)
    args = types.SimpleNamespace(prompt='a fox', model='fake-model', out_dir=out_dir, top_tokens=2, layers=None, heads=None, aggregation='avg', lora=None)
    monkeypatch.setattr(vla, 'parse_args', lambda: args)

    # patch tokenizer loader (patch transformers.AutoTokenizer.from_pretrained)
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda name: FakeTokenizer())
    # patch toolkit loader to return a simple object with a pipeline
    monkeypatch.setattr('toolkit.model_utils.load_model_for_inference', lambda name, device, dtype, apply_lora: types.SimpleNamespace(pipeline=FakePipe()))

    # Run main (should not raise)
    vla.main()

    # Check outputs
    assert os.path.exists(os.path.join(out_dir, 'mapping.json'))
    # overlays written
    files = os.listdir(out_dir)
    assert any(f.endswith('.png') for f in files)
    assert os.path.exists(os.path.join(out_dir, 'attn_data.npz'))
