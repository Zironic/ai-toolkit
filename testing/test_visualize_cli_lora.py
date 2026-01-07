import os
import types
import tempfile
from PIL import Image

import tools.visualize_lora_attention as vla


class FakeTokenizer:
    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"input_ids": [1, 2], "offset_mapping": [(0, 1), (2, 3)]}
    def convert_ids_to_tokens(self, ids):
        return ["a", "b"]
    def encode(self, text, add_special_tokens=False):
        return [1, 2]


class FakePipe(vla.FakePipe if hasattr(vla, 'FakePipe') else object):
    def __init__(self):
        # reuse FakePipe behavior if present
        try:
            super().__init__()
            self.unet = types.SimpleNamespace(set_attn_processor=lambda p: p)
        except Exception:
            self.unet = types.SimpleNamespace(set_attn_processor=lambda p: p)
        self.device = 'cpu'
        self.vae = types.SimpleNamespace(sample_size=64)
    def __call__(self, prompt, num_inference_steps=20, generator=None):
        return types.SimpleNamespace(images=[Image.new('RGB', (64, 64), color='white')])


def test_cli_lora_delta_mode(tmp_path, monkeypatch):
    out_dir = str(tmp_path)
    args = types.SimpleNamespace(prompt='a fox', model='fake-model', out_dir=out_dir, top_tokens=1, layers=None, heads=None, aggregation='avg', lora='fake.safetensors', map_type='delta')
    monkeypatch.setattr(vla, 'parse_args', lambda: args)
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda name: FakeTokenizer())
    # return an sd-like object with a pipeline
    monkeypatch.setattr('toolkit.model_utils.load_model_for_inference', lambda name, device, dtype, apply_lora: types.SimpleNamespace(pipeline=FakePipe()))

    # monkeypatch loading of safetensors: return a simple fake dict with a lora weight
    monkeypatch.setattr('safetensors.torch.load_file', lambda path: {'lora_A.weight': __import__('torch').ones((4,4))})

    # monkeypatch LoRASpecialNetwork to be a lightweight object with apply_to, load_weights, is_active
    class FakeLoRA:
        def __init__(self, *a, **k):
            self.is_active = False
        def apply_to(self, *a, **k):
            return
        def _update_torch_multiplier(self):
            return
        def load_weights(self, sd):
            return
    monkeypatch.setattr('toolkit.lora_special.LoRASpecialNetwork', FakeLoRA)

    # Run
    vla.main()

    # Check outputs
    assert os.path.exists(os.path.join(out_dir, 'mapping.json'))
    files = os.listdir(out_dir)
    assert any(f.endswith('.png') for f in files)
    assert os.path.exists(os.path.join(out_dir, 'attn_base.npz'))
    assert os.path.exists(os.path.join(out_dir, 'attn_lora.npz'))
    assert os.path.exists(os.path.join(out_dir, 'attn_delta.npz'))