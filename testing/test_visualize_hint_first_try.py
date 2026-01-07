import types
import tools.visualize_lora_attention as vla


class FakePipe(vla.FakePipe if hasattr(vla, 'FakePipe') else object):
    def __init__(self):
        try:
            super().__init__()
            self.unet = types.SimpleNamespace(set_attn_processor=lambda p: p)
        except Exception:
            self.unet = types.SimpleNamespace(set_attn_processor=lambda p: p)
        self.device = 'cpu'
        self.vae = types.SimpleNamespace(sample_size=64)

    def __call__(self, prompt, num_inference_steps=20, generator=None):
        return types.SimpleNamespace(images=[vla.Image.new('RGB', (64, 64), color='white')])


def test_hint_used_first(monkeypatch, tmp_path):
    out_dir = str(tmp_path)
    args = types.SimpleNamespace(prompt='a fox', model='model.safetensors', out_dir=out_dir, top_tokens=1, layers=None, heads=None, aggregation='avg', lora=None)
    monkeypatch.setattr(vla, 'parse_args', lambda: args)

    # monkeypatch tokenizer
    class FakeTok:
        def __call__(self, *a, **k):
            return {'input_ids': [1], 'offset_mapping': [(0,1)]}
        def convert_ids_to_tokens(self, ids):
            return ['a']
        def encode(self, text, add_special_tokens=False):
            return [1]
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda name: FakeTok())

    called = {'args': None}

    def fake_loader(cfg, device, dtype, apply_lora=False):
        # ensure we received a hint dict rather than a str
        called['args'] = cfg
        return types.SimpleNamespace(pipeline=FakePipe())

    monkeypatch.setattr('toolkit.model_utils.load_model_for_inference', fake_loader)

    vla.main()
    # assert loader was called with a dict containing hint keys
    assert isinstance(called['args'], dict)
    assert 'unet_path' in called['args'] and 'te_name_or_path' in called['args']
    # also ensure we pass feature/image/safety component hints
    assert 'feature_extractor' in called['args'] and 'image_encoder' in called['args'] and 'safety_checker' in called['args']
