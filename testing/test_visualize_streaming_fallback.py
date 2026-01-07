import os
import types

import tools.visualize_lora_attention as vla


class FakeTokenizer:
    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        return {"input_ids": [1, 2, 3], "offset_mapping": [(0, 1), (2, 3), (4, 5)]}

    def convert_ids_to_tokens(self, ids):
        return ["a", "b", "c"]

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


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
        # reuse FakePipe behavior from other tests
        return types.SimpleNamespace(images=[vla.Image.new('RGB', (64, 64), color='white')])


def test_streaming_offload_fallback(tmp_path, monkeypatch):
    out_dir = str(tmp_path)
    args = types.SimpleNamespace(prompt='a fox', model='fake.safetensors', out_dir=out_dir, top_tokens=1, layers=None, heads=None, aggregation='avg', lora=None)
    monkeypatch.setattr(vla, 'parse_args', lambda: args)

    # Make AutoTokenizer.from_pretrained return a simple tokenizer
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', lambda name: FakeTokenizer())

    # Make load_model_for_inference first raise the component-mismatch style error, then
    # accept a hint dict and return a fake sd object with a pipeline
    call_state = {'called': 0}

    def loader_side_effect(name, device, dtype, apply_lora=False):
            # raise only when a bare string name is given
            if isinstance(name, str):
                raise ValueError("Pipeline <class 'diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline'> expected ['feature_extractor', 'image_encoder', 'safety_checker', 'scheduler', 'text_encoder', 'tokenizer', 'unet', 'vae'], but only {'tokenizer', 'scheduler', 'safety_checker', 'vae', 'text_encoder'} were passed.")
            # when passed a dict hint -> return an sd-like object
            return types.SimpleNamespace(pipeline=FakePipe())

    monkeypatch.setattr('toolkit.model_utils.load_model_for_inference', loader_side_effect)

    # Run main (should handle the fallback and not crash)
    vla.main()

    assert os.path.exists(os.path.join(out_dir, 'mapping.json'))
    files = os.listdir(out_dir)
    assert any(f.endswith('.png') for f in files)
