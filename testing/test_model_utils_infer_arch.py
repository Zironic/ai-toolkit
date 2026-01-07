import types


def test_infer_arch_for_tongyi(monkeypatch):
    recorded = {}

    class FakeModel:
        def __init__(self, device, model_config, dtype):
            # capture arch on construction
            recorded['arch'] = getattr(model_config, 'arch', None)
            self.pipeline = None

        def load_model(self):
            # no-op
            pass

    # Patch get_model_class to return our FakeModel and avoid heavy loads
    monkeypatch.setattr('toolkit.util.get_model.get_model_class', lambda mc: FakeModel)

    from toolkit.model_utils import load_model_for_inference

    # Call with the Tongyi Z-Image Turbo hub id (string path)
    m = load_model_for_inference('Tongyi-MAI/Z-Image-Turbo', device='cpu', dtype='float32', apply_lora=False)

    assert recorded.get('arch') == 'zimage', f"expected 'zimage' arch inferred, got {recorded.get('arch')}"
