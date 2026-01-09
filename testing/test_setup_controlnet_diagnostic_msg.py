import pytest
try:
    import torch
except Exception:
    pytest.skip("Skipping: PyTorch import failed in this environment", allow_module_level=True)

from types import SimpleNamespace
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def test_setup_controlnet_diagnostic_includes_model_info(monkeypatch):
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)

    # minimal sd object that lacks _predict_noise_zimage but has some other helpers
    sd = SimpleNamespace()
    sd.predict_noise = lambda *a, **k: None
    sd.encode_control_images = lambda *a, **k: None
    sd.unet = object()

    proc.sd = sd
    proc.adapter = SimpleNamespace()
    proc.adapter_config = SimpleNamespace(controlnet_mode='zimage')
    proc.dataset_configs = [SimpleNamespace(control_type='canny')]
    proc.train_config = SimpleNamespace(require_zimage_model=True)
    proc.model_config = SimpleNamespace(name_or_path='foo', arch='sd')

    # Monkeypatch helpers so we hit the zimage detection path
    import toolkit.controlnet_utils as cu
    monkeypatch.setattr(cu, 'is_zimage_adapter', lambda a, cfg: True)
    import toolkit.control_util as util
    monkeypatch.setattr(util, 'ensure_control_in_dim', lambda a, strict=False, fallback=None: True)
    monkeypatch.setattr(util, 'enforce_zimage_control_in_dim', lambda a, expected, force=True: True)

    with pytest.raises(RuntimeError) as excinfo:
        proc.setup_controlnet_training()

    msg = str(excinfo.value)
    assert 'Diagnostics:' in msg
    assert 'Model class=' in msg
    assert 'has_predict_noise=True' in msg
    assert 'has_unet=True' in msg
    assert 'model_name=foo' in msg
