import sys
import pytest
# Skip these tests when a compatible PyTorch is not available in the test environment.
pytest.importorskip("torch")
from types import SimpleNamespace

from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def test_setup_controlnet_requires_model_hook_raises(monkeypatch):
    # Minimal instance without __init__ to avoid heavy setup
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)

    # Minimal SD and adapter config to trigger zimage detection
    proc.sd = SimpleNamespace(is_controlnet_enabled=True, controlnet=SimpleNamespace())
    proc.adapter = None
    proc.adapter_config = SimpleNamespace(controlnet_mode='zimage')
    proc.dataset_configs = [SimpleNamespace(control_type='canny')]
    proc.train_config = SimpleNamespace(require_zimage_model=True)

    # Ensure VideoX wrapper symbol exists so wrapping code can run deterministically
    monkeypatch.setitem(sys.modules, 'toolkit.controlnet_compat', SimpleNamespace(VideoXControlnetWrapper=type('D', (), {})))

    # Patch helpers to avoid deep ControlNet logic: detect zimage and succeed at control_in_dim enforcement
    import toolkit.controlnet_utils as cu
    monkeypatch.setattr(cu, 'is_zimage_adapter', lambda a, cfg: True)
    import toolkit.control_util as util
    monkeypatch.setattr(util, 'ensure_control_in_dim', lambda a, strict=False, fallback=None: True)
    monkeypatch.setattr(util, 'enforce_zimage_control_in_dim', lambda a, expected, force=True: True)

    with pytest.raises(RuntimeError) as excinfo:
        proc.setup_controlnet_training()

    assert 'does not provide' in str(excinfo.value) or 'require_zimage_model' in str(excinfo.value)


def test_setup_controlnet_allows_fallback_when_config_false(monkeypatch):
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)

    proc.sd = SimpleNamespace(is_controlnet_enabled=True, controlnet=SimpleNamespace())
    proc.adapter = None
    proc.adapter_config = SimpleNamespace(controlnet_mode='zimage')
    proc.dataset_configs = [SimpleNamespace(control_type='canny')]
    proc.train_config = SimpleNamespace(require_zimage_model=False)

    monkeypatch.setitem(sys.modules, 'toolkit.controlnet_compat', SimpleNamespace(VideoXControlnetWrapper=type('D', (), {})))
    import toolkit.controlnet_utils as cu
    monkeypatch.setattr(cu, 'is_zimage_adapter', lambda a, cfg: True)
    import toolkit.control_util as util
    monkeypatch.setattr(util, 'ensure_control_in_dim', lambda a, strict=False, fallback=None: True)
    monkeypatch.setattr(util, 'enforce_zimage_control_in_dim', lambda a, expected, force=True: True)

    # Should not raise
    proc.setup_controlnet_training()


def test_setup_controlnet_accepts_model_hook(monkeypatch):
    # When the SD instance provides the `_predict_noise_zimage` hook, setup should not raise
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)

    # Provide an object that looks zimage-capable
    proc.sd = SimpleNamespace(is_controlnet_enabled=True, controlnet=SimpleNamespace(), _predict_noise_zimage=lambda *a, **k: None)
    proc.adapter = None
    proc.adapter_config = SimpleNamespace(controlnet_mode='zimage')
    proc.dataset_configs = [SimpleNamespace(control_type='canny')]
    proc.train_config = SimpleNamespace(require_zimage_model=True)

    monkeypatch.setitem(sys.modules, 'toolkit.controlnet_compat', SimpleNamespace(VideoXControlnetWrapper=type('D', (), {})))
    import toolkit.controlnet_utils as cu
    monkeypatch.setattr(cu, 'is_zimage_adapter', lambda a, cfg: True)
    import toolkit.control_util as util
    monkeypatch.setattr(util, 'ensure_control_in_dim', lambda a, strict=False, fallback=None: True)
    monkeypatch.setattr(util, 'enforce_zimage_control_in_dim', lambda a, expected, force=True: True)

    # Should not raise
    proc.setup_controlnet_training()
