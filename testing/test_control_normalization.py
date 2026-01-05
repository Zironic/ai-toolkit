import pytest
from toolkit.config_modules import DatasetConfig, ModelConfig


def test_dataset_controls_are_normalized():
    ds = DatasetConfig(controls=[' Pose ', 'CANNY', ' ', 'Depth'])
    assert ds.controls == ['pose', 'canny', 'depth']


def test_preflight_detects_missing_controlnet_aux_for_normalized_controls(monkeypatch):
    # Ensure BaseSDTrainProcess preflight will raise when dataset requests pose but controlnet_aux missing
    from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess
    from types import SimpleNamespace

    dummy = SimpleNamespace()
    dummy.model_config = SimpleNamespace(controlnet_enabled=True, controlnet_name_or_path='some', controlnet_file='ckpt.safetensors')
    dummy.dataset_configs = [DatasetConfig(controls=[' Pose '])]
    # Simulate absence of controlnet_aux by removing it from sys.modules and forcing import to fail
    import sys

    monkeypatch.setitem(sys.modules, 'controlnet_aux', None)

    # call hook_before_train_loop and expect a RuntimeError
    with pytest.raises(RuntimeError):
        BaseSDTrainProcess.hook_before_train_loop(dummy)
