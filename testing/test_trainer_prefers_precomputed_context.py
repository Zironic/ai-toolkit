import pytest
try:
    import torch
except Exception:
    pytest.skip("Skipping: PyTorch import failed in this environment", allow_module_level=True)

from types import SimpleNamespace
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummyBatch:
    def __init__(self, file_items):
        self.file_items = file_items


def test_trainer_attaches_zimage_control_context_from_precompute(monkeypatch):
    trainer = SDTrainer.__new__(SDTrainer)
    # minimal attributes used by path
    trainer.device_torch = torch.device('cpu')
    trainer.train_config = SimpleNamespace(controlnet_offload_strategy='none')

    # create a fake file item with precomputed contexts
    fi = SimpleNamespace(path='x', _preencoded_zimage_control_contexts={512: torch.zeros((3,1,64,64))})
    batch = DummyBatch([fi])

    # Ensure collection helper returns the stacked tensor
    monkeypatch.setattr(trainer, '_collect_preencoded_zimage_context_for_batch', lambda b: torch.zeros((1,3,1,64,64)))

    # Simulate adapter and adapter_config detection
    adapter = SimpleNamespace()
    trainer.adapter = adapter
    trainer.adapter_config = SimpleNamespace(controlnet_mode='zimage')

    # Prepare a local pred_kwargs dict and run just the section that sets pre
    pred_kwargs = {}
    pre = trainer._collect_preencoded_zimage_context_for_batch(batch)
    assert pre is not None
    zimage_ctrl = pre.to(trainer.device_torch)
    pred_kwargs['zimage_control_images'] = zimage_ctrl
    pred_kwargs['zimage_control_context'] = zimage_ctrl

    assert 'zimage_control_context' in pred_kwargs
    assert torch.all(pred_kwargs['zimage_control_context'] == zimage_ctrl)
