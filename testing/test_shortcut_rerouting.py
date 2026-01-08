import torch
from types import SimpleNamespace
import extensions_built_in.sd_trainer.SDTrainer as sdmod


def test_precompute_helper_removed():
    # Ensure the legacy helper was intentionally removed
    assert not hasattr(sdmod, 'use_precomputed_control_residuals')


def test_adapter_still_called_when_batch_has_control_residuals():
    trainer = SimpleNamespace()
    trainer.train_config = SimpleNamespace(controlnet_reroute='precompute')
    trainer.batch = SimpleNamespace()
    trainer.batch.control_residuals = (torch.ones(1, 3, 8, 8),)
    trainer.device_torch = torch.device('cpu')

    called = {'flag': False}

    def adapter_fn(x):
        called['flag'] = True
        return [torch.zeros(1, 3, 8, 8)]

    trainer.adapter = adapter_fn

    # The trainer should still be able to call the adapter when needed
    trainer.adapter(None)
    assert called['flag'] is True
