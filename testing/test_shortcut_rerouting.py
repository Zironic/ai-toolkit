import torch
from types import SimpleNamespace

from extensions_built_in.sd_trainer.SDTrainer import use_precomputed_control_residuals


def test_use_precomputed_control_residuals_returns_residuals():
    trainer = SimpleNamespace()
    trainer.train_config = SimpleNamespace(controlnet_reroute='precompute')
    trainer.batch = SimpleNamespace()
    # tuple of per-scale tensors with batch dim
    trainer.batch.control_residuals = (torch.ones(2, 3, 8, 8), torch.ones(2, 6, 4, 4))
    trainer.device_torch = torch.device('cpu')

    res = use_precomputed_control_residuals(trainer, dtype=torch.float32)
    assert res is not None
    assert isinstance(res, list)
    assert len(res) == 2
    assert res[0].shape == (2, 3, 8, 8)


def test_use_precomputed_control_residuals_respects_config_flag():
    trainer = SimpleNamespace()
    trainer.train_config = SimpleNamespace(controlnet_reroute='none')
    trainer.batch = SimpleNamespace()
    trainer.batch.control_residuals = (torch.ones(1, 1, 8, 8),)
    trainer.device_torch = torch.device('cpu')

    res = use_precomputed_control_residuals(trainer, dtype=torch.float32)
    assert res is None


def test_adapter_not_called_when_precomputed_present():
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

    # use helper to detect precomputed residuals
    pre = use_precomputed_control_residuals(trainer, dtype=torch.float32)
    assert pre is not None
    # ensure adapter was not invoked
    assert called['flag'] is False


def test_use_precomputed_rejects_non_tensor_entries():
    trainer = SimpleNamespace()
    trainer.train_config = SimpleNamespace(controlnet_reroute='precompute')
    trainer.batch = SimpleNamespace()
    trainer.batch.control_residuals = (torch.ones(1, 3, 8, 8), "not-a-tensor")
    trainer.device_torch = torch.device('cpu')

    res = use_precomputed_control_residuals(trainer, dtype=torch.float32)
    assert res is None


def test_fallback_adapter_invoked_when_precompute_invalid():
    trainer = SimpleNamespace()
    trainer.train_config = SimpleNamespace(controlnet_reroute='precompute')
    trainer.batch = SimpleNamespace()
    trainer.batch.control_residuals = (torch.ones(1, 3, 8, 8), "bad")
    trainer.device_torch = torch.device('cpu')

    called = {'flag': False}
    def adapter_fn(x):
        called['flag'] = True
        return [torch.zeros(1, 3, 8, 8)]

    trainer.adapter = adapter_fn

    pre = use_precomputed_control_residuals(trainer, dtype=torch.float32)
    assert pre is None

    # simulate SDTrainer fallback by calling adapter
    trainer.adapter(None)
    assert called['flag'] is True

