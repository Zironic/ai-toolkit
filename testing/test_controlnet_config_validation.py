import pytest
from toolkit.config_modules import TrainConfig, validate_configs, ModelConfig, SaveConfig, DatasetConfig


def test_invalid_offload_strategy_raises():
    t = TrainConfig(controlnet_offload_strategy='banana')
    with pytest.raises(ValueError):
        validate_configs(t, ModelConfig(name_or_path='m'), SaveConfig(), [DatasetConfig()])


def test_invalid_residual_storage_raises():
    t = TrainConfig(controlnet_residual_storage='weird')
    with pytest.raises(ValueError):
        validate_configs(t, ModelConfig(name_or_path='m'), SaveConfig(), [DatasetConfig()])


def test_invalid_aux_loss_raises():
    t = TrainConfig(controlnet_aux_loss='not-a-method')
    with pytest.raises(ValueError):
        validate_configs(t, ModelConfig(name_or_path='m'), SaveConfig(), [DatasetConfig()])


def test_precompute_requires_use_raises():
    t = TrainConfig(controlnet_precompute_control=True, controlnet_use=False)
    with pytest.raises(ValueError):
        validate_configs(t, ModelConfig(name_or_path='m'), SaveConfig(), [DatasetConfig()])


def test_valid_values_do_not_raise():
    t = TrainConfig(controlnet_offload_strategy='none', controlnet_residual_storage='gpu', controlnet_aux_loss='edge')
    validate_configs(t, ModelConfig(name_or_path='m'), SaveConfig(), [DatasetConfig()])
