import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


def test_compute_adapter_multiplier_deterministic_cpu():
    # Ensure determinism when sampling via torch RNG on CPU
    torch.manual_seed(12345)
    v1 = SDTrainer.compute_adapter_multiplier(False, True, 'cpu', torch.float32)
    torch.manual_seed(12345)
    v2 = SDTrainer.compute_adapter_multiplier(False, True, 'cpu', torch.float32)
    assert isinstance(v1, float)
    assert v1 == v2


def test_compute_adapter_multiplier_t2i_returns_one():
    v = SDTrainer.compute_adapter_multiplier(True, False, 'cpu', torch.float32)
    assert v == 1.0
