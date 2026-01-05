import torch
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess


def test_format_progress_bar_handles_nested_and_tensors():
    proc = BaseSDTrainProcess.__new__(BaseSDTrainProcess)
    loss_dict = {
        'loss': {'sub': 0.123},
        'acc': 0.5,
        'meta': {'nested': 'x'},
        'tens': torch.tensor(0.2),
        'big': torch.zeros((2, 3))
    }
    s = proc._format_progress_bar(1e-4, loss_dict)
    assert "lr: 1.0e-04" in s
    assert "loss.sub: 1.230e-01" in s
    assert "acc: 5.000e-01" in s
    assert "meta.nested: x" in s
    assert "tens: 2.000e-01" in s or "tens: 2.000e-01" in s
    assert "big: tensor" in s
