import torch
import types
import pytest

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummyFile:
    def __init__(self, path, contexts=None):
        self.path = path
        self._preencoded_zimage_control_contexts = contexts


class DummyBatch:
    def __init__(self, file_items, tensor=None):
        self.file_items = file_items
        self.tensor = tensor


def test_collect_preencoded_logs_missing_entries(capsys):
    # One file missing contexts should cause collector to return None and print diagnostics
    fi1 = DummyFile('a.jpg', contexts={512: torch.zeros((3, 1, 512, 512))})
    fi2 = DummyFile('b.jpg', contexts=None)
    batch = DummyBatch([fi1, fi2], tensor=torch.zeros((2, 33, 1, 512, 512)))

    # Call the unbound method with a light-weight self
    trainer_like = types.SimpleNamespace()
    res = SDTrainer._collect_preencoded_zimage_context_for_batch(trainer_like, batch)
    assert res is None
    captured = capsys.readouterr()
    assert '[PRECOMPUTE] precompute not usable for batch' in captured.out or '[PRECOMPUTE] precompute not usable for batch' in captured.err
    assert 'b.jpg: missing _preencoded_zimage_control_contexts' in captured.out or 'b.jpg: missing _preencoded_zimage_control_contexts' in captured.err


def test_collect_preencoded_success_returns_tensor():
    # Both files have matching precomputed size -> should return stacked tensor
    ctx = torch.zeros((3, 1, 128, 128))
    fi1 = DummyFile('a.jpg', contexts={128: ctx})
    fi2 = DummyFile('b.jpg', contexts={128: ctx})
    batch = DummyBatch([fi1, fi2], tensor=torch.zeros((2, 33, 1, 128, 128)))

    trainer_like = types.SimpleNamespace()
    res = SDTrainer._collect_preencoded_zimage_context_for_batch(trainer_like, batch)
    assert isinstance(res, torch.Tensor)
    assert res.shape == (2, 3, 1, 128, 128)
