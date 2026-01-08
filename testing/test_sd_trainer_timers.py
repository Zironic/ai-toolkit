import pytest
from types import SimpleNamespace
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
import torch


def make_job_and_cfg():
    job = SimpleNamespace()
    job.training_folder = "./tmp_training"
    job.raw_config = {}
    job.name = 'test_job'
    job.log_dir = None
    job.training_seed = None
    job.meta = {}
    cfg = {}
    cfg['model'] = {'name_or_path': 'dummy_model'}
    cfg['train'] = {'steps': 1}
    return job, cfg


def test_preservation_backward_and_cpu_transfer(monkeypatch):
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # ensure timer is reset
    try:
        trainer.timer.reset()
    except Exception:
        pass

    # Create preds that require grad so backward will be invoked
    preservation_pred = torch.zeros((1, 4), requires_grad=True)
    prior_pred = torch.zeros((1, 4), requires_grad=True)

    # Stub out accelerator.backward to avoid actual autograd complexity in this unit test
    class DummyAcc:
        def backward(self, x):
            return None

    trainer.accelerator = DummyAcc()

    # Call the preservation loss helper
    res = trainer._compute_and_apply_preservation_loss(preservation_pred, prior_pred, 1.0)

    assert 'preservation_backward' in trainer.timer.timers
    assert 'cpu_transfer' in trainer.timer.timers
    assert res is not None
