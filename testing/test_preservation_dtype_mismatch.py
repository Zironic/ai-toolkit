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


class DummyEmbeds:
    def __init__(self, tensor):
        self.tensor = tensor
    def expand_to_batch(self, bsize):
        t = self.tensor.repeat(bsize, 1)
        return t
    def to(self, device, dtype=None):
        # return as tensor with given dtype if dtype provided
        if dtype is not None and isinstance(dtype, torch.dtype):
            return self.tensor.to(dtype=dtype)
        return self.tensor


def test_preservation_loss_with_dtype_mismatch(monkeypatch):
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    trainer.train_config.diff_output_preservation = True
    trainer.train_config.diff_output_preservation_after_steps = 0
    trainer.train_config.diff_output_preservation_every = 1  # remains as full-res scheduling value

    # set prepared preservation embeds
    trainer.diff_output_preservation_embeds = DummyEmbeds(torch.zeros((1,4)))

    # create preservation_pred float32 and prior_pred bfloat16 to simulate mismatch
    preservation_pred = torch.ones((1,4), dtype=torch.float32)
    prior_pred = torch.zeros((1,4), dtype=torch.bfloat16)

    # monkeypatch predict_noise to return preservation_pred
    def fake_predict_noise(**kwargs):
        return preservation_pred
    monkeypatch.setattr(trainer, 'predict_noise', fake_predict_noise)

    # run helper and compute loss
    pl = trainer._compute_and_apply_preservation_loss(preservation_pred, prior_pred, multiplier=1.0)

    assert pl is not None
    assert hasattr(trainer, '_last_preservation_loss')
    assert trainer._last_preservation_loss is not None