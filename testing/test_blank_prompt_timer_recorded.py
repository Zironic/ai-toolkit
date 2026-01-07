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
        return self.tensor


def test_blank_prompt_timer_recorded(monkeypatch):
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # enable blank prompt preservation
    trainer.train_config.blank_prompt_preservation = True

    # set a small dummy prior_pred and noisy_latents
    prior_pred = torch.zeros((1, 4))
    noisy_latents = torch.zeros((1, 4))
    timesteps = torch.tensor([0])

    # set prepared preservation embeds (blank)
    preservation_embeds = DummyEmbeds(torch.zeros((1, 4)))

    # monkeypatch predict_noise to return a tensor
    def fake_predict_noise(**kwargs):
        return torch.zeros_like(prior_pred)

    monkeypatch.setattr(trainer, 'predict_noise', fake_predict_noise)

    # call the helper and indicate this is a blank preservation
    pred = trainer._run_preservation_forward(
        noisy_latents,
        timesteps,
        preservation_embeds,
        None,
        None,
        {},
        'float32',
        prior_pred,
        preservation_resolution=None,
        preservation_kind='blank',
    )

    # timer should have recorded 'blank_predict'
    assert 'blank_predict' in trainer.timer.timers
    assert pred is not None
