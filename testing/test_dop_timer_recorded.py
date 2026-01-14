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
        # return an object that has .to and acts like a tensor
        t = self.tensor.repeat(bsize, 1)
        return t
    def to(self, device, dtype=None):
        return self.tensor


def test_dop_timer_recorded(monkeypatch):
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # enable DOP and prepare embeddings
    trainer.train_config.diff_output_preservation = True
    trainer.train_config.diff_output_preservation_after_steps = 0
    trainer.train_config.diff_output_preservation_every = 1  # full-res schedule (compat)

    # set a small dummy prior_pred and noisy_latents
    prior_pred = torch.zeros((1, 4))
    noisy_latents = torch.zeros((1, 4))
    timesteps = torch.tensor([0])
    unconditional_embeds = None
    batch = None

    # set prepared preservation embeds
    trainer.diff_output_preservation_embeds = DummyEmbeds(torch.zeros((1,4)))

    # monkeypatch predict_noise to return a tensor
    def fake_predict_noise(**kwargs):
        return torch.zeros_like(prior_pred)
    monkeypatch.setattr(trainer, 'predict_noise', fake_predict_noise)

    # call the helper
    pred = trainer._run_preservation_forward(noisy_latents, timesteps, trainer.diff_output_preservation_embeds, unconditional_embeds, batch, {}, torch.float32, prior_pred)

    # timer should have recorded 'dop_predict'
    assert 'dop_predict' in trainer.timer.timers
    assert pred is not None
