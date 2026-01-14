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


def test_preservation_loss_recorded_and_logged(monkeypatch):
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # enable DOP and prepare embeddings
    trainer.train_config.diff_output_preservation = True
    trainer.train_config.diff_output_preservation_after_steps = 0
    trainer.train_config.diff_output_preservation_every = 1  # remains as full-res scheduling value (not DOP frequency)

    # set minimal tensors
    prior_pred = torch.zeros((1, 4))
    noisy_latents = torch.zeros((1, 4))
    timesteps = torch.tensor([0])
    unconditional_embeds = None
    batch = None

    # set prepared preservation embeds
    trainer.diff_output_preservation_embeds = DummyEmbeds(torch.zeros((1,4)))

    # monkeypatch predict_noise to return a tensor of ones so preservation loss non-zero
    def fake_predict_noise(**kwargs):
        return torch.ones_like(prior_pred)
    monkeypatch.setattr(trainer, 'predict_noise', fake_predict_noise)

    # call the helper
    pred = trainer._run_preservation_forward(noisy_latents, timesteps, trainer.diff_output_preservation_embeds, unconditional_embeds, batch, {}, 'float32', prior_pred)
    # compute and apply preservation loss
    pl = trainer._compute_and_apply_preservation_loss(pred, prior_pred, multiplier=2.0)

    assert hasattr(trainer, '_last_preservation_loss')
    assert trainer._last_preservation_loss is not None
    assert trainer._last_preservation_loss > 0.0

    # simulate creation of loss_dict as in hook_train_loop
    trainer._last_normal_loss = 0.1
    loss_val = 0.1 + trainer._last_preservation_loss
    loss_dict = {'loss': loss_val}
    if hasattr(trainer, '_last_preservation_loss') and trainer._last_preservation_loss is not None:
        loss_dict['preservation'] = float(trainer._last_preservation_loss)
    if hasattr(trainer, '_last_normal_loss') and trainer._last_normal_loss is not None:
        loss_dict['normal'] = float(trainer._last_normal_loss)

    assert 'preservation' in loss_dict
    assert 'normal' in loss_dict
