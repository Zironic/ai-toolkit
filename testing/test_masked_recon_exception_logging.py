import torch
from types import SimpleNamespace
import pytest

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


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


def test_masked_recon_exception_is_logged(monkeypatch):
    # force the build_control_mask helper to raise
    def boom(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr('toolkit.masked_recon.build_control_mask', boom)

    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # Minimal SD with a working vae.decode
    class FakeVAE:
        def decode(self, latents):
            b, c, h, w = latents.shape
            return SimpleNamespace(sample=torch.rand((b, 3, h, w), dtype=latents.dtype))

    trainer.sd = SimpleNamespace()
    trainer.sd.vae = FakeVAE()

    # configure masked recon
    trainer.train_config = SimpleNamespace(
        masked_recon_weight=1.0,
        masked_recon_type='control',
        masked_recon_control_threshold=0.05,
        masked_recon_control_dilate=3,
        masked_recon_control_dilate_auto=True,
        masked_recon_control_dilate_scale_factor=2.0,
        masked_recon_control_blur=3,
        masked_recon_control_ref_max_size=1024,
        mask_preview_enabled=False
    )

    # dummy batch with control tensor
    batch = SimpleNamespace()
    batch.control_tensor = torch.zeros((1, 3, 16, 16))

    noisy_latents = torch.randn((1, 4, 16, 16), requires_grad=True)
    imgs = torch.rand((1, 3, 16, 16))

    # capture print_acc calls
    calls = []
    monkeypatch.setattr('toolkit.print.print_acc', lambda msg: calls.append(msg))

    loss, mloss = trainer._compute_and_apply_masked_recon_loss(torch.tensor(0.0), noisy_latents, imgs, batch, torch.float32)

    # masked_recon helper failure should be caught and logged by trainer; function should return unchanged loss
    assert torch.allclose(loss, torch.tensor(0.0))
    assert mloss is None
    assert any('[MASKED_RECON]' in str(c) for c in calls)
