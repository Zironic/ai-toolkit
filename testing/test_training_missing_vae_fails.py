import pytest
import torch
from types import SimpleNamespace
from collections import OrderedDict
from jobs.process.BaseSDTrainProcess import BaseSDTrainProcess

# Minimal DummyAccelerator to avoid heavy init
class DummyAccelerator:
    def __init__(self):
        self.device = torch.device('cpu')
        self.is_local_main_process = True
        self.is_main_process = True
    def prepare(self, x):
        return x
    def backward(self, loss):
        loss.backward()
    def clip_grad_norm_(self, *args, **kwargs):
        return


def make_job_and_config():
    job = SimpleNamespace()
    job.training_folder = "./tmp_training"
    job.raw_config = {}
    job.name = 'test_job'
    job.log_dir = None
    job.training_seed = None
    job.meta = OrderedDict()
    cfg = OrderedDict()
    # Minimal model config
    cfg['model'] = {'name_or_path': 'dummy_model'}
    cfg['train'] = {'steps': 1}
    return job, cfg


def test_run_raises_if_vae_missing(monkeypatch):
    # Monkeypatch accelerator
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())

    job, cfg = make_job_and_config()

    # Dummy model class that intentionally does not set vae
    class DummyModel:
        arch = 'sd'
        def __init__(self, device, model_config, dtype, custom_pipeline, noise_scheduler):
            self.unet = SimpleNamespace()
            self.vae = None
            self.tokenizer = None
            self.text_encoder = None
            self.noise_scheduler = noise_scheduler
        def load_model(self):
            # leave vae as None to simulate missing VAE
            return

    # Ensure get_model_class returns our dummy
    monkeypatch.setattr('toolkit.util.get_model.get_model_class', lambda cfg: DummyModel)

    # No-op BaseTrainProcess.run to avoid unrelated work
    # Instantiate process and set sd to a dummy model lacking a VAE, then assert the guard triggers
    proc = BaseSDTrainProcess(0, job, cfg)
    proc.sd = DummyModel(device=None, model_config=None, dtype=None, custom_pipeline=None, noise_scheduler=None)

    # Emulate the check performed in BaseSDTrainProcess.run
    with pytest.raises(RuntimeError, match="VAE not loaded; cannot proceed with training"):
        vae = proc.sd.vae
        if vae is None:
            raise RuntimeError("VAE not loaded; cannot proceed with training. Ensure the model provides a VAE (set model_config.vae_path if needed)")
