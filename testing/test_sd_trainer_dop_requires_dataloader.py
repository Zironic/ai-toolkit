import pytest
from types import SimpleNamespace
from toolkit.config_modules import ModelConfig
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummyAccelerator:
    def __init__(self):
        self.device = None
        self.is_local_main_process = True
        self.is_main_process = True
    def prepare(self, x):
        return x
    def backward(self, loss):
        loss.backward()
    def clip_grad_norm_(self, *args, **kwargs):
        return


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


def test_dop_caching_requires_dataloader(monkeypatch):
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # minimal sd stub used in hook
    trainer.sd = SimpleNamespace()
    trainer.sd.unet = SimpleNamespace(to=lambda *a, **k: None)
    trainer.sd.device_torch = 'cpu'
    trainer.sd.torch_dtype = None
    trainer.sd.encode_control_in_text_embeddings = False
    trainer.sd.has_multiple_control_images = False
    trainer.sd.vae = None
    trainer.sd.text_encoder = None
    trainer.sd.refiner_unet = None
    trainer.sd.noise_scheduler = None

    # simulate enabling both features
    trainer.train_config.diff_output_preservation = True
    trainer.is_caching_text_embeddings = True

    # ensure no dataloader present
    trainer.data_loader = None

    with pytest.raises(RuntimeError) as exc:
        trainer.hook_before_train_loop()
    assert 'Differential Output Preservation with cached text embeddings requires dataset' in str(exc.value)
