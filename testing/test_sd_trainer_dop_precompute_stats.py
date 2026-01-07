import os
import tempfile
from PIL import Image
from types import SimpleNamespace

from toolkit.config_modules import DatasetConfig
from toolkit.prompt_utils import PromptEmbeds
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


def test_dop_precompute_stats(monkeypatch):
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # minimal sd stub with encode_prompt
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

    # stub encode_prompt returning PromptEmbeds
    def fake_encode_prompt(prompt, **k):
        return PromptEmbeds(__import__('torch').zeros((1, 16, 8)))

    trainer.sd.encode_prompt = fake_encode_prompt

    with tempfile.TemporaryDirectory() as td:
        # make two images
        for i in range(2):
            Image.new('RGB', (8, 8), (255, 255, 255)).save(os.path.join(td, f'{i}.png'))

        trainer.dataset_configs = [DatasetConfig(folder_path=td)]
        trainer.datasets = None
        trainer.data_loader = None

        trainer.train_config.diff_output_preservation = True
        trainer.is_caching_text_embeddings = True

        # try building dataloader directly to see if it raises
        from toolkit.data_loader import get_dataloader_from_datasets
        try:
            dl = get_dataloader_from_datasets(trainer.dataset_configs, trainer.train_config.batch_size, trainer.sd)
        except Exception as e:
            raise RuntimeError(f"Direct dataloader construction failed: {e}") from e

        trainer.hook_before_train_loop()

        assert hasattr(trainer, 'dop_cache_stats')
        stats = trainer.dop_cache_stats
        assert stats['total_files'] == 2
        assert stats['created'] == 2
        assert stats['failed'] == 0
