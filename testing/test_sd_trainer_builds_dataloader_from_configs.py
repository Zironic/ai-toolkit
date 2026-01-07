import os
import tempfile
from PIL import Image
from types import SimpleNamespace

import torch

from toolkit.config_modules import DatasetConfig
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


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


def test_hook_builds_dataloader_from_configs(monkeypatch):
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # provide minimal sd attributes
    trainer.sd = SimpleNamespace()
    trainer.sd.unet = SimpleNamespace(to=lambda *a, **k: None)
    trainer.sd.device_torch = 'cpu'
    trainer.sd.torch_dtype = None
    trainer.sd.encode_control_in_text_embeddings = False
    trainer.sd.has_multiple_control_images = False
    trainer.sd.get_bucket_divisibility = lambda : 1
    trainer.sd.encode_prompt = lambda *a, **k: __import__('toolkit').prompt_utils.PromptEmbeds(__import__('torch').zeros((1,16,8)))
    trainer.sd.vae = None
    trainer.sd.text_encoder = None
    trainer.sd.refiner_unet = None
    trainer.sd.noise_scheduler = None

    # create a temporary dataset folder with an image so AiToolkitDataset can be constructed
    with tempfile.TemporaryDirectory() as td:
        img_path = os.path.join(td, 'img.png')
        Image.new('RGB', (8, 8), (255, 255, 255)).save(img_path)

        # Provide raw dataset_configs (as dicts) but leave datasets and data_loader empty
        trainer.dataset_configs = [{'folder_path': td}]
        trainer.datasets = None
        trainer.data_loader = None

        trainer.train_config.diff_output_preservation = True
        trainer.is_caching_text_embeddings = True

            # Sanity check: ensure file exists
            assert os.path.exists(img_path), f"Image missing: {img_path}"
            assert len(os.listdir(td)) > 0, f"Temp directory empty: {td}"
            # Try building dataloader directly to see if an exception is raised in that path
            from toolkit.data_loader import get_dataloader_from_datasets
            try:
                dl = get_dataloader_from_datasets(trainer.dataset_configs, trainer.train_config.batch_size, trainer.sd)
            except Exception as e:
                # Add more debug info
                files = list(os.walk(td))
                raise RuntimeError(f"Direct dataloader construction failed: {e}; td files: {files}") from e

            # Should not raise; it should build a data_loader from dataset_configs
            trainer.hook_before_train_loop()
            assert trainer.data_loader is not None
            # dataloader should have at least one dataset
            assert hasattr(trainer.data_loader, 'dataset')
