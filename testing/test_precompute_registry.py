import os
import tempfile
from types import SimpleNamespace
from PIL import Image
import torch

from toolkit.precompute_cache import set_preencoded_control_contexts, get_preencoded_control_contexts, clear
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets, get_dataloader_datasets
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


def test_precompute_registry_loading(monkeypatch):
    monkeypatch.setattr('toolkit.accelerator.get_accelerator', lambda: DummyAccelerator())
    job, cfg = make_job_and_cfg()
    trainer = SDTrainer(0, job, cfg)

    # minimal sd stub
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

    with tempfile.TemporaryDirectory() as td:
        # make an image
        img_path = os.path.join(td, 'img.png')
        Image.new('RGB', (8, 8), (255, 255, 255)).save(img_path)

        # Create a fake precomputed contexts dict and register it
        contexts = {512: torch.randn((16, 8, 8))}
        set_preencoded_control_contexts(img_path, contexts)

        # Create a new FileItem via dataloader and ensure it picks up cached contexts
        cfg_obj = DatasetConfig(folder_path=td, dataset_path=td)
        dl = get_dataloader_from_datasets([cfg_obj], batch_size=1, sd=trainer.sd)
        dsets = get_dataloader_datasets(dl)
        fi = dsets[0].file_list[0]

        # In normal usage, the collector will attempt to load from registry when contexts missing
        from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
        batch = DataLoaderBatchDTO(file_items=[fi])

        # call trainer helper to collect preencoded contexts for batch
        res = trainer._collect_preencoded_zimage_context_for_batch(batch)
        assert res is not None, "Collector should find registry entry and return tensor"
        # ensure loaded tensor has expected shape
        assert res.shape[0] == 1
        # cleanup registry
        clear()