import tempfile
import shutil
import os
from PIL import Image
import torch
from toolkit.config_modules import DatasetConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO, DataLoaderBatchDTO
from extensions_built_in.sd_trainer.SDTrainer import use_precomputed_control_residuals


def make_image(path, size=(32, 32), color=(128, 128, 128)):
    img = Image.new('RGB', size, color)
    img.save(path)


def test_precomputed_residuals_are_loaded_and_used(tmp_path):
    # create dataset folder and image
    dataset_root = tmp_path
    img_path = dataset_root / 'image_0001.jpg'
    make_image(str(img_path))

    # create residuals folder
    residuals_dir = dataset_root / 'residuals'
    residuals_dir.mkdir()

    # create a small tuple of per-scale tensors and save
    residuals = (torch.ones(1, 3, 8, 8), torch.ones(1, 6, 4, 4))
    residuals_file = residuals_dir / 'image_0001_residuals.pt'
    torch.save(residuals, str(residuals_file))

    # dataset config pointing to residuals folder
    ds = DatasetConfig(control_residuals_path=str(residuals_dir))

    # create file item, then a batch
    fi = FileItemDTO(path=str(img_path), dataset_config=ds, dataset_root=str(dataset_root))
    # ensure tensor exists so DataLoaderBatchDTO concatenates tensors
    fi.tensor = torch.zeros(3, 32, 32)
    assert getattr(fi, 'control_residuals', None) is not None
    assert isinstance(fi.control_residuals, tuple)

    batch = DataLoaderBatchDTO(file_items=[fi])
    assert getattr(batch, 'control_residuals', None) is not None
    assert isinstance(batch.control_residuals, tuple)
    assert batch.control_residuals[0].shape[0] == 1  # batch dim

    # trainer helper should pick them up when configured
    trainer = type('T', (), {})()
    trainer.train_config = type('C', (), {'controlnet_reroute': 'precompute'})
    trainer.batch = batch
    trainer.device_torch = torch.device('cpu')

    pre = use_precomputed_control_residuals(trainer, dtype=torch.float32)
    assert pre is not None
    assert len(pre) == 2
    assert pre[0].shape == (1, 3, 8, 8)
