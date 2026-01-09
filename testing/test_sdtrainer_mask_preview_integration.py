import tempfile
import shutil
import os
import types
import glob
from PIL import Image

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


def test_sdtrainer_generate_mask_previews(monkeypatch, tmp_path):
    # Create a simple dataset with one image and one control image
    base_img_path = tmp_path / 'img1.png'
    ctrl_img_path = tmp_path / 'img1_ctrl.png'
    img = Image.new('RGB', (32, 32), color=(255, 255, 255))
    img.save(str(base_img_path))
    ctrl = Image.new('RGB', (32, 32), color=(0, 0, 0))
    # draw a white stick
    for y in range(4, 28):
        for x in range(15, 17):
            ctrl.putpixel((x, y), (255, 255, 255))
    ctrl.save(str(ctrl_img_path))

    file_item = types.SimpleNamespace(path=str(base_img_path), control_path=str(ctrl_img_path), crop_height=32, crop_width=32)
    dataset = types.SimpleNamespace(file_list=[file_item])

    # Create a minimal trainer instance without invoking heavy init
    trainer = SDTrainer.__new__(SDTrainer)
    trainer.train_config = types.SimpleNamespace()
    trainer.train_config.mask_preview_enabled = True
    trainer.train_config.mask_preview_save_path = str(tmp_path / '{job_name}' / 'masks')
    trainer.train_config.mask_preview_overwrite = True
    trainer.train_config.mask_preview_overlay = True
    trainer.data_loader = object()
    trainer.job = types.SimpleNamespace(name='testjob')
    trainer.sd = None

    # Patch get_dataloader_datasets to return our dataset list
    monkeypatch.setattr('toolkit.data_loader.get_dataloader_datasets', lambda dl: [dataset])

    # Run the helper
    trainer.generate_mask_previews_if_enabled()

    save_dir = tmp_path / 'testjob' / 'masks'
    mask_files = list(save_dir.glob('*.png'))
    idx = save_dir / 'index.json'

    assert save_dir.exists()
    assert idx.exists()
    # ensure at least one mask file and an overlay exists
    assert any(p.name.endswith('_mask.png') for p in mask_files)
    assert any(p.name.endswith('_overlay.png') for p in mask_files)
