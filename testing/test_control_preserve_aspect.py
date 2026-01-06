import os
import torch
from PIL import Image
import types
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.config_modules import DatasetConfig


def test_load_control_preserves_aspect(tmp_path):
    # create non-square control image
    img_path = tmp_path / "ctrl.png"
    Image.new('RGB', (320, 512), color=(123, 123, 123)).save(img_path)

    cfg = DatasetConfig(dataset_path=str(tmp_path))
    cfg.control_size = 256

    fi = FileItemDTO(path=str(img_path), dataset_config=cfg)
    # ensure we take the non-full-size branch
    fi.full_size_control_images = False

    # call loader
    fi.load_control_image()
    assert fi.control_tensor is not None
    tensor = fi.control_tensor
    # tensor shape: (C, H, W)
    assert len(tensor.shape) == 3
    C, H, W = tensor.shape
    assert max(H, W) == 256, f"long side should be 256 but got H={H}, W={W}"


def test_precompute_canny_preserves_aspect(tmp_path):
    from tools.precompute_control import precompute_dataset

    img_path = tmp_path / "img1.png"
    Image.new('RGB', (400, 200), color=(73, 109, 137)).save(img_path)

    manifest = precompute_dataset(tmp_path, out_dir=tmp_path / 'canny', control_size=300, overwrite=True)
    # ensure output file exists and is non-square with long side == 300
    out = tmp_path / 'canny' / 'img1_canny.png'
    assert out.exists()
    img = Image.open(out)
    w, h = img.size
    assert max(w, h) == 300
