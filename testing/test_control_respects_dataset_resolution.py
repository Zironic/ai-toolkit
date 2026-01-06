import os
from PIL import Image
import torch
from toolkit.config_modules import DatasetConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.buckets import get_bucket_for_image_size


def make_temp_image(path, size=(1800, 1200), color=(128, 128, 128)):
    img = Image.new('RGB', size, color=color)
    img.save(path)


def test_control_respects_dataset_resolution(tmp_path):
    # create a sample image
    img_path = tmp_path / "sample.jpg"
    make_temp_image(img_path, size=(1800, 1200))

    for res in (512, 768, 1024):
        dc = DatasetConfig(resolution=res)
        # ensure control_size is not set
        assert getattr(dc, 'control_size', None) is None
        # create file item
        fi = FileItemDTO(path=str(img_path), dataset_config=dc)
        # load control image (should use file path fallback)
        fi.load_control_image()
        assert fi.control_tensor is not None
        # control tensor shape is (C,H,W)
        C, H, W = fi.control_tensor.shape
        w, h = Image.open(img_path).size
        bucket = get_bucket_for_image_size(w, h, resolution=res)
        target_w, target_h = bucket['width'], bucket['height']
        # note: tensor is (C,H,W) so compare H,W
        assert H == target_h and W == target_w, f"Expected {(target_h, target_w)} got {(H, W)} for resolution {res}"
        # channels should be 3 or 4
        assert C in (1, 3, 4)
