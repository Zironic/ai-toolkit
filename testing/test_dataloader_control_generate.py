import tempfile
import os
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms

from toolkit.config_modules import DatasetConfig
from toolkit.data_transfer_object.data_loader import FileItemDTO


def test_canny_generate_after_augmentation(tmp_path: Path):
    # create a simple image
    img_path = tmp_path / "img1.jpg"
    Image.new("RGB", (256, 256), color=(255, 0, 0)).save(img_path)

    ds = DatasetConfig(controls=['canny'], control_generate_on_the_fly=True)

    # create FileItemDTO and process image
    file_item = FileItemDTO(path=str(img_path), dataset_config=ds, size_database={}, dataset_root=str(tmp_path))

    file_item.load_and_process_image(transform=transforms.ToTensor(), only_load_latents=False)

    assert file_item.control_tensor is not None, "control_tensor should be created when generate_on_the_fly is enabled"
    assert isinstance(file_item.control_tensor, torch.Tensor)

    # control must match spatial dims of the image tensor (H,W)
    assert file_item.control_tensor.shape[1] == file_item.tensor.shape[1]
    assert file_item.control_tensor.shape[2] == file_item.tensor.shape[2]
