import os
import tempfile
import types
from PIL import Image
import torch

from toolkit.dataloader_mixins import ControlFileItemDTOMixin


class DummyBase:
    def __init__(self, *args, **kwargs):
        # accept arbitrary args because ControlFileItemDTOMixin calls super().__init__(**kwargs)
        pass


class DummyDatasetConfig:
    def __init__(self, control_path, full_size_control_images=False):
        self.control_path = control_path
        self.full_size_control_images = full_size_control_images
        self.control_transparent_color = (0, 0, 0)
        self.buckets = False


class TestFileItem(DummyBase, ControlFileItemDTOMixin):
    def __init__(self, sd, dataset_config, path):
        # Do NOT call the mixin __init__ (it expects a full FileItemDTO hierarchy). Instead,
        # set the minimal attributes required by load_control_image directly for testing.
        self.sd = sd
        self.dataset_config = dataset_config
        self.control_path = None
        self.full_size_control_images = False
        self.aug_replay_spatial_transforms = False
        self.flip_x = False
        self.flip_y = False
        self.scale_to_width = 512
        self.scale_to_height = 512
        self.crop_x = 0
        self.crop_y = 0
        self.crop_width = 512
        self.crop_height = 512
        self.path = path
        self.control_tensor = None
        self.control_tensor_list = None


def _create_rgb_png(path):
    img = Image.new('RGB', (32, 32), color=(128, 128, 128))
    img.save(path, 'PNG')


def test_pad_alpha_when_model_expects_4_channels(tmp_path):
    # create control image file
    d = tmp_path / "ctrl"
    d.mkdir()
    img_path = os.path.join(str(d), "img1.png")
    _create_rgb_png(img_path)

    dataset_config = DummyDatasetConfig(control_path=str(d))

    # fake sd with controlnet expecting 4 channels
    sd = types.SimpleNamespace(controlnet=types.SimpleNamespace(control_in_dim=4))

    item = TestFileItem(sd=sd, dataset_config=dataset_config, path='/some/place/img1.jpg')
    # set attributes used by loader
    item.control_path = img_path
    item.full_size_control_images = False

    item.load_control_image()

    assert item.control_tensor is not None, "control_tensor should be set"
    assert item.control_tensor.shape[0] == 4, f"Expected 4 channels after padding, got {item.control_tensor.shape[0]}"


def test_no_pad_when_model_expects_3_channels(tmp_path):
    d = tmp_path / "ctrl2"
    d.mkdir()
    img_path = os.path.join(str(d), "img1.png")
    _create_rgb_png(img_path)

    dataset_config = DummyDatasetConfig(control_path=str(d))

    sd = types.SimpleNamespace(controlnet=types.SimpleNamespace(control_in_dim=3))

    item = TestFileItem(sd=sd, dataset_config=dataset_config, path='/some/place/img1.jpg')
    item.control_path = img_path
    item.full_size_control_images = False

    item.load_control_image()

    assert item.control_tensor is not None
    assert item.control_tensor.shape[0] == 3, f"Expected 3 channels, got {item.control_tensor.shape[0]}"