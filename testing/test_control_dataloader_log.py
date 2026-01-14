import os
import types
from PIL import Image

from toolkit.dataloader_mixins import ControlFileItemDTOMixin


class DummyBase:
    pass


class DummyDatasetConfig:
    def __init__(self, control_path):
        self.control_path = control_path
        self.full_size_control_images = False
        self.control_transparent_color = (0, 0, 0)
        self.buckets = True


class TestFileItem(DummyBase, ControlFileItemDTOMixin):
    def __init__(self, sd, dataset_config, path):
        # set minimal attributes used by load_control_image
        self.sd = sd
        self.dataset_config = dataset_config
        self.control_path = None
        self.full_size_control_images = False
        self.aug_replay_spatial_transforms = False
        self.flip_x = False
        self.flip_y = False
        self.scale_to_width = 256
        self.scale_to_height = 1032
        self.crop_x = 0
        self.crop_y = 0
        self.crop_width = 256
        self.crop_height = 1024
        self.path = path
        self.control_tensor = None
        self.control_tensor_list = None
        self.use_raw_control_images = False


def _create_rgb_png(path):
    img = Image.new('RGB', (300, 300), color=(128, 128, 128))
    img.save(path, 'PNG')


def test_control_log_final_width_x_height(tmp_path, capsys):
    d = tmp_path / "ctrl"
    d.mkdir()
    img_path = os.path.join(str(d), "img1.png")
    _create_rgb_png(img_path)

    dataset_config = DummyDatasetConfig(control_path=str(d))

    sd = types.SimpleNamespace(controlnet=types.SimpleNamespace(control_in_dim=3))

    item = TestFileItem(sd=sd, dataset_config=dataset_config, path='/some/place/img1.jpg')
    item.control_path = img_path
    item.full_size_control_images = False

    item.load_control_image()

    captured = capsys.readouterr()
    assert "final=(256x1024)" in captured.out, f"Unexpected log output: {captured.out}"
