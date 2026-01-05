import os
from PIL import Image
import pytest
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset


class SDStub:
    def __init__(self):
        # used in FileItemDTO and dataset init
        import torch
        self.use_raw_control_images = False
        self._encode_control_in_text_embeddings = False
        self.is_xl = False
        self.is_vega = False
        self.is_ssd = False
        self.device = torch.device('cpu')

    def get_bucket_divisibility(self):
        return 8

    @property
    def encode_control_in_text_embeddings(self):
        return self._encode_control_in_text_embeddings

    def set_device_state_preset(self, preset):
        # no-op for tests
        pass

    def restore_device_state(self):
        # no-op for tests
        pass


def make_image(path):
    img = Image.new('RGB', (256, 256), color=(255, 0, 0))
    img.save(path)


def test_precompute_creates_cache_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img_path = images_dir / "img1.jpg"
    make_image(img_path)

    cfg = DatasetConfig(folder_path=str(images_dir), dataset_path=str(images_dir))
    # signal that this dataset expects control conditioning
    cfg.controls = ['pose']
    # explicitly disable on-the-fly so we test the precompute path
    cfg.generate_control_on_the_fly = False
    cfg.control_precompute_control = True
    cfg.control_cache_path = None

    sd = SDStub()

    ds = AiToolkitDataset(cfg, batch_size=1, sd=sd)

    assert hasattr(cfg, 'control_cache_path') and cfg.control_cache_path is not None
    assert os.path.isdir(cfg.control_cache_path)


def test_constructor_precompute_flag_from_kwargs_creates_cache_dir(tmp_path):
    images_dir = tmp_path / "images2"
    images_dir.mkdir()
    img_path = images_dir / "img1.jpg"
    make_image(img_path)

    # pass precompute flag via constructor kwargs (simulating job JSON)
    cfg = DatasetConfig(folder_path=str(images_dir), dataset_path=str(images_dir), control_precompute_control=True)
    cfg.controls = ['pose']
    cfg.generate_control_on_the_fly = False
    cfg.control_cache_path = None

    sd = SDStub()
    ds = AiToolkitDataset(cfg, batch_size=1, sd=sd)

    assert hasattr(cfg, 'control_cache_path') and cfg.control_cache_path is not None
    assert os.path.isdir(cfg.control_cache_path)


def test_default_generate_on_the_fly_is_true(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img_path = images_dir / "img1.jpg"
    make_image(img_path)

    cfg = DatasetConfig(folder_path=str(images_dir), dataset_path=str(images_dir))
    # signal that this dataset expects control conditioning but do not set generate_control_on_the_fly
    cfg.controls = ['pose']

    sd = SDStub()

    # Should not raise and default should be True
    ds = AiToolkitDataset(cfg, batch_size=1, sd=sd)
    assert cfg.generate_control_on_the_fly is True


def test_missing_cache_raises(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    img_path = images_dir / "img1.jpg"
    make_image(img_path)

    cfg = DatasetConfig(folder_path=str(images_dir), dataset_path=str(images_dir))
    cfg.controls = ['pose']
    cfg.generate_control_on_the_fly = False
    cfg.control_precompute_control = False
    cfg.control_cache_path = None

    sd = SDStub()

    with pytest.raises(RuntimeError) as exc:
        AiToolkitDataset(cfg, batch_size=1, sd=sd)

    assert 'control_cache_path is not set' in str(exc.value)
