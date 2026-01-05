import os
import tempfile
import pytest

from toolkit.dataloader_mixins import BucketsMixin
from toolkit.config_modules import DatasetConfig


class DummyDataset(BucketsMixin):
    def __init__(self, dataset_config):
        super().__init__()
        self.dataset_config = dataset_config
        self.file_list = []
        self.dataset_path = getattr(dataset_config, 'folder_path', 'dummy')


def test_validate_control_dataset_missing_cache_raises():
    cfg = DatasetConfig(control_type='openpose', generate_control_on_the_fly=False, control_cache_path=None)
    ds = DummyDataset(cfg)
    with pytest.raises(RuntimeError):
        ds.validate_control_dataset()


def test_validate_control_dataset_empty_cache_raises(tmp_path):
    cache_dir = tmp_path / "control_cache"
    cache_dir.mkdir()
    cfg = DatasetConfig(control_type='openpose', generate_control_on_the_fly=False, control_cache_path=str(cache_dir))
    ds = DummyDataset(cfg)
    with pytest.raises(RuntimeError):
        ds.validate_control_dataset()


def test_validate_control_dataset_on_the_fly_ok():
    cfg = DatasetConfig(control_type='canny', generate_control_on_the_fly=True)
    ds = DummyDataset(cfg)
    assert ds.validate_control_dataset() is True
