import copy
import io
from pathlib import Path
from PIL import Image

from toolkit.data_transfer_object.data_loader import FileItemDTO


def test_fileitem_deepcopy_with_non_picklable_dataset_config(tmp_path):
    # create a small image
    img_path = tmp_path / "img.png"
    Image.new("RGB", (16, 16)).save(img_path)

    # create a dummy dataset config with attributes used by FileItemDTO
    class DummyConfig:
        num_frames = 1
        fast_image_size = False
        augments = []
        loss_multiplier = 1.0
        network_weight = 1.0
        is_reg = False
        prior_reg = False
        scale = 1.0
        flip_x = False
        flip_y = False
        standardize_images = False
        buckets = False

    dummy = DummyConfig()
    # add a non-picklable attribute (open file)
    logfile = tmp_path / "log.txt"
    dummy.log = open(logfile, "a")

    try:
        fi = FileItemDTO(path=str(img_path), dataset_config=dummy)
        # should not raise
        new_fi = copy.deepcopy(fi)
        assert new_fi.path == fi.path
    finally:
        dummy.log.close()
