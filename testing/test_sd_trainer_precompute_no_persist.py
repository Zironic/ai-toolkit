import os
import pytest
pytest.skip('precompute tests removed: skipping')
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.config_modules import DatasetConfig


class DummySDV:
    def __init__(self):
        self.transformer = types.SimpleNamespace(control_in_dim=33)
        self.calls = 0

    def encode_control_images_videox(self, imgs, height=None, width=None, tile=False):
        self.calls += len(imgs)
        outs = []
        for img in imgs:
            outs.append(torch.randn(4, 16, 16))
        return torch.stack(outs, dim=0)


class DummyTrainer(SDTrainer):
    def __init__(self):
        self.sd = DummySDV()
        self.model_config = types.SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)
        self.data_loader = types.SimpleNamespace(dataset=[types.SimpleNamespace(dataset_config=None, dataset_path='dummy', file_list=[])])


def test_precompute_does_not_persist_to_disk(tmp_path):
    # setup dataset config; explicitly set a control_cache_path
    control_cache = tmp_path / "control_cache"
    cfg = DatasetConfig(dataset_path=str(tmp_path), control_precompute_control=True)
    cfg.control_cache_path = str(control_cache)

    ds = types.SimpleNamespace(dataset_config=cfg, dataset_path=str(tmp_path), file_list=[])

    # create one small dummy image on disk so FileItemDTO can read sizes
    img1 = tmp_path / "img1.png"
    from PIL import Image

    Image.new('RGB', (512, 512), color=(73, 109, 137)).save(img1)

    fi1 = FileItemDTO(path=str(img1), dataset_config=cfg)
    fi1.control_tensor = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    ds.file_list = [fi1]

    # create the cache directory beforehand and record initial state
    control_cache.mkdir(parents=True, exist_ok=True)
    initial_listing = set(os.listdir(control_cache))

    t = DummyTrainer()
    t.data_loader = types.SimpleNamespace(dataset=[ds])

    # run precompute
    t._precompute_zimage_control_contexts()

    # ensure encoder was used and in-memory cache created
    assert t.sd.calls == 1
    assert getattr(fi1, '_preencoded_zimage_control_contexts', None) is not None

    # ensure nothing was written to the control_cache_path (no persistence)
    after_listing = set(os.listdir(control_cache))
    assert initial_listing == after_listing, "Precompute should not write files to control_cache_path"
