import os
import types
import torch
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.config_modules import DatasetConfig
from toolkit.precompute_cache import clear


def test_save_and_load_multiple_resolutions(tmp_path):
    cfg = DatasetConfig(dataset_path=str(tmp_path), control_precompute_control=True)
    cfg.cache_control_contexts_to_disk = True

    # create one small dummy image on disk so FileItemDTO can read sizes
    img1 = tmp_path / "img1.png"
    from PIL import Image

    Image.new('RGB', (768, 768), color=(73, 109, 137)).save(img1)

    fi = FileItemDTO(path=str(img1), dataset_config=cfg)

    # Simulate precomputed contexts for two sizes
    ctx_512 = torch.randn(33, 512 // 8, 512 // 8)
    ctx_768 = torch.randn(33, 768 // 8, 768 // 8)
    contexts = {512: ctx_512, 768: ctx_768}

    # save
    path = fi.save_control_contexts(contexts)
    assert os.path.exists(path), f"Expected saved file at {path}"

    # load via the same file item
    loaded = fi.load_control_contexts()
    assert loaded is not None
    assert 512 in loaded and 768 in loaded
    assert loaded[512].shape == ctx_512.shape
    assert loaded[768].shape == ctx_768.shape

    # simulate new process: clear registry and construct a fresh FileItemDTO
    clear()
    fi2 = FileItemDTO(path=str(img1), dataset_config=cfg)
    # constructor should have loaded control contexts from disk
    assert getattr(fi2, '_preencoded_zimage_control_contexts', None) is not None
    ctxs2 = fi2._preencoded_zimage_control_contexts
    assert 512 in ctxs2 and 768 in ctxs2
    assert ctxs2[512].shape == ctx_512.shape
    assert ctxs2[768].shape == ctx_768.shape
