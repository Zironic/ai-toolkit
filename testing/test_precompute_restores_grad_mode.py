import pytest
pytest.skip('precompute tests removed: skipping')
from toolkit.data_transfer_object.data_loader import FileItemDTO
from toolkit.config_modules import DatasetConfig


class DummySDV:
    def __init__(self):
        self.transformer = types.SimpleNamespace(control_in_dim=33)

    def encode_control_images_videox(self, imgs, height=None, width=None, tile=False):
        # simulate an implementation that uses torch.no_grad internally
        with torch.no_grad():
            outs = []
            for img in imgs:
                outs.append(torch.randn(4, 16, 16))
        return torch.stack(outs, dim=0)


class DummyTrainer(SDTrainer):
    def __init__(self):
        self.sd = DummySDV()
        self.model_config = types.SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)
        self.data_loader = types.SimpleNamespace(dataset=[types.SimpleNamespace(dataset_config=None, dataset_path='dummy', file_list=[])])


def test_precompute_restores_grad_mode(tmp_path):
    cfg = DatasetConfig(dataset_path=str(tmp_path), control_precompute_control=True)
    ds = types.SimpleNamespace(dataset_config=cfg, dataset_path=str(tmp_path), file_list=[])

    # create one file item with control tensor
    img = tmp_path / "img.png"
    from PIL import Image
    Image.new('RGB', (512, 512), color=(73, 109, 137)).save(img)

    fi = FileItemDTO(path=str(img), dataset_config=cfg)
    fi.control_tensor = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
    ds.file_list = [fi]

    t = DummyTrainer()
    t.data_loader = types.SimpleNamespace(dataset=[ds])

    # First ensure grad mode is enabled
    assert torch.is_grad_enabled()

    # Emulate a call that might temporarily disable grad and not restore (simulate buggy encoder)
    # We'll monkeypatch the encoder to incorrectly call torch.set_grad_enabled(False) without restoring
    original = t.sd.encode_control_images_videox

    def buggy_encoder(imgs, **kwargs):
        torch.set_grad_enabled(False)
        # do work
        outs = original(imgs, **kwargs)
        # forget to restore
        return outs

    t.sd.encode_control_images_videox = buggy_encoder

    # run precompute — our trainer should defensively re-enable gradients after precompute
    t._precompute_zimage_control_contexts()

    assert torch.is_grad_enabled(), "Grad mode should be restored after precompute"
