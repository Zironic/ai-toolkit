import os
from PIL import Image
import tempfile
import torch
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset


class DummySDFull:
    def __init__(self):
        self.model_config = type('C', (), {'latent_space_version': None, 'arch': 'sd1', 'is_pixart_sigma': False})
        import torch as _torch
        self.device = 'cpu'
        self.device_torch = _torch.device('cpu')
        self.torch_dtype = _torch.float32
        self.is_xl = False
        self.is_v3 = False
        self.is_auraflow = False
        self.is_flux = False

    def set_device_state_preset(self, s):
        pass
    def restore_device_state(self):
        pass

    def encode_images(self, imgs):
        B = imgs.shape[0]
        h = imgs.shape[-2] // 8
        w = imgs.shape[-1] // 8
        return torch.randn(B, 4, h, w)


def test_ai_toolkit_dataset_cache_latents_runs(tmp_path):
    # prepare image
    img = tmp_path / 'img.png'
    Image.new('RGB', (512, 512)).save(img)

    cfg = DatasetConfig(dataset_path=str(tmp_path), cache_latents=True)
    sd = DummySDFull()

    ds = AiToolkitDataset(cfg, sd=sd)

    # after setup, latents should be cached in memory
    assert any(getattr(fi, 'is_latent_cached', False) for fi in ds.file_list)
