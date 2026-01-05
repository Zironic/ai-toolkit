import numpy as np
from PIL import Image
import torch
from toolkit.dataloader_mixins import tile_image
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
from toolkit.config_modules import ModelConfig


def test_reassemble_tile_latents(monkeypatch):
    # Create a 200x200 image
    w, h = 200, 200
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    cfg = ModelConfig(name_or_path='some_model')
    model = ZImageModel('cpu', cfg)

    # Monkeypatch VAE encode to return constant latents shaped (1, C, h_tile, w_tile)
    class DummyVAE:
        def encode(self, x):
            # x shape (1, C, H, W) return (1, 4, H//1, W//1) same spatial dims for simplicity
            bs, c, H, W = x.shape
            return torch.full((1, 4, H, W), fill_value=7.0)
    model.vae = DummyVAE()

    tiles = tile_image(img, tile_size=100, overlap=0)
    # Encode tiled
    tiled = []
    for tile, pos in tiles:
        # simulate encode
        t = torch.full((1, 4, tile.size[1], tile.size[0]), 7.0)
        tiled.append((t, pos, tile.size))

    full = model.reassemble_tile_latents(tiled, (w, h), latent_downsample=1)
    # Expect full spatial shape equal to original
    assert full.shape[2] == h and full.shape[3] == w
    # Check values placed
    assert torch.all(full == 7.0)
