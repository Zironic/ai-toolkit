import torch
from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


def test_reassemble_with_downsample():
    # Create a single-tile latent representing a full 512x512 image with downsample=8
    w_px, h_px = 512, 512
    latent_downsample = 8
    H_lat = h_px // latent_downsample
    W_lat = w_px // latent_downsample

    cfg = ModelConfig(name_or_path='some_model')
    model = ZImageModel('cpu', cfg)

    # single tile covering full image
    latent = torch.full((1, 4, H_lat, W_lat), 9.0)
    tile_latents = [(latent, (0, 0), (w_px, h_px))]

    full = model.reassemble_tile_latents(tile_latents, (w_px, h_px), latent_downsample=latent_downsample)

    assert full.shape[2] == H_lat and full.shape[3] == W_lat
    assert torch.all(full == 9.0)
