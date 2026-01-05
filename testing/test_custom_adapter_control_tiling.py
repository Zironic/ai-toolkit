import torch
from types import SimpleNamespace
from toolkit.custom_adapter import CustomAdapter
from toolkit.config_modules import AdapterConfig, TrainConfig


def test_custom_adapter_uses_tiled_encode(monkeypatch):
    # create dummy sd with minimal attributes
    class FluxTransformer2DModel:
        def __init__(self):
            self.x_embedder = torch.nn.Linear(64, 3072)
            class Config(dict):
                def __getattr__(self, k):
                    return self[k]
                def __setattr__(self, k, v):
                    self[k] = v
            self.config = Config({'in_channels': 3})
            self.device = torch.device('cpu')

    unet = FluxTransformer2DModel()

    called = {}

    def fake_encode_control_images(control_tensor, tile=False, tile_size=None, overlap=None):
        # record call
        called['tile'] = tile
        called['tile_size'] = tile_size
        called['overlap'] = overlap
        # return a fake latent tensor matching expected latent shape
        bs = control_tensor.shape[0]
        return torch.zeros((bs, 4, 16, 16))

    class SD:
        pass

    sd = SD()
    sd.unet = unet
    sd.dtype = 'fp32'
    sd.device_torch = torch.device('cpu')
    sd.vae_device_torch = torch.device('cpu')
    sd.torch_dtype = torch.float32
    sd.model_config = SimpleNamespace(control_use_tiling=True, control_tiling_size=128, control_tiling_overlap=16)
    sd.encode_control_images = fake_encode_control_images

    adapter_cfg = AdapterConfig(type='control_lora', num_control_images=1, control_image_dropout=0.0)
    train_cfg = TrainConfig()

    adapter = CustomAdapter(sd, adapter_cfg, train_cfg)

    # create dummy batch and latents
    latents = torch.randn(1, 4, 16, 16)
    batch = SimpleNamespace()
    batch.control_tensor = torch.rand(1, 3, 64, 64)  # 0-1 control image
    batch.tensor = torch.rand(1, 3, 64, 64)
    batch.latents = latents

    out = adapter.condition_noisy_latents(latents, batch)

    # ensure encode_control_images was called with tiling parameters
    assert called.get('tile') is True
    assert called.get('tile_size') == 128
    assert called.get('overlap') == 16

    # ensure output has extra channels appended (control channels = 4)
    assert out.shape[1] == latents.shape[1] + 4
