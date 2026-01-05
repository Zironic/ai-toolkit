import torch
from types import SimpleNamespace
import pytest

# list of (module path, class name) to test
targets = [
    ('extensions_built_in.flex2.flex2', 'Flex2'),
    ('extensions_built_in.diffusion_models.qwen_image.qwen_image_edit', 'QwenImageEditModel'),
    ('extensions_built_in.diffusion_models.omnigen2.__init__', 'OmniGen2Model'),
    ('extensions_built_in.diffusion_models.hidream.hidream_e1_model', 'HidreamE1Model'),
    ('extensions_built_in.diffusion_models.flux_kontext.flux_kontext', 'FluxKontextModel'),
]


@pytest.mark.parametrize('module_path,class_name', targets)
def test_model_uses_tiled_encode_when_configured(module_path, class_name):
    mod = __import__(module_path, fromlist=[class_name])
    cls = getattr(mod, class_name)
    # get unbound method
    cond_fn = getattr(cls, 'condition_noisy_latents')

    called = {}

    def fake_encode_control_images(control_tensor, tile=False, tile_size=None, overlap=None):
        called['tile'] = tile
        called['tile_size'] = tile_size
        called['overlap'] = overlap
        bs = control_tensor.shape[0]
        return torch.zeros((bs, 4, 16, 16))

    # build a dummy self with only the attributes used by the method
    dummy = SimpleNamespace()
    dummy.control_dropout = 0.0
    dummy.inpaint_dropout = 0.0
    dummy.inpaint_random_chance = 0.0
    dummy.invert_inpaint_mask_chance = 0.0
    dummy.vae_device_torch = torch.device('cpu')
    dummy.device_torch = torch.device('cpu')
    dummy.vae = SimpleNamespace(to=lambda *args, **kwargs: None, eval=lambda *args, **kwargs: None)
    dummy.torch_dtype = torch.float32
    dummy.do_random_inpainting = False
    dummy.random_blur_mask = False
    dummy.random_dialate_mask = False
    dummy.model_config = SimpleNamespace(control_use_tiling=True, control_tiling_size=128, control_tiling_overlap=16)
    # attach our fake encode
    dummy.encode_control_images = fake_encode_control_images
    dummy.encode_images = lambda x: torch.zeros((x.shape[0], 4, 16, 16))
    # handler for file_items-based size lookup
    dummy.file_items = [SimpleNamespace(crop_height=64, crop_width=64)]

    # minimal batch and latents
    latents = torch.randn(1, 4, 16, 16)
    batch = SimpleNamespace()
    batch.control_tensor = torch.rand(1, 3, 64, 64)
    batch.tensor = torch.rand(1, 3, 64, 64)
    batch.latents = latents
    batch.inpaint_tensor = None
    batch.mask_tensor = None

    # call the method
    out = cond_fn(dummy, latents, batch)

    assert called.get('tile') is True
    assert called.get('tile_size') == 128
    assert called.get('overlap') == 16
    # output should have extra control channels concatenated OR model may keep control in an internal attribute
    if hasattr(dummy, '_control_latent'):
        assert dummy._control_latent is not None
    else:
        # some models modify latents differently (inpainting, stacked channels etc.)
        # assert that output is a tensor and batch dim is preserved
        assert isinstance(out, torch.Tensor)
        assert out.shape[0] == latents.shape[0]
