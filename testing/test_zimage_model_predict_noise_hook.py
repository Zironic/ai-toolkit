try:
    import torch
except Exception as e:
    import pytest

    # Some test environments produce an inspect-related AttributeError when importing torch
    # which prevents collection. Skip these tests early with an informative message.
    if isinstance(e, AttributeError) and "'__file__' has no attribute 'endswith'" in str(e):
        pytest.skip("Skipping tests due to broken torch import in this environment", allow_module_level=True)
    raise

from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel


def test_predict_noise_zimage_delegates_to_transformer():
    sd = ZImageModel(device='cpu', model_config=ModelConfig(name_or_path='dummy'))

    class DummyTransformer:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, **kwargs):
            # Return a simple deterministic tensor to verify delegation
            return latents + 1.0

    sd.transformer = DummyTransformer()
    lat = torch.zeros((1, 4, 8, 8))
    text_emb = torch.zeros((1, 1, 512))
    timestep = torch.tensor([1.0])

    out = sd._predict_noise_zimage(lat, text_emb, timestep)
    # Implementation may return a tensor or a tuple; accept both
    if isinstance(out, tuple):
        out = out[0]
    assert torch.allclose(out, lat + 1.0)


def test_predict_noise_zimage_controlnet_flow_sets_flags():
    sd = ZImageModel(device='cpu', model_config=ModelConfig(name_or_path='dummy'))

    class DummyTransformer:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, **kwargs):
            # If control_context is passed as kwarg, reflect that in output for test
            if 'control_context' in kwargs:
                return latents + 2.0
            if 'down_block_additional_residuals' in kwargs:
                return latents + 3.0
            return latents

    class DummyControlNet:
        def __call__(self, sample_for_controlnet, timestep, control_context, conditioning_scale=1.0):
            # Return a per-block residual list to trigger fallback path
            return [sample_for_controlnet * 0.1]

    sd.transformer = DummyTransformer()
    lat = torch.ones((1, 4, 8, 8))
    text_emb = torch.zeros((1, 1, 512))
    timestep = torch.tensor([1.0])

    out = sd._predict_noise_zimage(lat, text_emb, timestep, zimage_controlnet=DummyControlNet(), zimage_control_images=torch.zeros((1, 3, 64, 64)), zimage_conditioning_scale=1.0)
    if isinstance(out, tuple):
        out = out[0]

    # Since DummyControlNet returned per-block residuals and DummyTransformer handles
    # "down_block_additional_residuals" by adding 3.0, expect that behavior.
    assert torch.allclose(out, lat + 3.0)
    # Ensure diagnostic flags were set by the call
    assert getattr(sd, '_last_zimage_control_hints_present', False) is True
    assert getattr(sd, '_last_zimage_fellback_to_down_blocks', False) is True
