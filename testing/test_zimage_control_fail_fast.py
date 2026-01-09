try:
    import torch
except Exception as e:
    import pytest

    # Some environments have a broken torch import during test collection; skip in that case
    if isinstance(e, AttributeError) and "'__file__' has no attribute 'endswith'" in str(e):
        pytest.skip("Skipping tests due to broken torch import in this environment", allow_module_level=True)
    raise

from toolkit.config_modules import ModelConfig
from extensions_built_in.diffusion_models.z_image.z_image import ZImageModel
import pytest


def test_predict_noise_zimage_raises_when_no_control_applied():
    sd = ZImageModel(device='cpu', model_config=ModelConfig(name_or_path='dummy'))

    class DummyTransformer:
        def __call__(self, latents, timestep, cap_feats, return_dict=False, **kwargs):
            # Transformer rejects control_context kw
            if 'control_context' in kwargs:
                raise TypeError('unexpected kw arg control_context')
            # without control info it would normally return something, but we expect a RuntimeError upstream
            return latents

    class DummyControlNet:
        def __call__(self, sample_for_controlnet, timestep, control_context, conditioning_scale=1.0):
            # Adapter returns None (no per-block residuals)
            return None

    sd.transformer = DummyTransformer()
    lat = torch.zeros((1, 4, 8, 8))
    text_emb = torch.zeros((1, 1, 512))
    timestep = torch.tensor([1.0])

    with pytest.raises(RuntimeError) as exc:
        sd._predict_noise_zimage(lat, text_emb, timestep, zimage_controlnet=DummyControlNet(), zimage_control_images=torch.zeros((1, 16, 1, 64, 64)), zimage_conditioning_scale=1.0)

    assert 'Z-Image control routing failed' in str(exc.value)
