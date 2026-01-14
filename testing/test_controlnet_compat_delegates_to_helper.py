import pytest
pytest.importorskip("torch")
from toolkit.controlnet_compat import VideoXControlnetWrapper
from types import SimpleNamespace


def test_wrapper_delegates_to_compute(monkeypatch):
    called = {}

    class DummyInner:
        def __init__(self):
            self.name_or_path = 'dummy'
        def __call__(self, *args, **kwargs):
            # Should not be invoked when helper is monkeypatched
            called['inner_called'] = True
            return torch.zeros((1, 16, 1, 8, 8))

    inner = DummyInner()
    wrapper = VideoXControlnetWrapper(inner)

    def fake_compute(sd, noisy_latents, timesteps, zimage_controlnet=None, zimage_control_context=None, **kwargs):
        called['args'] = (getattr(noisy_latents, 'shape', None), getattr(timesteps, 'shape', None), getattr(zimage_control_context, 'shape', None))
        # return down list, mid, control_context, raw_out
        return [torch.zeros((noisy_latents.shape[0], 16, 1, noisy_latents.shape[-2], noisy_latents.shape[-1]))], None, zimage_control_context, torch.zeros((noisy_latents.shape[0], 16, 1, noisy_latents.shape[-2], noisy_latents.shape[-1]))

    monkeypatch.setattr('extensions_built_in.diffusion_models.z_image.z_image.compute_zimage_adapter_residuals', fake_compute)

    lat = torch.zeros((1, 4, 16, 16))
    ctrl = torch.zeros((1, 33, 16, 16))
    out = wrapper(lat, torch.tensor(10.0), ctrl, conditioning_scale=1.0)

    assert 'args' in called, "compute helper should be invoked by wrapper"
    assert out.shape[0] == 1
