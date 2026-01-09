import pytest
import torch

from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyAdapter:
    def __init__(self, name='dummy'):
        self.name_or_path = name

    def __call__(self, *args, **kwargs):
        # Deliberately return None to simulate broken adapter
        return None


def test_video_x_wrapper_raises_on_none_return():
    dummy = DummyAdapter('broken-adapter')
    wrapper = VideoXControlnetWrapper(dummy)

    # Build a minimal latent tensor [B,C,H,W]
    sample = torch.randn(1, 16, 72, 56)
    timestep = torch.tensor([1.0])

    with pytest.raises(RuntimeError) as exc:
        _ = wrapper(sample, timestep, sample, conditioning_scale=1.0)

    assert 'returned None' in str(exc.value) or 'no control hints' in str(exc.value)