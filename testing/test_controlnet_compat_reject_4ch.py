import torch
import pytest
from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.called_with = None

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        self.called_with = control_context
        return torch.zeros(1)


def test_wrapper_rejects_3_channel_raw_images():
    inner = DummyInner()
    # Ensure adapter identity is present so the wrapper does not reject by adapter misconfiguration
    inner.name_or_path = 'dummy-adapter'
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros((1, 16, 64, 64))
    ctrl = torch.zeros((1, 3, 8, 8))  # 3-channel raw images — should be rejected

    with pytest.raises(RuntimeError) as exc:
        wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    msg = str(exc.value)
    assert 'Received raw 3-channel' in msg or 'do not pass raw images' in msg
