import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyInner(torch.nn.Module):
    def __init__(self, control_in_dim=None):
        super().__init__()
        self.control_in_dim = control_in_dim
        self.called_with = None

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        self.called_with = control_context
        return torch.zeros(1)


def test_33_channel_passthrough_respects_precomputed_even_when_adapter_expects_4():
    inner = DummyInner(control_in_dim=4)
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros((1, 16, 64, 64))
    # 33-channel assembled control_context (already assembled)
    ctrl = torch.zeros((1, 33, 8, 8))

    out = wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    assert isinstance(inner.called_with, torch.Tensor)
    assert inner.called_with.shape[1] == 33
