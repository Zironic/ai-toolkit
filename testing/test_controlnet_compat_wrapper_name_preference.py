import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class InnerNoName(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, latents, timestep, control_context, *args, **kwargs):
        return torch.zeros(1)


def test_wrapper_prefers_own_name_if_inner_missing():
    inner = InnerNoName()
    wrapper = VideoXControlnetWrapper(inner)
    # Simulate loader having forced name onto the wrapper (common case)
    wrapper.name_or_path = 'forced-name'

    lat = torch.zeros((1, 16, 64, 64))
    ctrl = torch.zeros((1, 33, 8, 8))

    out = wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)
    assert out is not None
