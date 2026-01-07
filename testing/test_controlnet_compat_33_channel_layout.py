import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyInner(torch.nn.Module):
    def __init__(self, control_in_dim=None):
        super().__init__()
        self.control_in_dim = control_in_dim
        self.called_with = None

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        # capture a clone for later inspection to avoid accidental in-place mutation
        self.called_with = control_context.clone() if isinstance(control_context, torch.Tensor) else control_context
        return torch.zeros(1)


def test_33_channel_packed_layout_preserved_and_passed_through():
    inner = DummyInner(control_in_dim=4)  # adapter explicitly claims 4 but wrapper should accept 33
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros((1, 16, 64, 64))
    # Construct 33-channel packed control context: 16 zeros, 1 ones mask, 16 twos
    zeros16 = torch.zeros((1, 16, 8, 8))
    mask1 = torch.ones((1, 1, 8, 8))
    twos16 = torch.full((1, 16, 8, 8), 2.0)
    ctrl = torch.cat([zeros16, mask1, twos16], dim=1)

    out = wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    assert isinstance(inner.called_with, torch.Tensor)
    assert inner.called_with.shape[1] == 33
    # Verify channel-wise content preserved
    assert torch.allclose(inner.called_with[:, :16], zeros16)
    assert torch.allclose(inner.called_with[:, 16:17], mask1)
    assert torch.allclose(inner.called_with[:, 17:], twos16)
