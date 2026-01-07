import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class ReturnerInner(torch.nn.Module):
    def __init__(self, control_in_dim=None, return_tensor=None):
        super().__init__()
        self.control_in_dim = control_in_dim
        self.called_with = None
        self.return_tensor = return_tensor

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        self.called_with = control_context
        if self.return_tensor is not None:
            return self.return_tensor
        return torch.zeros(1)


def test_33_channel_forward_returns_inner_output():
    # Inner expects 4 but receives a precomputed 33-channel tensor
    out_tensor = torch.full((1, 16, 64, 64), 7.0)
    inner = ReturnerInner(control_in_dim=4, return_tensor=out_tensor)
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros((1, 16, 64, 64))
    ctrl = torch.zeros((1, 33, 8, 8))

    out = wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    # The wrapper should return the inner's output unchanged
    assert torch.all(out == out_tensor)
    # The inner should have been called with the 33-channel control_context
    assert isinstance(inner.called_with, torch.Tensor)
    assert inner.called_with.shape[1] == 33


def test_33_channel_list_passthrough():
    inner = ReturnerInner(control_in_dim=4)
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros((1, 16, 64, 64))
    ctrl = [torch.zeros((1, 33, 8, 8)), torch.zeros((1, 33, 8, 8))]

    _ = wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    # When provided a list, inner should receive a list and each item should be 33-channel
    assert isinstance(inner.called_with, list)
    assert len(inner.called_with) == 2
    assert all(isinstance(x, torch.Tensor) and x.shape[1] == 33 for x in inner.called_with)
