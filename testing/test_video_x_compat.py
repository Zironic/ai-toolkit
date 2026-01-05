import torch
from types import SimpleNamespace
from toolkit.controlnet_compat import VideoXControlnetWrapper


def test_videox_wrapper_calls_inner_with_expected_signature():
    # A fake inner that records calls
    calls = {}

    class FakeInner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)

        def forward(self, *args, **kwargs):
            calls['args'] = args
            calls['kwargs'] = kwargs
            return ['res1', 'res2']

    inner = FakeInner()
    wrapper = VideoXControlnetWrapper(inner)

    # Ensure wrapper is an nn.Module and honors `.to()` for offload helpers
    wrapper.to(torch.device('cpu'))
    params = list(wrapper.parameters())
    assert len(params) > 0

    latents = torch.zeros((1, 4, 16, 16))
    timestep = torch.tensor([10])
    control_context = [torch.zeros((1, 3, 64, 64))]

    out = wrapper(latents, timestep, control_context, conditioning_scale=0.7)
    assert 'args' in calls
    assert isinstance(calls['args'][0], torch.Tensor)
    assert calls['kwargs'].get('conditioning_scale', 0.0) == 0.7
    assert out == ['res1', 'res2']
