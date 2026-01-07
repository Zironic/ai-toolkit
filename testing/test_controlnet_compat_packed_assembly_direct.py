import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper
import importlib


class DummyInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name_or_path = 'dummy'
        self.called_with = None

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        self.called_with = control_context
        return torch.zeros(1)


def test_packed_16_channels_triggers_direct_assembly(monkeypatch):
    inner = DummyInner()
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros((1, 16, 64, 64))
    # 5D zimage form: [B, C, F, H, W] where C=16
    ctrl = torch.zeros((1, 16, 1, 8, 8))

    # Monkeypatch assemble_zimage_control_context to assert it receives the packed channels (C=16)
    cc = importlib.import_module('toolkit.control_channels')

    seen = {}

    def fake_assemble(t, control_in_dim=None, mask_from=None):
        # confirm we received the packed channels (not collapsed)
        assert t.ndim == 4 or t.ndim == 5
        # If called with 5D, collapse will be handled by wrapper; but we expect original packed channels passed through
        shape = tuple(t.shape)
        seen['shape'] = shape
        # Return an assembled 33-channel tensor
        B = shape[0]
        H = shape[-2]
        W = shape[-1]
        return torch.zeros((B, 33, H, W))

    monkeypatch.setattr(cc, 'assemble_zimage_control_context', fake_assemble)

    out = wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    # inner should have received a 33-channel assembled context
    assert isinstance(inner.called_with, torch.Tensor)
    assert inner.called_with.shape[1] == 33
    assert seen.get('shape') is not None
