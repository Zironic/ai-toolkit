import pytest
import torch
import importlib

from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyInner:
    def __init__(self):
        self.name_or_path = 'dummy-adapter'

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        return torch.zeros(1)


def test_assembly_failure_bubbles_up(monkeypatch):
    inner = DummyInner()
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros((1, 16, 64, 64))
    # Packed latents shape that will go into assembly branch (C multiple of base)
    ctrl = torch.zeros((1, 8, 8, 8))  # 8 % 4 == 0 so we will try to reshape/collapse then assemble

    # Monkeypatch assemble_zimage_control_context to raise
    cc = importlib.import_module('toolkit.control_channels')

    def boom(*args, **kwargs):
        raise ValueError('assembly-boom')

    monkeypatch.setattr(cc, 'assemble_zimage_control_context', boom)

    with pytest.raises(RuntimeError) as exc:
        wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    msg = str(exc.value)
    assert 'Deterministic assembly failed' in msg or 'assembly-boom' in msg
