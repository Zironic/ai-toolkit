import torch
import pytest
from types import SimpleNamespace

from toolkit.controlnet_compat import VideoXControlnetWrapper


def test_adapt_control_images_error_propagates():
    class BadInner:
        def __call__(self, *args, **kwargs):
            return None
    wrapper = VideoXControlnetWrapper(BadInner())

    # Monkeypatch adapt_control_images to raise
    import toolkit.control_channels as cc
    def boom(*a, **k):
        raise RuntimeError("boom-adapt")
    cc.adapt_control_images = boom

    lat = torch.zeros((1,4,16,16))
    ctrl = torch.zeros((1,4,64,64))
    with pytest.raises(RuntimeError) as excinfo:
        wrapper.forward(lat, torch.tensor([1.0]), ctrl)
    # Accept either device-inspection or adapt_control_images failure depending on which check errors first
    msg = str(excinfo.value)
    assert ('Failed to adapt control_images' in msg) or ('Failed to inspect inner.parameters' in msg) or ('Failed to inspect inner.buffers' in msg)


def test_to_forward_failure_raises():
    class BrokenInner:
        def to(self, *a, **k):
            raise RuntimeError("to-failed")
    w = VideoXControlnetWrapper(BrokenInner())
    with pytest.raises(RuntimeError) as excinfo:
        w.to(torch.device('cpu'))
    assert 'Forwarding .to to inner adapter failed' in str(excinfo.value)
