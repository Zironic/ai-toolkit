import torch
import pytest
from toolkit.controlnet_compat import VideoXControlnetWrapper


class InnerRequiresControlNetCond(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.record = {}

    def forward(self, latents, timestep, controlnet_cond=None, controlnet_conditioning_scale=1.0, *args, **kwargs):
        assert controlnet_cond is not None, "controlnet_cond was not provided"
        # ensure shape is what wrapper produced
        if isinstance(controlnet_cond, torch.Tensor):
            self.record['shape'] = tuple(controlnet_cond.shape)
        elif isinstance(controlnet_cond, (list, tuple)) and len(controlnet_cond) > 0:
            self.record['shape'] = tuple(controlnet_cond[0].shape)
        return torch.zeros_like(latents[0])


class InnerRequiresControlContext(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.record = {}

    def forward(self, latents, timestep, control_context=None, control_context_scale=1.0, *args, **kwargs):
        assert control_context is not None, "control_context was not provided"
        if isinstance(control_context, torch.Tensor):
            self.record['shape'] = tuple(control_context.shape)
        elif isinstance(control_context, (list, tuple)) and len(control_context) > 0:
            self.record['shape'] = tuple(control_context[0].shape)
        return torch.zeros_like(latents[0])


def test_wrapper_rejects_non_control_context_adapters():
    inner = InnerRequiresControlNetCond()
    # Construction should succeed (shim applied at runtime if needed)
    wrapper = VideoXControlnetWrapper(inner)
    # Provide a pre-encoded/assembled VideoX control_context (B, C=33, 1, H, W)
    ctrl = torch.randn(1, 33, 1, 32, 32)
    lat = torch.randn(1, 16, 8, 8)
    # Calling the wrapper should work because the legacy shim maps `control_context`
    # to `controlnet_cond` for the inner implementation.
    out = wrapper(lat, 0.0, ctrl, conditioning_scale=1.0)
    # The inner should have been called and recorded a shape
    assert 'shape' in inner.record
    assert inner.record['shape'][1] == 33


def test_wrapper_accepts_control_context_adapters():
    inner = InnerRequiresControlContext()
    wrapper = VideoXControlnetWrapper(inner)
    # Provide a pre-encoded/assembled VideoX control_context (B, C=33, 1, H, W)
    ctrl = torch.randn(1, 33, 1, 32, 32)
    lat = torch.randn(1, 16, 8, 8)
    out = wrapper(lat, 0.0, ctrl, conditioning_scale=1.0)
    assert 'shape' in inner.record
    # The inner should receive a per-sample control_context element; channel count should be 33
    assert inner.record['shape'][1] == 33


def test_legacy_controlnet_shim_allows_wrapping():
    # Ensure the legacy shim makes legacy adapters compatible with the strict VideoX wrapper
    inner = InnerRequiresControlNetCond()
    # Import shim
    from toolkit.controlnet_compat import ControlNetLegacyAdapter
    shim = ControlNetLegacyAdapter(inner)
    # Now wrapping the shim should succeed
    wrapper = VideoXControlnetWrapper(shim)
    # Provide a pre-encoded VideoX control_context
    ctrl = torch.randn(1, 33, 1, 32, 32)
    lat = torch.randn(1, 16, 8, 8)
    out = wrapper(lat, 0.0, ctrl, conditioning_scale=1.0)
    # The inner should have received a controlnet_cond via the shim and recorded a shape
    assert 'shape' in inner.record
    assert inner.record['shape'][1] == 33
