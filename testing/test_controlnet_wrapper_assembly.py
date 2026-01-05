import types
import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyInner:
    def __init__(self):
        # VideoX-style expected control_in_dim
        self.control_in_dim = 33
        self.name_or_path = 'dummy_videox'
        self.called = False

    def forward(self, latents_list, timestep, control_context=None, conditioning_scale=1.0, *args, **kwargs):
        # Assert we received an assembled control_context (5D with control_in_dim channels)
        assert isinstance(control_context, torch.Tensor)
        # assembled context should be 5D [B, C, F, H, W] or 4D when singleton frame collapsed (we expect 5D from assembler)
        assert control_context.ndim == 5 or (control_context.ndim == 4 and int(control_context.shape[1]) == 33)
        if control_context.ndim == 5:
            assert control_context.shape[1] == 33
        else:
            assert control_context.shape[1] == 33
        self.called = True
        # return a trivial output matching one sample
        B = latents_list[0].shape[0] if isinstance(latents_list, list) else latents_list.shape[0]
        return torch.zeros((B, 4, 64, 64))


def test_wrapper_assembles_precomputed_latents():
    inner = DummyInner()
    wrapper = VideoXControlnetWrapper(inner)

    # create precomputed latents packed along channels: base=4, frames=4 => C=16
    pre_latents = torch.randn(1, 16, 64, 64)
    latents = torch.randn(1, 4, 64, 64)
    # Tag as precomputed latents to enable deterministic consumer-side assembly
    from toolkit.control_channels import tag_tensor
    tag_tensor(pre_latents, 'precompute:control_latents:size=512')

    out = wrapper(latents, 10, pre_latents, conditioning_scale=1.0)
    assert inner.called, "Inner adapter was not called"

def test_wrapper_assembles_tagged_precomputed_latents():
    inner = DummyInner()
    wrapper = VideoXControlnetWrapper(inner)
    pre_latents = torch.randn(1, 16, 64, 64)
    latents = torch.randn(1, 4, 64, 64)
    # Tag as precomputed latents
    from toolkit.control_channels import tag_tensor
    tag_tensor(pre_latents, 'precompute:control_latents:size=512')
    out = wrapper(latents, 10, pre_latents, conditioning_scale=1.0)
    assert inner.called, "Inner adapter was not called for tagged precomputed latents"
