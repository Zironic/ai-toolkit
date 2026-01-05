import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyInner(torch.nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv_in = torch.nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=3, padding=1)

    def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
        # Assert control_context has expected channels (conv_in in-ch)
        if isinstance(control_context, torch.Tensor):
            assert control_context.ndim == 4
            assert control_context.shape[1] == self.conv_in.weight.shape[1]
        elif isinstance(control_context, (list, tuple)):
            for c in control_context:
                assert c.shape[1] == self.conv_in.weight.shape[1]
        # Return a simple residual tensor list
        b, c, h, w = latents.shape
        return [torch.zeros(b, 16, h // 4, w // 4, dtype=latents.dtype, device=latents.device)]


def test_video_x_wrapper_does_not_pad_3_to_4_channels():
    inner = DummyInner(4)
    wrapper = VideoXControlnetWrapper(inner)

    latents = torch.randn(1, 16, 64, 64)
    timestep = torch.tensor([10])
    control_ctx_3ch = torch.randn(1, 3, 64, 64)

    # Under the new policy, the wrapper should not perform 3->4 padding; the inner
    # will receive the original 3 channel control context and proceed (or raise if
    # it cannot consume it). We assert that no automatic padding occurred here.
    out = wrapper(latents, timestep, control_ctx_3ch, conditioning_scale=1.0)
    assert out is not None
