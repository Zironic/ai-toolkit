import torch
from types import SimpleNamespace

from toolkit.controlnet_compat import VideoXControlnetWrapper


def test_video_x_adapter_call_and_normalization():
    # Fake adapter that mimics VideoX forward
    class FakeCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, latents, timestep, control_context, conditioning_scale=1.0, *args, **kwargs):
            # Return a list of per-block residuals (3 blocks)
            B, C, H, W = control_context.shape
            return [torch.zeros((B, C, H // 4, W // 4), dtype=latents.dtype) for _ in range(3)]

    fake = FakeCN()
    wrapper = VideoXControlnetWrapper(fake)

    # Prepare inputs
    sample_for_controlnet = torch.zeros((1, 4, 16, 16))
    timesteps = torch.tensor([10])
    control_context = torch.zeros((1, 4, 16, 16))

    # Call wrapper with expected signature
    out = wrapper(sample_for_controlnet, timesteps, control_context, conditioning_scale=0.7)

    assert isinstance(out, list)
    assert len(out) == 3
    for t in out:
        assert isinstance(t, torch.Tensor)

    # Normalize into down_block_additional_residuals style
    dtype = torch.float32
    adapter_multiplier = 1.0
    if isinstance(out, (list, tuple)):
        down_block_additional_residuals = [sample.to(dtype=dtype) * adapter_multiplier for sample in out]
    elif hasattr(out, 'shape'):
        down_block_additional_residuals = [out.to(dtype=dtype) * adapter_multiplier]
    else:
        down_block_additional_residuals = None

    assert down_block_additional_residuals is not None
    assert all(isinstance(x, torch.Tensor) for x in down_block_additional_residuals)
