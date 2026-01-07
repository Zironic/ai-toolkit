import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper

class DummyInnerAccepts3:
    def __call__(self, latents, timestep, control_context, *args, **kwargs):
        ch = control_context.shape[1]
        if ch != 3:
            raise RuntimeError(f"Given groups=1, weight of size [16, 3, 3, 3], expected input[1, {ch}, 512, 512] to have 3 channels, but got {ch} channels instead")
        return torch.tensor([1])


def test_videox_wrapper_no_retry_on_conv_message():
    inner = DummyInnerAccepts3()
    wrapper = VideoXControlnetWrapper(inner)
    lat = torch.randn(1, 16, 32, 32)
    ctrl_bad = torch.randn(1, 5, 64, 64)
    # With strict policy we fail fast on unsupported channel counts (not rely on inner error messages)
    import pytest
    with pytest.raises(Exception):
        _ = wrapper(lat, 0, ctrl_bad, conditioning_scale=1.0)
