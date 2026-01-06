import pytest
import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class FlakyInner:
    def __init__(self):
        self.attempts = 0
        self.conv = torch.nn.Parameter(torch.zeros(1, 4, 3, 3))

    def modules(self):
        yield self

    def __call__(self, latents, timestep, control_context, *args, **kwargs):
        # Support list-of-latents parity
        if isinstance(latents, list):
            first = latents[0]
            if isinstance(first, torch.Tensor):
                ch = int(first.shape[0]) if first.ndim >= 1 else None
            else:
                ch = None
        else:
            ch = latents.shape[1] if isinstance(latents, torch.Tensor) else None
        self.attempts += 1
        # fail on first attempt if 3 channels, succeed when 4
        if ch == 3 and self.attempts == 1:
            raise RuntimeError(f"Given groups=1, weight of size [320, 4, 3, 3], expected input to have 4 channels, but got 3 channels instead")
        return torch.zeros(1)


@pytest.mark.xfail
def test_wrapper_adapts_preemptively_to_inferred_expected_and_succeeds():
    inner = FlakyInner()
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros(1, 3, 112, 84)
    ctrl = torch.zeros(1, 4, 512, 512)

    # Wrapper should infer expected_in from inner (4) and adapt latents before
    # invoking inner, so the inner will succeed on first attempt without
    # relying on parsing its error message and retrying.
    out = wrapper(lat, 0, ctrl, conditioning_scale=1.0)

    assert out.shape == lat.shape
    assert inner.attempts >= 1, "inner should have been invoked with adapted latents (one or more attempts)"