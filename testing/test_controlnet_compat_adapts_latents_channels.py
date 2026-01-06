import pytest
import torch
from toolkit.controlnet_compat import VideoXControlnetWrapper


class DummyInner:
    def __init__(self):
        # Create a dummy conv parameter with weight shape (out, in, kh, kw)
        self.conv = torch.nn.Parameter(torch.zeros(1, 4, 3, 3))
        self._weights = [self.conv]
        self.name_or_path = 'z-image-turbo-fun-controlnet-union-2.1'

    def modules(self):
        yield self

    def __call__(self, latents, timestep, control_context, *args, **kwargs):
        # Record the latents shape received and return a tensor for testing
        try:
            self.received_latents_shape = tuple(latents.shape)
        except Exception:
            self.received_latents_shape = None
        return torch.zeros(1)


@pytest.mark.xfail
def test_wrapper_does_not_adapt_latents_when_no_control_in_dim():
    inner = DummyInner()
    wrapper = VideoXControlnetWrapper(inner)

    # latents with 16 channels should remain unchanged when there is no explicit
    # `control_in_dim` on the adapter — adaptations should be driven by the
    # Z-Image pipeline's assembly process instead of wrapper heuristics.
    lat = torch.zeros(1, 16, 112, 84)
    ctrl = torch.zeros(1, 3, 512, 512)

    with pytest.raises(RuntimeError) as exc:
        wrapper(lat, 0, ctrl, conditioning_scale=1.0)

    msg = str(exc.value)
    assert 'raw pixel' in msg or 'received raw pixel images' in msg
    assert inner.received_latents_shape[1] == 16, f"inner expected latents with 16 channels but got {inner.received_latents_shape}"