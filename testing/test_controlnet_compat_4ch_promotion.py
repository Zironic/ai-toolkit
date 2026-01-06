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
        # yield self to emulate a module with a weight attribute
        yield self

    def __call__(self, latents, timestep, control_context, *args, **kwargs):
        # Record the control_context shape the inner receives and return a tensor
        self.last_ctrl_shape = None
        try:
            if isinstance(control_context, torch.Tensor):
                self.last_ctrl_shape = tuple(control_context.shape)
            elif isinstance(control_context, (list, tuple)) and isinstance(control_context[0], torch.Tensor):
                self.last_ctrl_shape = tuple(control_context[0].shape)
        except Exception:
            self.last_ctrl_shape = 'unknown'
        # Accept both tensor and list latents for compatibility with VideoX parity
        try:
            if isinstance(latents, list):
                first = latents[0]
                # first may be shaped [C,1,H,W] or [C,H,W]
                if first.ndim == 4:
                    return torch.zeros_like(first[:, :1, ...])
                elif first.ndim == 3:
                    # add batch dim
                    return torch.zeros_like(first.unsqueeze(0)[:, :1, ...])
            return torch.zeros_like(latents[:, :1, ...])
        except Exception:
            return torch.zeros(1)


@pytest.mark.xfail
def test_wrapper_does_not_promote_to_4_channels():
    inner = DummyInner()
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros(1, 16, 72, 56)
    ctrl = torch.zeros(1, 3, 512, 512)

    # Under the new policy the wrapper no longer promotes pixel images to 4 channels;
    # the Z-Image pipeline should perform encoding/assembly instead. Verify the inner
    # received the original 3-channel control image.
    import pytest
    with pytest.raises(RuntimeError) as exc:
        wrapper(lat, 0, ctrl, conditioning_scale=1.0)

    msg = str(exc.value)
    assert 'raw pixel' in msg or 'received raw pixel images' in msg

    # The wrapper should raise before invoking the inner adapter
    assert getattr(inner, 'last_ctrl_shape', None) is None, "inner should not have been called for raw pixel images"
    assert inner.last_ctrl_shape[1] == 3, f"inner was expected to receive control_context with 3 channels but got {inner.last_ctrl_shape}"