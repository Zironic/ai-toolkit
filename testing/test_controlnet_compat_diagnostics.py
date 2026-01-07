import torch
import pytest
from toolkit.controlnet_compat import VideoXControlnetWrapper


class FakeInnerExpect4:
    """Fake adapter that expects control_context tensors to have 4 channels.

    Its forward will raise a runtime error that mimics a Conv2d channel mismatch
    when provided an input whose channel count != 4. This helps test the
    wrapper's diagnostic heuristics and error messages.
    """
    def parameters(self):
        return []

    def forward(self, latents, timestep, control_context, *args, **kwargs):
        # Detect channel count robustly for both tensor and list-of-tensors inputs
        def _ch_of(obj):
            if isinstance(obj, torch.Tensor):
                # If shaped as [B, C, ...] prefer C at dim 1 when B==1, else prefer dim 0
                if obj.ndim >= 2:
                    if obj.ndim >= 4 and obj.shape[0] == 1:
                        return int(obj.shape[1])
                    return int(obj.shape[0])
            if isinstance(obj, (list, tuple)) and len(obj) > 0 and isinstance(obj[0], torch.Tensor):
                return _ch_of(obj[0])
            return None

        ch = _ch_of(latents)
        # Simulate conv expecting 4 channels but got something else
        if ch is not None and ch != 4:
            raise RuntimeError(f"Given groups=1, weight of size [320, 4, 3, 3], expected input[1, {ch}, 148, 60] to have 4 channels, but got {ch} channels instead")
        # Return a tensor-shaped placeholder matching expected form if not raising
        if isinstance(latents, list):
            return torch.zeros_like(latents[0])
        return torch.zeros_like(latents)


def test_wrapper_reports_offending_tensor_and_origin():
    inner = FakeInnerExpect4()
    wrapper = VideoXControlnetWrapper(inner)

    latents = torch.zeros((1, 16, 148, 60))
    control_context = torch.zeros((1, 3, 512, 512))

    # With strict VideoX parity the inner adapter error should bubble up (no safe fallbacks)
    with pytest.raises(RuntimeError) as exc:
        _ = wrapper(latents, torch.tensor(10.0), control_context, conditioning_scale=1.0)

    msg = str(exc.value)
    # The wrapper should fail fast on raw pixel inputs and instruct the caller to provide
    # pre-encoded control latents rather than attempting to call the inner adapter.
    assert 'VideoXControlnetWrapper received raw pixel images' in msg or 'provide pre-encoded control latents' in msg