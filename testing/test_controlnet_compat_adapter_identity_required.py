import torch
import pytest
from toolkit.controlnet_compat import VideoXControlnetWrapper


class UnnamedAdapterNoControlDim:
    def __init__(self):
        pass

    def forward(self, latents, timestep, control_context, *args, **kwargs):
        # simply return something if invoked
        return torch.zeros(1)


def test_adapter_identity_required_for_deterministic_routing():
    inner = UnnamedAdapterNoControlDim()
    wrapper = VideoXControlnetWrapper(inner)

    lat = torch.zeros((1, 16, 64, 64))
    # VAE-encoded/multi-frame latents that would normally trigger assembly (e.g., C==16)
    ctrl = torch.zeros((1, 16, 8, 8))

    with pytest.raises(RuntimeError) as exc:
        wrapper(lat, torch.tensor([1.0]), ctrl, conditioning_scale=1.0)

    assert "missing 'name'/'name_or_path'" in str(exc.value) or "missing 'control_in_dim'" in str(exc.value)
