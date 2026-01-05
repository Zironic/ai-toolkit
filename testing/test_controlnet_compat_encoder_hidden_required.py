import torch
import pytest
from toolkit.controlnet_compat import VideoXControlnetWrapper


class InnerRequiresEncoderHidden(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.record = {}

    # note: encoder_hidden_states is a required positional argument (no default)
    def forward(self, latents, timestep, encoder_hidden_states, controlnet_cond=None, **kwargs):
        assert encoder_hidden_states is None, "encoder_hidden_states placeholder not provided"
        assert controlnet_cond is not None, "controlnet_cond was not provided"
        if isinstance(controlnet_cond, torch.Tensor):
            self.record['shape'] = tuple(controlnet_cond.shape)
        return torch.zeros_like(latents[0])


def test_wrapper_rejects_flux1_style_adapter_missing_control_context():
    inner = InnerRequiresEncoderHidden()
    # instantiation should fail fast for Flux1-style adapters
    with pytest.raises(RuntimeError, match="requi.re 'encoder_hidden_states'|missing required parameter 'control_context'|Adapter appears to require 'encoder_hidden_states'"):
        VideoXControlnetWrapper(inner)
