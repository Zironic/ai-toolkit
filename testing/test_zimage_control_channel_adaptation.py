import torch
import pytest
from toolkit.control_channels import adapt_noisy_latents_for_adapter
from types import SimpleNamespace


def test_noisy_latents_rejected_on_channel_mismatch():
    # Simulate noisy latents with 16 channels while adapter expects 4
    nl = torch.randn(1, 16, 8, 8)
    with pytest.raises(RuntimeError, match=r"Noisy latents have 16 channels but expected 4"):
        adapt_noisy_latents_for_adapter(nl, 4)


def test_noisy_latents_ok_when_channels_match():
    nl = torch.randn(1, 4, 8, 8)
    out = adapt_noisy_latents_for_adapter(nl, 4)
    assert out.shape[1] == 4
