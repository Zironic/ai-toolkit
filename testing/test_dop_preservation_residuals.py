import torch
import types

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummySD:
    def __init__(self):
        self.last_predict_calls = []

    def predict_noise(self, *args, **kwargs):
        # record kwargs for inspection
        self.last_predict_calls.append(dict(kwargs))
        return torch.zeros(1, 3, 16, 16)


class DummyTrainer:
    pass


def test_run_preservation_forward_cleans_residuals_and_preserves_original_pred_kwargs():
    sd = DummySD()
    t = DummyTrainer()

    # attach minimal attributes used by the method
    t.sd = sd
    t.device_torch = 'cpu'
    t.accelerator = types.SimpleNamespace(backward=lambda x: None)

    # Create noisy latents and embeds
    noisy = torch.randn(1, 3, 16, 16)
    timesteps = torch.tensor([10.0])
    preservation_embeds = types.SimpleNamespace(to=lambda device, dtype=None: torch.randn(1, 10, 768))
    unconditional_embeds = None

    # pred_kwargs contains residuals that should be removed for preservation when match_adapter_assist True
    pred_kwargs = {
        'down_block_additional_residuals': [torch.ones(1, 4, 8, 8)],
        'mid_block_additional_residual': [torch.ones(1, 4, 8, 8)],
        'extra': 42,
    }

    # bind method
    method = SDTrainer._run_preservation_forward.__get__(t, DummyTrainer)

    # call with match_adapter_assist True
    res = method(
        noisy_latents=noisy,
        timesteps=timesteps,
        preservation_embeds=preservation_embeds,
        unconditional_embeds=unconditional_embeds,
        batch=None,
        pred_kwargs=pred_kwargs,
        dtype='float32',
        prior_pred=torch.zeros(1, 3, 16, 16),
        preservation_resolution=None,
        preservation_kind='dop',
        match_adapter_assist=True,
        network_weight_list=None,
    )

    # One or more predict calls should have been made (preservation). Inspect the first
    assert len(sd.last_predict_calls) >= 1
    first_kwargs = sd.last_predict_calls[0]

    # residual keys should not be present in the preservation predict kwargs
    assert 'down_block_additional_residuals' not in first_kwargs
    assert 'mid_block_additional_residual' not in first_kwargs

    # original pred_kwargs unchanged
    assert 'down_block_additional_residuals' in pred_kwargs
    assert 'mid_block_additional_residual' in pred_kwargs
    assert pred_kwargs['extra'] == 42