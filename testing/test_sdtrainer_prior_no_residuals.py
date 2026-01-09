import torch
import types

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummySD:
    def __init__(self):
        self.unet = types.SimpleNamespace(training=False)
        self.last_predict_calls = []

    def predict_noise(self, *args, **kwargs):
        # record kwargs for inspection
        self.last_predict_calls.append(dict(kwargs))
        # return a dummy tensor
        return torch.zeros(1, 3, 16, 16)


class DummyTrainer:
    """Lightweight object to bind SDTrainer.get_prior_prediction on in tests."""
    pass


def test_get_prior_prediction_uses_copy_and_preserves_original_pred_kwargs():
    # Setup
    sd = DummySD()
    tr = DummyTrainer()

    # Attach attributes used by get_prior_prediction
    tr.sd = sd
    tr.adapter = None
    tr.network = None
    tr.train_config = types.SimpleNamespace(cfg_scale=1.0, do_guidance_loss=False, unload_text_encoder=False, cfg_rescale=None)
    tr.device_torch = 'cpu'

    # Minimal inputs
    noisy_latents = torch.randn(1, 3, 16, 16)
    conditional_embeds = torch.randn(1, 10, 768)
    timesteps = torch.tensor([10.0])
    pred_kwargs = {
        'down_block_additional_residuals': [torch.ones(1, 4, 8, 8)],
        'mid_block_additional_residual': [torch.ones(1, 4, 8, 8)],
        'some_other_kwarg': 123,
    }

    # Bind the method
    method = SDTrainer.get_prior_prediction.__get__(tr, DummyTrainer)

    # Call with match_adapter_assist=True which should remove residuals in the prior call only
    prior_pred = method(
        noisy_latents=noisy_latents,
        conditional_embeds=conditional_embeds,
        match_adapter_assist=True,
        network_weight_list=[],
        timesteps=timesteps,
        pred_kwargs=pred_kwargs,
        batch=None,
        noise=torch.randn(1, 3, 16, 16),
        unconditional_embeds=None,
    )

    # Assert sd.predict_noise was called
    assert len(sd.last_predict_calls) >= 1

    # Inspect the kwargs of the first predict call (the prior)
    first_call_kwargs = sd.last_predict_calls[0]
    # It should NOT contain residual keys
    assert 'down_block_additional_residuals' not in first_call_kwargs
    assert 'down_intrablock_additional_residuals' not in first_call_kwargs
    assert 'mid_block_additional_residual' not in first_call_kwargs

    # Original pred_kwargs must remain unchanged for later training
    assert 'down_block_additional_residuals' in pred_kwargs
    assert 'mid_block_additional_residual' in pred_kwargs
    assert pred_kwargs['some_other_kwarg'] == 123

    # prior_pred should be a tensor
    assert isinstance(prior_pred, torch.Tensor)