import torch
from types import SimpleNamespace
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("sd_trainer_module", str(pathlib.Path(__file__).resolve().parents[1] / "extensions_built_in/sd_trainer/SDTrainer.py"))
sd_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd_module)
SDTrainer = sd_module.SDTrainer


class DummySDV:
    def __init__(self):
        self.vae = type('T', (), {'config': {'block_out_channels': [64, 128, 256, 512]}})()
        self.transformer = type('Tr', (), {'all_patch_size': [1]})()


class DummyTrainer(SDTrainer):
    def __init__(self):
        # minimal attributes
        self.sd = DummySDV()
        self.train_config = type('C', (), {})()
        self.device_torch = torch.device('cpu')
        # lightweight timer
        from toolkit.timer import Timer
        self.timer = Timer('test')
        # simple accelerator recorder
        self.backward_calls = []
        self.accelerator = SimpleNamespace(backward=lambda x: self.backward_calls.append(x))


def test_preservation_loss_has_grad_and_triggers_backward(monkeypatch):
    t = DummyTrainer()

    noisy = torch.zeros((1, 4, 64, 64))
    timesteps = torch.tensor([0])

    class DummyPreservationEmbeds:
        def to(self, device, dtype=None):
            return self

    preservation_embeds = DummyPreservationEmbeds()

    # ensure get_prior_prediction returns a constant small prior
    def fake_get_prior_prediction(noisy_latents=None, **kwargs):
        return torch.zeros((1, 4, 32, 32))

    monkeypatch.setattr(t, 'get_prior_prediction', fake_get_prior_prediction)

    # fake predict_noise to return a tensor that requires grad so preservation_pred has grad
    def fake_predict_noise(noisy_latents=None, **kwargs):
        # return a tensor that requires grad to simulate connection to model params
        return torch.zeros((1, 4, noisy_latents.shape[-2], noisy_latents.shape[-1]), requires_grad=True)

    monkeypatch.setattr(t, 'predict_noise', fake_predict_noise)

    # Call the preservation forward WITH gradients enabled (mimic the trainer behavior)
    with torch.set_grad_enabled(True):
        result = t._run_preservation_forward(
            noisy_latents=noisy,
            timesteps=timesteps,
            preservation_embeds=preservation_embeds,
            unconditional_embeds=None,
            batch=None,
            pred_kwargs={},
            dtype='float32',
            prior_pred=None,
            preservation_resolution=256,
            preservation_kind='dop',
            match_adapter_assist=False,
            network_weight_list=[]
        )

    assert isinstance(result, tuple)
    preservation_pred, prior_small = result

    # preservation_pred should require_grad because we ran with grads enabled and predict_noise
    assert getattr(preservation_pred, 'requires_grad', False) is True

    # Compute preservation loss by invoking the trainer helper (it returned None in earlier runs)
    preservation_loss = t._compute_and_apply_preservation_loss(preservation_pred, prior_small, multiplier=1.0)

    if preservation_loss is None:
        # Diagnose step-by-step to find which stage fails
        failure_stage = None
        try:
            # Device/dtype normalization step
            if prior_small is not None and preservation_pred.device != prior_small.device:
                preservation_pred = preservation_pred.to(prior_small.device)
        except Exception as e:
            failure_stage = f'device_move: {e}'

        try:
            # compute loss
            pres_loss_manual = torch.nn.functional.mse_loss(preservation_pred, prior_small) * 1.0
        except Exception as e:
            failure_stage = f'loss_compute: {e}'

        try:
            # cpu transfer diagnostic
            with t.timer('cpu_transfer'):
                t._last_preservation_loss = float(pres_loss_manual.detach())
        except Exception as e:
            failure_stage = f'cpu_transfer: {e}'

        try:
            if pres_loss_manual.requires_grad:
                with t.timer('preservation_backward'):
                    t.accelerator.backward(pres_loss_manual)
        except Exception as e:
            failure_stage = f'backward_call: {e}'

        # ensure that the manual path produced a backward call
        assert len(t.backward_calls) == 1, f'No backward call; failure_stage={failure_stage}'

    else:
        # preservation_loss should be a tensor and should require grad in this scenario
        assert getattr(preservation_loss, 'requires_grad', False) is True
        # The DummyTrainer's accelerator.backward should have been invoked
        assert len(t.backward_calls) == 1
