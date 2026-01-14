import types
import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummySDV:
    def __init__(self):
        # mimic vae.config with 4 blocks -> vae_scale = 8
        self.vae = types.SimpleNamespace(config={'block_out_channels': [64, 128, 256, 512]})
        self.transformer = types.SimpleNamespace(control_in_dim=33)
        self.calls = 0

    def encode_control_images_videox(self, imgs, height=None, width=None, tile=False):
        self.calls += len(imgs)
        outs = []
        for img in imgs:
            outs.append(torch.randn(4, 16, 16))
        return torch.stack(outs, dim=0)


class DummyTrainer(SDTrainer):
    def __init__(self):
        self.sd = DummySDV()
        self.model_config = types.SimpleNamespace(control_use_tiling=False, control_tiling_size=256, control_tiling_overlap=32)
        # minimal train_config stub
        self.train_config = types.SimpleNamespace(
            diff_output_preservation=False,
            diff_output_preservation_every=1,  # full-resolution schedule (default for backward compatibility in tests)
            diff_output_preservation_after_steps=0,
            diff_output_preservation_resolution=None,
            blank_prompt_preservation=False,
            blank_prompt_preservation_resolution=None,
        )


def test_dop_runs_at_reduced_resolution(monkeypatch):
    from toolkit.timer import Timer
    t = DummyTrainer()
    t.timer = Timer('test')

    # set config to request DOP at 256px (latent target long-side = 256/8 = 32)
    t.train_config.diff_output_preservation = True
    t.train_config.diff_output_preservation_after_steps = 0
    t.train_config.diff_output_preservation_resolution = 256

    # construct a full-res latent (H=64, W=64)
    noisy = torch.zeros((1, 4, 64, 64))
    prior = torch.zeros_like(noisy)
    timesteps = torch.tensor([0])
    unconditional_embeds = None
    batch = None

    # Provide device and timer on trainer
    t.device_torch = torch.device('cpu')

    # monkeypatch predict_noise to assert spatial size
    def fake_predict_noise(noisy_latents=None, **kwargs):
        # expect downsample to 32x32
        assert noisy_latents.shape[-2:] == (32, 32)
        return torch.zeros((1, 4, 32, 32))

    monkeypatch.setattr(t, 'predict_noise', fake_predict_noise)

    # Provide a minimal preservation_embeds with .to
    class DummyPreservationEmbeds:
        def to(self, device, dtype=None):
            return self

    result = t._run_preservation_forward(noisy, timesteps, DummyPreservationEmbeds(), unconditional_embeds, batch, {}, 'float32', prior, preservation_resolution=256)

    # since we downsampled, expect a tuple (preservation_pred, prior_small)
    assert isinstance(result, tuple)
    preservation_pred, prior_small = result
    assert preservation_pred.shape[-2:] == (32, 32)
    assert prior_small.shape[-2:] == (32, 32)


def test_dop_runs_at_128_resolution(monkeypatch):
    from toolkit.timer import Timer
    t = DummyTrainer()
    t.timer = Timer('test')

    # set config to request DOP at 128px (latent target long-side = 128/8 = 16)
    t.train_config.diff_output_preservation = True
    t.train_config.diff_output_preservation_after_steps = 0
    t.train_config.diff_output_preservation_resolution = 128

    noisy = torch.zeros((1, 4, 64, 64))
    prior = torch.zeros_like(noisy)
    timesteps = torch.tensor([0])
    unconditional_embeds = None
    batch = None

    t.device_torch = torch.device('cpu')

    def fake_predict_noise(noisy_latents=None, **kwargs):
        # expect downsample to 16x16
        assert noisy_latents.shape[-2:] == (16, 16)
        return torch.zeros((1, 4, 16, 16))

    monkeypatch.setattr(t, 'predict_noise', fake_predict_noise)

    class DummyPreservationEmbeds:
        def to(self, device, dtype=None):
            return self

    result = t._run_preservation_forward(noisy, timesteps, DummyPreservationEmbeds(), unconditional_embeds, batch, {}, 'float32', prior, preservation_resolution=128)
    assert isinstance(result, tuple)
    preservation_pred, prior_small = result
    assert preservation_pred.shape[-2:] == (16, 16)
    assert prior_small.shape[-2:] == (16, 16)