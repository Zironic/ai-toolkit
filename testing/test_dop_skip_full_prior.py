import torch
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


class DummySDV:
    def __init__(self):
        self.vae = type('T', (), {'config': {'block_out_channels': [64, 128, 256, 512]}})()
        self.transformer = type('Tr', (), {'all_patch_size': [1]})()


class DummyTrainer(SDTrainer):
    def __init__(self):
        # avoid calling Base ctor
        # set minimal attributes used by helper
        self.sd = DummySDV()
        self.train_config = type('C', (), {})()
        # default flags
        self.train_config.do_prior_divergence = False
        self.train_config.inverted_mask_prior = False
        self.train_config.correct_pred_norm = False
        self.device_torch = torch.device('cpu')


def test_should_skip_full_prior_for_dop_downsampling():
    t = DummyTrainer()
    # request a downsample resolution of 256 (vae_scale 8 -> target_long=32 < current 64)
    noisy = torch.zeros((1, 4, 64, 64))
    assert t._should_skip_full_prior(noisy, preservation_resolution=256) is True


def test_should_not_skip_when_do_prior_divergence_enabled():
    t = DummyTrainer()
    noisy = torch.zeros((1, 4, 64, 64))
    t.train_config.do_prior_divergence = True
    assert t._should_skip_full_prior(noisy, preservation_resolution=256) is False


def test_should_not_skip_when_inverted_mask_prior_enabled():
    t = DummyTrainer()
    noisy = torch.zeros((1, 4, 64, 64))
    t.train_config.inverted_mask_prior = True
    assert t._should_skip_full_prior(noisy, preservation_resolution=256) is False


def test_should_not_skip_when_correct_pred_norm_enabled():
    t = DummyTrainer()
    noisy = torch.zeros((1, 4, 64, 64))
    t.train_config.correct_pred_norm = True
    assert t._should_skip_full_prior(noisy, preservation_resolution=256) is False


def test_should_not_skip_when_preservation_resolution_none():
    t = DummyTrainer()
    noisy = torch.zeros((1, 4, 64, 64))
    assert t._should_skip_full_prior(noisy, preservation_resolution=None) is False


def test_should_not_skip_when_target_not_downsampling():
    t = DummyTrainer()
    noisy = torch.zeros((1, 4, 16, 16))
    # target long side equals or larger: 256 -> latent 32 >= 16 -> will not downsample
    assert t._should_skip_full_prior(noisy, preservation_resolution=256) is False