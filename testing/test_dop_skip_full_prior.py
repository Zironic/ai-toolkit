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


def test_run_preservation_forward_computes_reduced_prior_when_none(monkeypatch):
    t = DummyTrainer()
    t.device_torch = torch.device('cpu')

    noisy = torch.zeros((1, 4, 64, 64))
    timesteps = torch.tensor([0])

    class DummyPreservationEmbeds:
        def to(self, device, dtype=None):
            return self

    preservation_embeds = DummyPreservationEmbeds()
    unconditional_embeds = None
    batch = None
    pred_kwargs = {}

    # provide a lightweight Timer used by helpers
    from toolkit.timer import Timer
    t.timer = Timer('test')

    # fake get_prior_prediction to assert it's called with downsampled latents and return a small prior
    def fake_get_prior_prediction(noisy_latents=None, **kwargs):
        assert noisy_latents.shape[-2:] == (32, 32)
        return torch.zeros((1, 4, 32, 32))

    monkeypatch.setattr(t, 'get_prior_prediction', fake_get_prior_prediction)

    # fake predict_noise to assert it's called at reduced size
    def fake_predict_noise(noisy_latents=None, **kwargs):
        assert noisy_latents.shape[-2:] == (32, 32)
        return torch.zeros((1, 4, 32, 32))

    monkeypatch.setattr(t, 'predict_noise', fake_predict_noise)

    result = t._run_preservation_forward(
        noisy,
        timesteps,
        preservation_embeds,
        unconditional_embeds,
        batch,
        pred_kwargs,
        'float32',
        prior_pred=None,
        preservation_resolution=256,
        preservation_kind='dop',
        match_adapter_assist=False,
        network_weight_list=[]
    )

    assert isinstance(result, tuple)
    preservation_pred, prior_small = result
    assert preservation_pred.shape[-2:] == (32, 32)
    assert prior_small.shape[-2:] == (32, 32)


def test_do_not_skip_on_full_res_schedule():
    """If a full-resolution schedule is configured and we're on a scheduled step,
    _should_skip_full_prior should return False (i.e., do full-resolution prior).
    """
    t = DummyTrainer()
    noisy = torch.zeros((1, 4, 64, 64))
    # choose preservation_resolution that would normally downsample
    # configure full-resolution every 2 steps and set current batch=2
    t.train_config.diff_output_preservation_every = 2
    t._total_batch_count = 2
    assert t._should_skip_full_prior(noisy, preservation_resolution=256) is False
