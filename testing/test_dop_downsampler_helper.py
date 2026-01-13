import torch
import types
import importlib.util
import pathlib
import pytest

spec = importlib.util.spec_from_file_location("sd_trainer_module", str(pathlib.Path(__file__).resolve().parents[1] / "extensions_built_in/sd_trainer/SDTrainer.py"))
sd_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd_module)
SDTrainer = sd_module.SDTrainer


class DummySD:
    def __init__(self):
        self.called = False
        self.last_args = None

    def add_noise(self, latents, noise, timesteps):
        self.called = True
        self.last_args = (latents.clone(), noise.clone(), timesteps.clone())
        # return a deterministic transformation for verification
        return latents + 5.0


class DummySDRaises(DummySD):
    def add_noise(self, latents, noise, timesteps):
        raise RuntimeError("forced failure")


class DummyTrainer(SDTrainer):
    def __init__(self):
        # minimal attributes required by helper
        self.sd = DummySD()
        self.device_torch = torch.device('cpu')
        self.train_config = types.SimpleNamespace(debug_dump_dop=True)


class DummyTrainerFallback(SDTrainer):
    def __init__(self):
        self.sd = DummySDRaises()
        self.device_torch = torch.device('cpu')
        self.train_config = types.SimpleNamespace(debug_dump_dop=False)


def test_create_downsampled_noisy_latents_scheduler_path():
    t = DummyTrainer()
    # create a fake batch latents (B,C,H,W)
    orig = torch.zeros((1, 4, 32, 32))
    timesteps = torch.tensor([100])

    latents_small, noise_small, noisy_small = t._create_downsampled_noisy_latents(orig, timesteps, 16, 16, 'float32')

    assert latents_small.shape == (1, 4, 16, 16)
    assert noise_small.shape == latents_small.shape
    # DummySD returns latents + 5
    assert torch.allclose(noisy_small, latents_small + 5.0)
    # ensure add_noise was called with expected args
    assert t.sd.called is True
    lat_arg, noise_arg, ts_arg = t.sd.last_args
    assert lat_arg.shape == latents_small.shape
    assert noise_arg.shape == latents_small.shape
    assert ts_arg.shape[0] == 1


def test_create_downsampled_noisy_latents_fallback_path():
    t = DummyTrainerFallback()
    orig = torch.ones((2, 4, 40, 36))  # test non-square dims
    timesteps = torch.tensor([50, 50])

    with pytest.raises(RuntimeError, match=r"failed to construct noisy_small"):
        t._create_downsampled_noisy_latents(orig, timesteps, 18, 14, 'float32')
