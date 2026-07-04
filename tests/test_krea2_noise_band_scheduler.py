import pytest
import torch

from extensions_built_in.diffusion_models.krea2.src.noise_band_scheduler import (
    Krea2NoiseBandScheduler,
)
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

# Mirrors extensions_built_in/diffusion_models/krea2/krea2.py's scheduler_config.
SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "max_image_seq_len": 6400,
    "base_shift": 0.5,
    "max_shift": 0.9,
    "min_shift": 0.33,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": True,
    "time_shift_type": "exponential",
}


def test_default_band_matches_base_scheduler():
    banded = Krea2NoiseBandScheduler(**SCHEDULER_CONFIG)
    base = CustomFlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG)

    banded_timesteps = banded.set_train_timesteps(1000, device="cpu", timestep_type="linear")
    base_timesteps = base.set_train_timesteps(1000, device="cpu", timestep_type="linear")

    assert torch.equal(banded_timesteps, base_timesteps)


def test_rejects_invalid_band_bounds():
    with pytest.raises(ValueError):
        Krea2NoiseBandScheduler(noise_band_min=0.85, noise_band_max=0.05, **SCHEDULER_CONFIG)
    with pytest.raises(ValueError):
        Krea2NoiseBandScheduler(noise_band_min=0.5, noise_band_max=0.5, **SCHEDULER_CONFIG)


def test_unsupported_timestep_type_raises_when_band_active():
    scheduler = Krea2NoiseBandScheduler(noise_band_min=0.05, noise_band_max=0.85, **SCHEDULER_CONFIG)

    with pytest.raises(NotImplementedError):
        scheduler.set_train_timesteps(1000, device="cpu", timestep_type="linear")

    with pytest.raises(NotImplementedError):
        scheduler.set_train_timesteps(1000, device="cpu", timestep_type="sigmoid")


def test_dynamic_shift_band_keeps_sigmas_consistent_with_timesteps():
    scheduler = Krea2NoiseBandScheduler(noise_band_min=0.05, noise_band_max=0.85, **SCHEDULER_CONFIG)
    latents = torch.zeros(1, 16, 32, 32)

    timesteps = scheduler.set_train_timesteps(
        50, device="cpu", timestep_type="shift", latents=latents, patch_size=1
    )

    # self.sigmas has one extra terminal entry appended past self.timesteps.
    assert scheduler.sigmas.numel() == timesteps.numel() + 1

    sigmas = scheduler.get_sigmas(timesteps, n_dim=1, dtype=torch.float32, device="cpu").flatten()

    assert (sigmas <= 0.85 + 1e-4).all()
    assert (sigmas >= 0.05 - 1e-4).all()
    assert torch.allclose(sigmas * 1000, timesteps, atol=1e-2)
