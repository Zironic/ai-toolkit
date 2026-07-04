"""Noise-band compression wrapper around Krea 2's training-timestep scheduler.

Affinely remaps the post-shift noise-fraction ladder into a bounded band
``[noise_band_min, noise_band_max]`` instead of letting it span the full 0-1
noise range. This runs *after* ``CustomFlowMatchEulerDiscreteScheduler`` has
already applied Krea 2's per-batch, resolution-dependent dynamic shift (``mu``
depends on ``image_seq_len``), so the band is an actual noise-fraction bound
regardless of which resolution bucket a batch lands in -- unlike clamping raw
ladder indices via ``min_denoising_steps``/``max_denoising_steps``, which only
approximates a fraction bound at a single fixed resolution.

Only ``timestep_type in ('shift', 'flux_shift', 'lumina2_shift')`` is
supported when a band is active: that is the one branch of the base
scheduler's ``set_train_timesteps`` that keeps ``self.sigmas`` derived from
``self.timesteps`` (rather than left over from ``__init__``'s default
schedule), so it is the only case where remapping ``self.sigmas`` alongside
``self.timesteps`` is actually correct.
"""

from typing import Optional

import torch

from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

_SIGMA_SYNCED_TIMESTEP_TYPES = ('shift', 'flux_shift', 'lumina2_shift')


class Krea2NoiseBandScheduler(CustomFlowMatchEulerDiscreteScheduler):
    def __init__(self, *args, noise_band_min: float = 0.0, noise_band_max: float = 1.0, **kwargs):
        if not (0.0 <= noise_band_min < noise_band_max <= 1.0):
            raise ValueError(
                "noise_band_min/noise_band_max must satisfy 0 <= min < max <= 1, "
                f"got min={noise_band_min}, max={noise_band_max}"
            )
        self.noise_band_min = float(noise_band_min)
        self.noise_band_max = float(noise_band_max)
        super().__init__(*args, **kwargs)

    def set_train_timesteps(
        self,
        num_timesteps,
        device,
        timestep_type: str = 'linear',
        latents: Optional[torch.Tensor] = None,
        patch_size: int = 1,
    ):
        timesteps = super().set_train_timesteps(
            num_timesteps,
            device,
            timestep_type=timestep_type,
            latents=latents,
            patch_size=patch_size,
        )

        band_min, band_max = self.noise_band_min, self.noise_band_max
        if band_min == 0.0 and band_max == 1.0:
            return timesteps

        if timestep_type not in _SIGMA_SYNCED_TIMESTEP_TYPES:
            raise NotImplementedError(
                f"Krea2NoiseBandScheduler does not support timestep_type={timestep_type!r} "
                f"with a noise band active; self.sigmas would go stale relative to the "
                f"compressed timesteps. Use timestep_type in {_SIGMA_SYNCED_TIMESTEP_TYPES} "
                "or leave noise_band_min/noise_band_max at their defaults."
            )

        num_train_timesteps = self.config.num_train_timesteps

        frac = self.timesteps / num_train_timesteps
        self.timesteps = ((band_min + (band_max - band_min) * frac) * num_train_timesteps).to(device=device)

        # This branch populates self.sigmas as timesteps/num_train_timesteps with
        # one extra terminal entry appended (diffusers convention). Rebuild the
        # body directly from the just-remapped timesteps so get_sigmas() lookups
        # stay consistent; leave the terminal entry untouched.
        self.sigmas = torch.cat([self.timesteps / num_train_timesteps, self.sigmas[-1:]])

        return self.timesteps
