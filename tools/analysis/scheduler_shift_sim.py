"""Simulate scheduler sigma distributions across resolutions and patch sizes.

Saves histograms and CSV summary to output/analysis/scheduler_shift_sim
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

# ensure repo root on path so 'toolkit' package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from toolkit.samplers.custom_flowmatch_sampler import calculate_shift

# time_shift implementation copied from flash_flow_match
import math

def time_shift(mu: float, sigma: float, t: np.ndarray):
    # t expected in [0,1]
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def multiplicative_shift(shift: float, s: np.ndarray):
    return (shift * s) / (1.0 + (shift - 1.0) * s)


def simulate(num_samples=200000, resolutions=[256,512,768,1024], patch_sizes=[1,2,4,8,16], base_shift_vals=[1.0,3.0]):
    outdir = 'output/analysis/scheduler_shift_sim'
    os.makedirs(outdir, exist_ok=True)

    results = []
    rng = np.random.default_rng(1234)
    r = rng.standard_normal(num_samples)
    tvals = 1.0 / (1.0 + np.exp(-r))  # sigmoid
    base_sigmas = 1.0 - tvals  # as in code: sigma = 1 - sigmoid(r)

    for res in resolutions:
        h = res
        w = res
        for patch in patch_sizes:
            image_seq_len = (h * w) // (patch * patch)
            mu = calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16)
            # compute dynamic shifted sigmas
            shifted_sigmas = np.array([time_shift(mu, 1.0, s) for s in base_sigmas])
            for shift_val in base_shift_vals:
                mult_sigmas = multiplicative_shift(shift_val, base_sigmas)
                # compute stats
                for name, arr in [('base', base_sigmas), ('dynamic_shift', shifted_sigmas), (f'mult_shift_{shift_val}', mult_sigmas)]:
                    mean = float(np.mean(arr))
                    median = float(np.median(arr))
                    q10 = float(np.quantile(arr, 0.1))
                    q90 = float(np.quantile(arr, 0.9))
                    results.append({'res': res, 'patch': patch, 'image_seq_len': image_seq_len, 'mu': mu, 'scheme': name, 'shift_val': shift_val, 'mean': mean, 'median': median, 'q10': q10, 'q90': q90})

            # Save histograms for this resolution/patch
            plt.figure(figsize=(8,4))
            plt.hist(base_sigmas, bins=80, alpha=0.6, label='base', density=True)
            plt.hist(shifted_sigmas, bins=80, alpha=0.6, label=f'dynamic(mu={mu:.3f})', density=True)
            for shift_val in base_shift_vals:
                plt.hist(multiplicative_shift(shift_val, base_sigmas), bins=80, alpha=0.4, label=f'mult_shift_{shift_val}', density=True)
            plt.title(f'Res {res} Patch {patch} seq={image_seq_len} mu={mu:.3f}')
            plt.xlabel('sigma')
            plt.ylabel('density')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'hist_res{res}_patch{patch}.png'), dpi=150)
            plt.close()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(outdir, 'scheduler_shift_summary.csv'), index=False)
    print('Saved summary to', os.path.abspath(outdir))


if __name__ == '__main__':
    simulate()
