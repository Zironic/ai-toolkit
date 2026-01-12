"""Compare token-count-based shift vs Qwen log-pixels shift for specified resolutions.
Saves histograms and a CSV summary in output/analysis/compare_shift_methods
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from toolkit.samplers.custom_flowmatch_sampler import calculate_shift


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def base_sigmas_from_sigmoid(num_samples=200000, seed=1234):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(num_samples)
    tvals = sigmoid(r)
    return 1.0 - tvals, tvals


def time_shift(mu: float, sigma: float, t: np.ndarray):
    # Vectorized implementation; t in (0,1)
    # y = exp(mu) / (exp(mu) + (1/t - 1)**sigma)
    em = math.exp(mu)
    return em / (em + np.power((1.0 / t - 1.0), sigma))


def multiplicative_shift(shift: float, s: np.ndarray):
    return (shift * s) / (1.0 + (shift - 1.0) * s)


def compute_and_save(resolutions, patch_size=8, shift_val=3.0, outdir='output/analysis/compare_shift_methods'):
    os.makedirs(outdir, exist_ok=True)
    base_sigmas, base_t = base_sigmas_from_sigmoid()

    rows = []
    for r in resolutions:
        h = r; w = r
        pixels = h * w
        seq_len = pixels // (patch_size * patch_size)

        # token-count-based mu (old behavior)
        mu_token = calculate_shift(seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16)
        shifted_token = time_shift(mu_token, 1.0, base_t)

        # Qwen log-pixels mu (new behavior, with clamping)
        mu_qwen = math.log(max(pixels, 1.0) / (1024.0 ** 2))
        mu_qwen = max(min(mu_qwen, 5.0), -5.0)
        shifted_qwen = time_shift(mu_qwen, 1.0, base_t)

        # multiplicative reference
        mult_shifted = multiplicative_shift(shift_val, base_sigmas)

        # stats helper
        def stats(arr):
            return {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'q10': float(np.quantile(arr, 0.1)),
                'q90': float(np.quantile(arr, 0.9)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }

        summary = {
            'res': r,
            'pixels': int(pixels),
            'patch_size': patch_size,
            'seq_len': int(seq_len),
            'mu_token': float(mu_token),
            'exp_mu_token': float(math.exp(mu_token)),
            'mu_qwen': float(mu_qwen),
            'exp_mu_qwen': float(math.exp(mu_qwen)),
            'base': stats(base_sigmas),
            'shifted_token': stats(shifted_token),
            'shifted_qwen': stats(shifted_qwen),
            'mult_shifted': stats(mult_shifted),
        }
        rows.append(summary)

        # plot
        plt.figure(figsize=(6,4))
        plt.hist(base_sigmas, bins=80, density=True, alpha=0.5, label='sigmoid (base)')
        plt.hist(shifted_token, bins=80, density=True, alpha=0.6, label=f'token_shift (mu={mu_token:.3f})')
        plt.hist(shifted_qwen, bins=80, density=True, alpha=0.6, label=f'qwen_shift (mu={mu_qwen:.3f})')
        plt.hist(mult_shifted, bins=80, density=True, alpha=0.3, label=f'mult_shift_{shift_val}')
        plt.xlabel('sigma')
        plt.ylabel('density')
        plt.title(f'sigma for {r}x{r} patch={patch_size} seq={seq_len}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'hist_{r}x{r}_patch{patch_size}.png'), dpi=150)
        plt.close()

    df = pd.json_normalize(rows)
    df.to_csv(os.path.join(outdir, 'compare_shift_methods_summary.csv'), index=False)
    print('Saved outputs to', os.path.abspath(outdir))
    return rows


if __name__ == '__main__':
    resolutions = [256, 512, 768, 1024]
    compute_and_save(resolutions)
