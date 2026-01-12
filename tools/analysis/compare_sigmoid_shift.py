"""Compare sigmoid timestep sampling vs shift (dynamic and multiplicative) for a target resolution.
Saves histograms and a CSV summary in output/analysis/compare_sigmoid_shift
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ensure repo root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from toolkit.samplers.custom_flowmatch_sampler import calculate_shift
import math


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def base_sigmas_from_sigmoid(num_samples=200000, seed=1234):
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(num_samples)
    tvals = sigmoid(r)
    return 1.0 - tvals


def multiplicative_shift(shift: float, s: np.ndarray):
    return (shift * s) / (1.0 + (shift - 1.0) * s)


def time_shift(mu: float, sigma: float, t: np.ndarray):
    # t in (0,1)
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def analyze_for_resolution(h, w, patch_size=8, shift_val=3.0, num_samples=200000, outdir='output/analysis/compare_sigmoid_shift'):
    os.makedirs(outdir, exist_ok=True)
    image_seq_len = (h * w) // (patch_size * patch_size)
    mu = calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096, base_shift=0.5, max_shift=1.16)

    base = base_sigmas_from_sigmoid(num_samples=num_samples)
    dyn = np.array([time_shift(mu, 1.0, t) for t in (1.0 - base)])  # careful: time_shift expects t in (0,1) where t corresponds to base sigma mapping; replicate used earlier
    # The usage here mirrors earlier simulation: base_sigmas = 1-sigmoid(r) so t = 1-base_sigmas = sigmoid(r)
    dyn_correct = np.array([time_shift(mu, 1.0, s) for s in (1.0 - base)])
    # but we want sigma after dynamic shift; earlier we computed shifted_sigmas = time_shift(mu,1.0,base_sigmas?) replicate earlier behavior exactly
    # compute shifted from base t values: base_t = 1 - base (i.e., sigmoid(r)), shift applies to base_t; final sigma = shifted t (which corresponds directly)
    shifted_sigmas = dyn  # alias for clarity

    mult_shifted = multiplicative_shift(shift_val, base)

    # Stats
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
        'h': h,
        'w': w,
        'patch_size': patch_size,
        'image_seq_len': image_seq_len,
        'mu': mu,
        'base': stats(base),
        'dynamic_shift': stats(shifted_sigmas),
        f'multiplicative_shift_{shift_val}': stats(mult_shifted),
    }

    # Save histograms
    plt.figure(figsize=(6,4))
    plt.hist(base, bins=80, density=True, alpha=0.6, label='sigmoid (base)')
    plt.hist(shifted_sigmas, bins=80, density=True, alpha=0.6, label=f'shift (dynamic mu={mu:.3f})')
    plt.hist(mult_shifted, bins=80, density=True, alpha=0.4, label=f'shift mult (k={shift_val})')
    plt.xlabel('sigma')
    plt.ylabel('density')
    plt.title(f'sigma distribution for {h}x{w}, patch={patch_size} (seq_len={image_seq_len})')
    plt.legend()
    png_path = os.path.join(outdir, f'hist_{h}x{w}_patch{patch_size}.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # Save CSV of summary
    df = pd.DataFrame([{**{'metric': k}, **({'value': v} if isinstance(v, (int,float,str)) else v)} for k,v in summary.items() if k in ['base','dynamic_shift', f'multiplicative_shift_{shift_val}'] for k2,v2 in ([ (k + '_' + k2, v2) for k2,v2 in v.items() ])])
    # Simpler: save a JSON-like CSV
    outcsv = os.path.join(outdir, f'summary_{h}x{w}_patch{patch_size}.csv')
    pd.DataFrame([summary]).to_csv(outcsv, index=False)

    print('Saved:', png_path, outcsv)
    return summary, png_path, outcsv


if __name__ == '__main__':
    s, p, c = analyze_for_resolution(512, 512, patch_size=8, shift_val=3.0)
    print('Summary:')
    print(s)
