"""Quick compare of sigmoid timestep biases: balanced, high_noise, low_noise, mid_noise
Saves histograms and prints median/quantiles for comparison.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import math

ROOT_OUT = 'output/analysis/compare_sigmoid_biases'
os.makedirs(ROOT_OUT, exist_ok=True)

rng = np.random.default_rng(1234)
num = 200000
r = rng.standard_normal(num)

modes = [
    ('balanced', None),
    ('high_noise', -0.5),
    ('low_noise', 0.5),
    ('mid_noise', 0.4),
]

rows = []
for name, param in modes:
    if name == 'balanced':
        rr = r
    elif name == 'high_noise':
        rr = r - abs(param)
    elif name == 'low_noise':
        rr = r + abs(param)
    elif name == 'mid_noise':
        rr = r * param
    tvals = 1.0 / (1.0 + np.exp(-rr))
    base_sigmas = 1.0 - tvals

    rows.append((name, base_sigmas))

    plt.figure(figsize=(6,3))
    plt.hist(base_sigmas, bins=160, density=True, alpha=0.8)
    plt.title(f'sigmoid bias {name}')
    plt.xlabel('sigma')
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_OUT, f'hist_{name}.png'), dpi=150)
    plt.close()

    print(name)
    print(' median', np.median(base_sigmas))
    print(' q10', np.quantile(base_sigmas, 0.1), 'q90', np.quantile(base_sigmas,0.9))
    print()

print('Saved plots to', os.path.abspath(ROOT_OUT))
