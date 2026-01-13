"""Create combined visualizations comparing sigmoid timestep bias presets.
Saves:
 - output/analysis/compare_sigmoid_biases/grid.png  (2x2 grid of histograms)
 - output/analysis/compare_sigmoid_biases/overlay.png (all four overlaid + quantile lines)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT = 'output/analysis/compare_sigmoid_biases'
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(1234)
num = 200000
r = rng.standard_normal(num)

modes = [
    ('balanced', None),
    ('high_noise', -0.5),
    ('low_noise', 0.5),
    ('mid_noise', 0.4),
]

results = {}
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
    sigmas = 1.0 - tvals
    results[name] = sigmas

# 2x2 grid
fig, axs = plt.subplots(2,2, figsize=(10,6))
for ax, (name, arr) in zip(axs.flatten(), results.items()):
    ax.hist(arr, bins=160, density=True, color='#4c72b0', alpha=0.85)
    med = np.median(arr)
    q10 = np.quantile(arr, 0.1)
    q90 = np.quantile(arr, 0.9)
    ax.axvline(med, color='k', linestyle='-', linewidth=1)
    ax.axvline(q10, color='k', linestyle='--', linewidth=0.8)
    ax.axvline(q90, color='k', linestyle='--', linewidth=0.8)
    ax.set_title(f"{name} (median={med:.3f})")
    ax.set_xlabel('sigma')
    ax.set_ylabel('density')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'grid.png'), dpi=200)
plt.close()

# Overlay plot
plt.figure(figsize=(8,5))
colors = {'balanced':'#4c72b0','high_noise':'#dd8452','low_noise':'#55a868','mid_noise':'#c44e52'}
for name, arr in results.items():
    plt.hist(arr, bins=240, density=True, alpha=0.35, label=f"{name} (med={np.median(arr):.3f})", color=colors.get(name))
    med = np.median(arr)
    q10 = np.quantile(arr,0.1)
    q90 = np.quantile(arr,0.9)
    plt.axvline(med, color=colors.get(name), linestyle='-', linewidth=1)
    plt.axvline(q10, color=colors.get(name), linestyle=':', linewidth=0.7)
    plt.axvline(q90, color=colors.get(name), linestyle=':', linewidth=0.7)

plt.xlabel('sigma')
plt.ylabel('density')
plt.title('Sigmoid bias modes: overlay')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'overlay.png'), dpi=200)
plt.close()

print('Saved grid and overlay to', os.path.abspath(OUT))
