#!/usr/bin/env python3
"""Print noise standard deviation per timestep for a model's scheduler.

Usage: python tools/print_noise_scales.py --model <name_or_path> [--show N]
"""
import argparse
import numpy as np
import sys

from toolkit.model_utils import load_model_for_inference

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--show', type=int, default=20, help='how many timesteps to show at start')
args = parser.parse_args()

try:
    print(f"Loading model/scheduler for: {args.model} (CPU, float32). This may take a moment...")
    sd = load_model_for_inference(args.model, device='cpu', dtype='float32')
except Exception as e:
    print('Failed to load model:', e)
    print('Try providing a local model path or set HF_TOKEN in environment if model is private.')
    sys.exit(2)

ns = getattr(sd, 'noise_scheduler', None)
if ns is None:
    print('Model does not have a noise_scheduler attribute')
    sys.exit(2)

# Try common attributes
alphas_cumprod = None
if hasattr(ns, 'alphas_cumprod'):
    try:
        ac = ns.alphas_cumprod
        import torch
        if isinstance(ac, torch.Tensor):
            alphas_cumprod = ac.detach().cpu().numpy()
        else:
            alphas_cumprod = np.array(ac)
    except Exception as e:
        print('Could not read alphas_cumprod:', e)

# fallback: try betas
if alphas_cumprod is None and hasattr(ns, 'betas'):
    try:
        betas = ns.betas
        import torch
        if isinstance(betas, torch.Tensor):
            betas = betas.detach().cpu().numpy()
        betas = np.array(betas)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
    except Exception as e:
        print('Could not derive alphas_cumprod from betas:', e)

if alphas_cumprod is None:
    print('Could not determine alpha cumulative products for scheduler; aborting')
    sys.exit(2)

stds = np.sqrt(1.0 - alphas_cumprod)

print('\nTimestep stats:')
print(f'  Count: {len(stds)}')
print(f'  Min std: {stds.min():.6f}, Max std: {stds.max():.6f}, Mean std: {stds.mean():.6f}, Median std: {np.median(stds):.6f}')

show = min(args.show, len(stds))
print(f'\nFirst {show} timesteps (t, std):')
for i in range(show):
    print(f'  {i:3d}: {stds[i]:.6f}')

# If timesteps represent increasing noise (e.g., t=0 small noise), also print last few
print('\nLast 5 timesteps:')
for i in range(len(stds)-5, len(stds)):
    print(f'  {i:3d}: {stds[i]:.6f}')

# expected std if timestep chosen uniformly at random
print(f'\nExpected std for uniformly random timestep: {stds.mean():.6f}')

# show a few example indices with rounding
indices = np.linspace(0, len(stds)-1, num=min(10, len(stds))).astype(int)
print('\nSampled timesteps and stds:')
for idx in indices:
    print(f'  t={idx}, std={stds[idx]:.6f}')
