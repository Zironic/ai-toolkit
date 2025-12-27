"""Benchmark helper for ControlNet adapter offload timings.

Usage: python tools/benchmark_offload.py --strategy accelerate --size-mb 200 --iters 5

This script measures time to bring adapter to device, compute a dummy residual, and offload it back.
"""
import argparse
import time
import torch
from toolkit.controlnet_offload import offload_adapter, bring_adapter, compute_control_residuals
from toolkit.accelerator import get_accelerator


class LargeDummyAdapter(torch.nn.Module):
    def __init__(self, size_mb=100):
        super().__init__()
        num_elems = (size_mb * 1024 * 1024) // 4
        self.param = torch.nn.Parameter(torch.randn(num_elems))

    def forward(self, control, noisy_latents, timesteps):
        batch = noisy_latents.shape[0]
        return torch.zeros((batch, 4), device=noisy_latents.device)


def run_once(adapter, strategy, device, size_mb):
    control = torch.randn((4, 3, 64, 64))
    noisy = torch.randn((4, 4))
    timesteps = torch.tensor([1, 2, 3, 4])

    t0 = time.time()
    bring_adapter(adapter, device, strategy=strategy)
    t1 = time.time()

    res = compute_control_residuals(adapter, control.to(device), noisy.to(device), timesteps.to(device), device=device)
    t2 = time.time()

    offload_adapter(adapter, strategy=strategy)
    t3 = time.time()

    return t1 - t0, t2 - t1, t3 - t2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=['accelerate', 'manual_swap'], default='accelerate')
    parser.add_argument('--size-mb', type=int, default=100)
    parser.add_argument('--iters', type=int, default=3)
    args = parser.parse_args()

    acc = get_accelerator()
    device = acc.device

    print(f"Benchmarking offload strategy={args.strategy} size_mb={args.size_mb} on device={device}")

    adapter = LargeDummyAdapter(size_mb=args.size_mb)

    results = []
    for i in range(args.iters):
        print(f"Iteration {i+1}/{args.iters}")
        try:
            bring_t, compute_t, offload_t = run_once(adapter, args.strategy, device, args.size_mb)
            print(f" bring: {bring_t:.3f}s, compute: {compute_t:.3f}s, offload: {offload_t:.3f}s")
            results.append((bring_t, compute_t, offload_t))
        except Exception as e:
            print(f"Iteration failed: {e}")
            break

    if results:
        import numpy as np
        arr = np.array(results)
        print("Mean times (s): bring, compute, offload ->", arr.mean(axis=0))


if __name__ == '__main__':
    main()
