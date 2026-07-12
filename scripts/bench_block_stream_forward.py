#!/usr/bin/env python3
"""Measure the forward effect of Slice 2 (per-block GPU staging) end to end.

Attaches a synthetic block-structured model to the real streaming path and times
the forward+backward pass with block streaming OFF (per-Linear H2D + one ring
event pair per Linear) vs ON (one transfer-stream burst + one ready event per
block). CUDA only. This exercises the actual _BouncingLinearFn / ring vs the
block-staging facility, unlike bench_bounce_fill_group.py which is worker-side.

Example:
    venv/Scripts/python scripts/bench_block_stream_forward.py --blocks 28 \
        --linears 8 --hidden 1536 --iters 30
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch

from toolkit.memory_management import MemoryManager
from toolkit.memory_management import manager_modules as mm


class Block(torch.nn.Module):
    def __init__(self, d, linears):
        super().__init__()
        self.lins = torch.nn.ModuleList(
            [torch.nn.Linear(d, d, bias=False) for _ in range(linears)]
        )

    def forward(self, x):
        for lin in self.lins:
            x = lin(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, d, n_blocks, linears):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block(d, linears) for _ in range(n_blocks)])
        self.head = torch.nn.Linear(d, d, bias=False)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)


def run(block_stream, *, d, n_blocks, linears, iters, warmup, batch, depth, device):
    torch.manual_seed(0)
    model = Model(d, n_blocks, linears)
    offload_ids = {id(m) for m in model.modules() if isinstance(m, torch.nn.Linear)}
    MemoryManager.attach(model, device, _offload_module_ids=offload_ids)
    handles = []
    try:
        if block_stream:
            mm.set_block_stream_enabled(device, True, depth=depth)
            mm.reset_block_stream(device)
            MemoryManager._wire_block_stream_forward_hooks(model, device)
            handles = getattr(model, "_mm_block_stream_handles", [])
        else:
            mm.set_block_stream_enabled(device, False)

        x = torch.randn(batch, d, device=device)
        times = []
        for i in range(warmup + iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            loss = out.square().mean()
            loss.backward()
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - t0)
            model.zero_grad(set_to_none=True)
        h2d, layers = mm.block_stream_stats(device) if block_stream else (0, 0)
        return statistics.median(times) * 1000.0, h2d, layers
    finally:
        for h in handles or []:
            h.remove()
        mm.set_block_stream_enabled(device, False)
        del model
        torch.cuda.empty_cache()


def run_ring_depth(ring_depth, *, d, n_blocks, linears, iters, warmup, batch, device):
    """Per-Linear ring (block staging OFF) at a given PIPELINE_DEPTH. Clears the
    cached device ring so the new depth takes effect."""
    mm._DEVICE_STATE.clear()
    mm.PIPELINE_DEPTH = ring_depth
    try:
        return run(
            False, d=d, n_blocks=n_blocks, linears=linears, iters=iters,
            warmup=warmup, batch=batch, depth=2, device=device,
        )
    finally:
        mm._DEVICE_STATE.clear()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", type=int, default=28)
    ap.add_argument("--linears", type=int, default=8, help="Linears per block")
    ap.add_argument("--hidden", type=int, default=1536)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--depth", type=int, default=2, help="block ring depth")
    ap.add_argument("--ring-depths", default="",
                    help="comma list of PIPELINE_DEPTH values to sweep (per-Linear ring); "
                         "if set, runs the depth sweep instead of the block-staging A/B")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    device = torch.device("cuda:0")
    per_layer_mb = args.hidden * args.hidden * 4 / 1024 ** 2
    print(
        f"device={torch.cuda.get_device_name(0)} blocks={args.blocks} "
        f"linears/block={args.linears} hidden={args.hidden} (fp32 {per_layer_mb:.1f} MB/layer) "
        f"batch={args.batch} iters={args.iters} depth={args.depth}"
    )
    common = dict(
        d=args.hidden, n_blocks=args.blocks, linears=args.linears,
        iters=args.iters, warmup=args.warmup, batch=args.batch,
        depth=args.depth, device=device,
    )
    off, _, _ = run(False, **common)
    on, h2d, layers = run(True, **common)
    n_layers = args.blocks * args.linears
    iters_total = args.iters + args.warmup
    print(f"\nstreamed layers/step = {n_layers}")
    print(f"  per-Linear staging (OFF): {off:8.2f} ms/step   H2D submits/step = {n_layers}")
    print(f"  block staging    (ON,d={args.depth}): {on:8.2f} ms/step   "
          f"H2D submits/step = {h2d / iters_total:.0f}  (covered {layers / iters_total:.0f} layers)")
    print(f"  H2D submit count: {n_layers} -> {h2d / iters_total:.0f} per step "
          f"({n_layers / max(1, h2d / iters_total):.1f}x fewer)")
    d = off - on
    print(f"  wall delta: {d:+.2f} ms ({d / off * 100:+.1f}%)")
    # PyTorch's CUDA event/stream finalization can segfault at interpreter exit
    # on Windows; we have our numbers, so flush and bypass the crashy teardown.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    # Serialize GPU scripts: two full-model runs on a 12 GB card do not just
    # measure badly, the second OOMs. See scripts/smoke_runtime.py.
    from smoke_runtime import run_locked

    sys.exit(run_locked("bench_block_stream_forward", main))
