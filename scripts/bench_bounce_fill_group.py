#!/usr/bin/env python3
"""Benchmark the worker-side cost of ``fill_group_size`` in PinnedBouncePool.

Drives the *real* bounce pool with real pinned host buffers and real H2D copies
on CUDA, modelling the training thread: each step does step_begin then one
``acquire`` + H2D per layer (what ``_stage_forward_weight`` does per Linear),
while the background workers stage upcoming layers from pinned buffers. Sweeps
``fill_group_size`` so you can see what block-batched fills change.

What this captures: worker lock/CV/scheduling overhead and its contention with
the consumer, on real pinned memory + PCIe. What it does NOT capture: the
gradient/optimizer/activation pressure of a real step, the Windows pageable->
pinned pagefault stall under VRAM pressure, or torch.compile. Treat it as a
lower bound on the worker-side effect; the real verdict is the krea2 A/B via
``digest_perf_log.py`` (worker_fills + offload-profile submit_s).

Example:
    venv/Scripts/python scripts/bench_bounce_fill_group.py \
        --layers 224 --block-size 8 --hidden 2048 --steps 40
"""

from __future__ import annotations

import argparse
import statistics
import time

import sys

import torch

from toolkit.memory_management import bounce_pool


def _wait_until(predicate, timeout=30.0):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return False


def build_sources(num_layers: int, hidden: int, dtype):
    # One Linear per "layer"; weight numel = hidden*hidden so per-layer bytes are
    # representative of a transformer projection. Kept as a list so the pool's
    # weakref sources stay alive for the whole run.
    return [torch.nn.Linear(hidden, hidden, bias=False).to(dtype) for _ in range(num_layers)]


def run_one(group, *, num_layers, hidden, steps, warmup, lookahead, workers, dtype, device):
    per_layer_bytes = hidden * hidden * torch.empty((), dtype=dtype).element_size()
    budget = int(per_layer_bytes * num_layers * 1.2)  # never the limiter
    pool = bounce_pool.PinnedBouncePool(
        device,
        budget_bytes=budget,
        lookahead=lookahead,
        target_ready_bytes=budget,
        num_workers=workers,
        ram_floor_bytes=0,
        fill_group_size=group,
    )
    modules = build_sources(num_layers, hidden, dtype)
    keys = [f"l{i}" for i in range(num_layers)]
    for k, m in zip(keys, modules):
        pool.register_source(k, m)
    pool.set_schedule(keys)

    stream = torch.cuda.Stream(device=device)
    step_times = []
    try:
        for step in range(warmup + steps):
            pool.step_begin(warmup_bytes=0, warmup_timeout_s=0.0)
            if step == warmup:
                pool.stats(reset=True)
            t0 = time.perf_counter()
            with torch.cuda.stream(stream):
                for k, m in zip(keys, modules):
                    w, b, ticket = pool.acquire(k, m.weight.data, None, operation="forward")
                    w_gpu = w.to(device, non_blocking=True)  # the real H2D
                    if ticket is not None:
                        pool.on_h2d_submitted(ticket, stream)
                    del w_gpu
            stream.synchronize()
            dt = time.perf_counter() - t0
            if step >= warmup:
                step_times.append(dt)
        stats = pool.stats(reset=False)
    finally:
        pool.shutdown()

    steps_done = max(1, len(step_times))
    return {
        "group": group,
        "median_step_ms": statistics.median(step_times) * 1000.0,
        "p10_step_ms": (sorted(step_times)[len(step_times) // 10]) * 1000.0,
        "fills": stats["fills"],
        "fill_batches": stats["fill_batches"],
        "fills_per_batch": stats["fills_per_batch"],
        "batches_per_step": stats["fill_batches"] / steps_done,
        "hit_rate": stats["hit_rate"],
        "hard_misses": stats["hard_misses"],
        "soft_misses": stats["soft_misses"],
        "cpu_wait_s": stats["cpu_wait_s"],
        "copy_s": stats["copy_s"],
        "copy_gbps": stats["copy_gbps"],
        "per_layer_mb": per_layer_bytes / 1024 ** 2,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=224, help="total streamed layers/step (28 blocks x 8)")
    ap.add_argument("--block-size", type=int, default=8, help="Linears per block = batched group size")
    ap.add_argument("--hidden", type=int, default=2048, help="Linear in/out dim (sets per-layer bytes)")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--lookahead", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this benchmark.")
    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)

    print(
        f"device={torch.cuda.get_device_name(0)} layers={args.layers} "
        f"block_size={args.block_size} hidden={args.hidden} dtype={args.dtype} "
        f"workers={args.workers} steps={args.steps}"
    )
    rows = []
    for group in (1, args.block_size):
        rows.append(run_one(
            group,
            num_layers=args.layers, hidden=args.hidden, steps=args.steps,
            warmup=args.warmup, lookahead=args.lookahead, workers=args.workers,
            dtype=dtype, device=device,
        ))
        time.sleep(0.2)

    print(f"\nper-layer={rows[0]['per_layer_mb']:.1f} MB  total/step={rows[0]['per_layer_mb'] * args.layers / 1024:.2f} GiB")
    hdr = (
        f"{'group':>6} {'step_ms(med)':>13} {'step_ms(p10)':>13} "
        f"{'batches/step':>13} {'fills/batch':>12} {'hit%':>6} "
        f"{'hard':>5} {'soft':>5} {'cpu_wait_s':>11} {'copy_GBps':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['group']:>6} {r['median_step_ms']:>13.2f} {r['p10_step_ms']:>13.2f} "
            f"{r['batches_per_step']:>13.1f} {r['fills_per_batch']:>12.2f} "
            f"{r['hit_rate'] * 100:>6.1f} {r['hard_misses']:>5} {r['soft_misses']:>5} "
            f"{r['cpu_wait_s']:>11.3f} {r['copy_gbps']:>10.2f}"
        )
    base, blk = rows[0], rows[-1]
    if base["batches_per_step"] and blk["batches_per_step"]:
        print(
            f"\nworker lock-cycles/step: {base['batches_per_step']:.0f} -> "
            f"{blk['batches_per_step']:.0f} "
            f"({base['batches_per_step'] / max(1e-9, blk['batches_per_step']):.1f}x fewer)"
        )
    d = base["median_step_ms"] - blk["median_step_ms"]
    print(f"median step time: {base['median_step_ms']:.2f} -> {blk['median_step_ms']:.2f} ms "
          f"({d:+.2f} ms, {d / max(1e-9, base['median_step_ms']) * 100:+.1f}%)")


if __name__ == "__main__":
    # Serialize GPU scripts: two full-model runs on a 12 GB card do not just
    # measure badly, the second OOMs. See scripts/smoke_runtime.py.
    from smoke_runtime import run_locked

    sys.exit(run_locked("bench_bounce_fill_group", main))
