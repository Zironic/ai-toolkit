"""Measure where LoRA-checkpoint save time actually goes, on the real GPU.

Motivation: saving a LoRA stalls training ~2-3s even though the payload is only
tens of MB. Is that the disk write, the device->host copy, or the surrounding
`empty_cache()` / `gc.collect()`? This bench rebuilds a real LoRA's tensors on
the GPU and times each phase, both idle and under simulated offload-stream
contention (which is the condition the stall actually happens under).

Run:
    venv/Scripts/python.exe scripts/bench_checkpoint_save.py \
        --lora "output/LA Jinx Krea the 7th/LA Jinx Krea the 7th_000001250.safetensors"

If --lora is omitted it synthesizes a Krea-ish LoRA (264 modules, rank 32).
"""

import argparse
import gc
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors.torch import save_file, load_file

from toolkit.async_save import AsyncSaver, atomic_save_file


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_from_real(path, device):
    """Load a real LoRA and re-materialize its tensors on `device` (fp32, as in training)."""
    cpu_sd = load_file(path)
    dev_sd = {}
    for k, v in cpu_sd.items():
        # training keeps trainable params in fp32 on-device; emulate that
        dev_sd[k] = v.to(device=device, dtype=torch.float32).contiguous()
    return dev_sd


def build_synthetic(num_modules, rank, dim, device):
    sd = {}
    for i in range(num_modules):
        sd[f"blk{i}.lora_down.weight"] = torch.randn(rank, dim, device=device)
        sd[f"blk{i}.lora_up.weight"] = torch.randn(dim, rank, device=device)
    return sd


def nbytes(sd):
    return sum(v.numel() * v.element_size() for v in sd.values())


# ---- the two device->host snapshot strategies ----------------------------

def d2h_per_param(dev_sd, dtype=torch.float16):
    """What get_state_dict does today: per-tensor .clone().to('cpu') -- each syncs."""
    out = {}
    for k, v in dev_sd.items():
        out[k] = v.detach().clone().to("cpu").to(dtype)
    return out


def d2h_batched_pinned(dev_sd, dtype=torch.float16):
    """Issue all D2H copies into pinned buffers, then one synchronize."""
    pinned = {}
    for k, v in dev_sd.items():
        dst = torch.empty(v.shape, dtype=v.dtype, device="cpu", pin_memory=True)
        dst.copy_(v, non_blocking=True)
        pinned[k] = dst
    _sync()
    # cast on CPU after the single sync
    return {k: t.to(dtype) for k, t in pinned.items()}


# ---- contention: keep the copy engine / PCIe busy like offload streaming --

class OffloadNoise:
    def __init__(self, device, mb=256):
        self._stop = threading.Event()
        self._device = device
        self._mb = mb
        self._t = None

    def __enter__(self):
        if not torch.cuda.is_available():
            return self
        n = (self._mb * 1024 * 1024) // 4
        host = torch.empty(n, dtype=torch.float32, pin_memory=True)
        dev = torch.empty(n, dtype=torch.float32, device=self._device)
        stream = torch.cuda.Stream()

        def loop():
            while not self._stop.is_set():
                with torch.cuda.stream(stream):
                    dev.copy_(host, non_blocking=True)
                    host.copy_(dev, non_blocking=True)
                stream.synchronize()

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        if self._t:
            self._t.join(2)


def timed(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        _sync()
        t0 = time.perf_counter()
        fn()
        _sync()
        best = min(best, time.perf_counter() - t0)
    return best


def flush_cost():
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return time.perf_counter() - t0


def run(dev_sd, label, tmpdir):
    print(f"\n=== {label} ===")
    print(f"tensors={len(dev_sd)}  payload={nbytes(dev_sd)/1e6:.1f} MB (fp32 on-device)")

    # phase timings
    t_perparam = timed(lambda: d2h_per_param(dev_sd))
    t_batched = timed(lambda: d2h_batched_pinned(dev_sd))

    snap = d2h_per_param(dev_sd)  # a real CPU snapshot to write
    save_path = os.path.join(tmpdir, "bench.safetensors")

    t_write_plain = timed(lambda: save_file(snap, save_path), repeat=3)
    t_write_atomic = timed(lambda: atomic_save_file(snap, save_path), repeat=3)
    t_flush = flush_cost()

    # OLD blocking path (what training pays today): perparam D2H + inline write
    old_total = t_perparam + t_write_plain
    # NEW path: training thread pays only the snapshot (batched); disk goes async
    new_blocking = t_batched

    print(f"  D2H per-param loop (current) : {t_perparam*1000:7.1f} ms")
    print(f"  D2H batched-pinned (proposed): {t_batched*1000:7.1f} ms")
    print(f"  disk write plain save_file   : {t_write_plain*1000:7.1f} ms")
    print(f"  disk write atomic (tmp+rename): {t_write_atomic*1000:7.1f} ms")
    print(f"  flush() empty_cache+gc       : {t_flush*1000:7.1f} ms")
    print(f"  ---")
    print(f"  OLD blocking (D2H+write)     : {old_total*1000:7.1f} ms")
    print(f"  NEW blocking (batched D2H)   : {new_blocking*1000:7.1f} ms  "
          f"-> disk {t_write_atomic*1000:.0f}ms moves off-thread")
    if new_blocking > 0:
        print(f"  training-thread speedup      : {old_total/new_blocking:5.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=None, help="real LoRA .safetensors to mirror shapes from")
    ap.add_argument("--modules", type=int, default=264)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--dim", type=int, default=3072)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: no CUDA; D2H timings will be meaningless.")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.lora and os.path.exists(args.lora):
        print(f"Mirroring shapes from {args.lora}")
        dev_sd = build_from_real(args.lora, device)
    else:
        print(f"Synthetic LoRA: {args.modules} modules, rank {args.rank}, dim {args.dim}")
        dev_sd = build_synthetic(args.modules, args.rank, args.dim, device)

    with tempfile.TemporaryDirectory() as tmp:
        run(dev_sd, "IDLE (no GPU contention)", tmp)
        with OffloadNoise(device):
            run(dev_sd, "UNDER OFFLOAD-STREAM CONTENTION", tmp)

    print("\nInterpretation: if the D2H loop dominates and grows under contention,")
    print("the stall is the synchronous copy, not the disk -- async write alone")
    print("won't fix it; the batched-pinned snapshot is what collapses it.")


if __name__ == "__main__":
    main()
