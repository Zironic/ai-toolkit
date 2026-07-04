import argparse
import gc
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from toolkit.memory_management.manager import MemoryManager
from toolkit.memory_management.manager_modules import LinearLayerMemoryManager, unpin_layer
from toolkit.memory_management import bounce_pool, dxgi_meminfo

GIB = 1024 ** 3


def _gib(value):
    if value is None:
        return "n/a"
    return f"{int(value) / GIB:.3f} GiB"


def _query(label):
    info = dxgi_meminfo.query_non_local_video_memory_info(min_interval_s=0.0)
    if info is None:
        print(f"{label}: DXGI unavailable")
        return None
    print(
        f"{label}: usage={_gib(info.current_usage_bytes)} "
        f"budget={_gib(info.budget_bytes)} "
        f"headroom={_gib(info.budget_bytes - info.current_usage_bytes)}"
    )
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Measure DXGI NON_LOCAL movement for MemoryManager pin/unpin_layer."
    )
    parser.add_argument("--gib", type=float, default=0.25, help="Approximate layer weight size to pin.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for pinned-memory probe")
    adapter = dxgi_meminfo.selected_adapter_info()
    if adapter is None:
        raise SystemExit("DXGI adapter unavailable")
    print(
        f"Adapter: index={adapter.index} description={adapter.description!r} "
        f"match={adapter.match_method} safe_for_control={adapter.safe_for_control}"
    )

    target_bytes = max(1, int(args.gib * GIB))
    elems = max(1, target_bytes // torch.empty((), dtype=torch.float32).element_size())
    side = max(1, int(math.sqrt(elems)))
    actual_bytes = side * side * torch.empty((), dtype=torch.float32).element_size()
    print(f"Synthetic Linear weight: {side}x{side} ({_gib(actual_bytes)})")

    before = _query("before")
    manager = None
    lin = None
    try:
        lin = torch.nn.Linear(side, side, bias=False, device="cpu", dtype=torch.float32)
        manager = MemoryManager(lin, torch.device(args.device), pinned_weight_gib=max(args.gib * 2, 0.01))
        LinearLayerMemoryManager.attach(lin, manager)
        time.sleep(0.25)
        after_pin = _query("after_pin")
        released = unpin_layer(lin)
        gc.collect()
        after_unpin = None
        for attempt in range(20):
            time.sleep(0.25)
            after_unpin = _query(f"after_unpin[{attempt + 1}]")
            if after_pin is None or after_unpin is None:
                break
            if after_unpin.current_usage_bytes < after_pin.current_usage_bytes:
                break
        print(
            f"manager_pinned={_gib(manager.pinned_weight_bytes)} "
            f"layer_released={_gib(released)} "
            f"ledger={_gib(bounce_pool._pinned_bytes_total)}"
        )
        if before is not None and after_pin is not None:
            print(
                "pin_delta="
                f"{_gib(after_pin.current_usage_bytes - before.current_usage_bytes)}"
            )
        if after_pin is not None and after_unpin is not None:
            print(
                "unpin_delta="
                f"{_gib(after_unpin.current_usage_bytes - after_pin.current_usage_bytes)}"
            )
    finally:
        if lin is not None:
            try:
                unpin_layer(lin)
            except Exception:
                pass
        del lin
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
