import argparse
import os
import sys
import time

try:
    import psutil
except Exception:
    psutil = None

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from toolkit.memory_management import bounce_pool, dxgi_meminfo

GIB = 1024 ** 3


def _gib(value):
    if value is None:
        return "n/a"
    return f"{int(value) / GIB:.2f} GiB"


def _print_memory(label, info):
    if info is None:
        print(f"{label}: unavailable")
        return
    print(
        f"{label}: budget={_gib(info.budget_bytes)} "
        f"usage={_gib(info.current_usage_bytes)} "
        f"available_for_reservation={_gib(info.available_for_reservation_bytes)} "
        f"current_reservation={_gib(info.current_reservation_bytes)}"
    )


def _old_proxy_ceiling():
    if psutil is None:
        return None
    try:
        fraction = float(os.environ.get("AI_TOOLKIT_PINNED_WEIGHT_WDDM_FRACTION", "0.25"))
        if fraction <= 0:
            return None
        return int(psutil.virtual_memory().total * fraction)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Probe DXGI video-memory budget readings.")
    parser.add_argument("--pin-gib", type=float, default=1.0)
    parser.add_argument("--no-pin", action="store_true")
    args = parser.parse_args()

    print("DXGI adapters:")
    adapters = dxgi_meminfo.enumerate_adapters()
    if not adapters:
        print("  unavailable")
    for adapter in adapters:
        kind = "software" if adapter.is_software else "hardware"
        print(
            f"  [{adapter.index}] {adapter.description!r} {kind} "
            f"vendor=0x{adapter.vendor_id:04x} device=0x{adapter.device_id:04x} "
            f"luid={adapter.luid} dedicated={_gib(adapter.dedicated_video_memory_bytes)} "
            f"shared={_gib(adapter.shared_system_memory_bytes)}"
        )

    selected = dxgi_meminfo.selected_adapter_info()
    if selected is None:
        print("Selected adapter: unavailable")
    else:
        print(
            f"Selected adapter: index={selected.index} description={selected.description!r} "
            f"vendor=0x{selected.vendor_id:04x} device=0x{selected.device_id:04x} "
            f"luid={selected.luid} match={selected.match_method}"
        )

    if psutil is not None:
        print(f"System RAM total: {_gib(psutil.virtual_memory().total)}")
    else:
        print("System RAM total: unavailable (psutil missing)")
    print(f"Old proxy ceiling: {_gib(_old_proxy_ceiling())}")
    print(f"Pinned ledger total: {_gib(bounce_pool._pinned_bytes_total)}")
    print(f"Spill reserve: {_gib(bounce_pool.dxgi_spill_reserve_bytes())}")

    before_local = dxgi_meminfo.query_local_video_memory_info(min_interval_s=0.0)
    before_nonlocal = dxgi_meminfo.query_non_local_video_memory_info(min_interval_s=0.0)
    _print_memory("LOCAL before", before_local)
    _print_memory("NON_LOCAL before", before_nonlocal)
    if before_nonlocal is not None:
        headroom = dxgi_meminfo.compute_non_local_headroom_bytes(
            before_nonlocal.budget_bytes,
            before_nonlocal.current_usage_bytes,
            bounce_pool.dxgi_spill_reserve_bytes(),
        )
        print(f"NON_LOCAL headroom after spill reserve: {_gib(headroom)}")

    if args.no_pin or args.pin_gib <= 0:
        print("Pinned allocation delta: skipped")
        return

    nbytes = int(args.pin_gib * GIB)
    print(f"Allocating pinned tensor: {_gib(nbytes)}")
    tensor = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    tensor[0] = 1
    time.sleep(0.25)
    after_nonlocal = dxgi_meminfo.query_non_local_video_memory_info(min_interval_s=0.0)
    _print_memory("NON_LOCAL after", after_nonlocal)
    if before_nonlocal is not None and after_nonlocal is not None:
        delta = after_nonlocal.current_usage_bytes - before_nonlocal.current_usage_bytes
        print(f"NON_LOCAL CurrentUsage delta: {_gib(delta)}")
    del tensor
    print(
        "Note: DXGI CurrentUsage is process-scoped; compare Budget, not usage, "
        "against Task Manager shared GPU memory figures."
    )


if __name__ == "__main__":
    main()
