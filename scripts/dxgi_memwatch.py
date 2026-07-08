"""Standalone GPU memory watcher: DXGI budgets + nvidia-smi usage, JSONL out.

Run it in a SEPARATE terminal next to a training/smoke run so humans and
agents share the same live external view of the two-cliff state:

    venv/Scripts/python.exe scripts/dxgi_memwatch.py
    venv/Scripts/python.exe scripts/dxgi_memwatch.py --out .codex/memwatch.jsonl --interval 2

Design constraints (why this shape):
  * NO torch / CUDA context: a watcher that imports torch.cuda would itself
    commit ~hundreds of MB of the card it is watching. DXGI is queried via
    toolkit's ctypes probe; usage comes from the nvidia-smi CLI.
  * DXGI CurrentUsage is PER-PROCESS, so from this process it is ~0 and
    useless -- what matters from a bystander is the BUDGET (the OS grant,
    which shrinks under system-wide pressure) for both segment groups:
    LOCAL (dedicated VRAM, the silent-paging cliff) and NON_LOCAL (shared,
    the pinned-memory budget both cliffs' overflow valves draw from).
  * nvidia-smi provides the actual dedicated usage: global used/total plus
    per-process breakdown, no CUDA context needed.

Each poll appends one JSON line to --out and prints a compact summary.
Stop with Ctrl+C.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit.memory_management import dxgi_meminfo  # noqa: E402

GIB = 1024 ** 3


def _dxgi_snapshot() -> dict:
    out: dict = {}
    adapter = dxgi_meminfo.selected_adapter_info()
    if adapter is not None:
        out["adapter"] = {
            "description": adapter.description,
            "match_method": adapter.match_method,
        }
    for group, query in (
        ("local", dxgi_meminfo.query_local_video_memory_info),
        ("non_local", dxgi_meminfo.query_non_local_video_memory_info),
    ):
        info = query(min_interval_s=0.0)
        if info is None:
            out[group] = None
            continue
        out[group] = {
            "budget_gib": round(info.budget_bytes / GIB, 3),
            # Per-process view of THIS watcher: ~0 by design; logged only so
            # a nonzero value flags that something in-process went wrong.
            "watcher_usage_gib": round(info.current_usage_bytes / GIB, 3),
            "available_for_reservation_gib": round(
                getattr(info, "available_for_reservation_bytes", 0) / GIB, 3
            ),
        }
    return out


def _nvidia_smi_snapshot() -> dict:
    try:
        total = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
        used_mib, total_mib = (int(v.strip()) for v in total.split(","))
        procs_raw = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        processes = []
        for line in procs_raw.splitlines():
            if not line.strip():
                continue
            pid, name, mem = (part.strip() for part in line.split(",", 2))
            try:
                # Per-process VRAM is "[N/A]" under WDDM (Windows does not
                # expose it); keep pid/name so the cohabitants are visible.
                used_gib = round(int(mem) / 1024, 3)
            except ValueError:
                used_gib = None
            try:
                pid_int = int(pid)
            except ValueError:
                continue
            processes.append({"pid": pid_int, "name": name, "used_gib": used_gib})
        return {
            "device_used_gib": round(used_mib / 1024, 3),
            "device_total_gib": round(total_mib / 1024, 3),
            "device_free_gib": round((total_mib - used_mib) / 1024, 3),
            "processes": processes,
        }
    except Exception as error:
        return {"error": f"nvidia-smi unavailable: {error}"}


def snapshot() -> dict:
    return {
        "ts": time.time(),
        "time": time.strftime("%H:%M:%S"),
        "dxgi": _dxgi_snapshot(),
        "smi": _nvidia_smi_snapshot(),
    }


def _summary_line(snap: dict) -> str:
    smi = snap["smi"]
    dxgi = snap["dxgi"]
    local = dxgi.get("local") or {}
    non_local = dxgi.get("non_local") or {}
    if "error" in smi:
        usage = smi["error"]
    else:
        usage = (
            f"used={smi['device_used_gib']:.2f}/{smi['device_total_gib']:.2f} GiB "
            f"free={smi['device_free_gib']:.2f} procs={len(smi['processes'])}"
        )
    return (
        f"[{snap['time']}] {usage} | dxgi_local_budget={local.get('budget_gib')} "
        f"dxgi_shared_budget={non_local.get('budget_gib')} GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--out", default=".codex/memwatch.jsonl")
    parser.add_argument("--once", action="store_true", help="single snapshot, exit")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[memwatch] polling every {args.interval:.1f}s -> {out_path}")
    while True:
        snap = snapshot()
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snap) + "\n")
        print(_summary_line(snap), flush=True)
        if args.once:
            return
        time.sleep(max(0.25, args.interval))


if __name__ == "__main__":
    main()
