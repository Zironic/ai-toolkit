#!/usr/bin/env python3
"""Digest a training performance_log.jsonl into a compact, readable summary.

Each line of performance_log.jsonl is one reconciled timing window. Reading the
raw file is verbose; this collapses every window to a few lines covering the
step breakdown, the active resolution buckets, the smart-offload layout, and the
key prefetch/profile signals (bounce-pool hit rate, compute-wait, backward ring
reuse) that are otherwise buried in the embedded text reports.

Examples:
    python scripts/digest_perf_log.py                  # newest run under output/
    python scripts/digest_perf_log.py my_job_name      # output/my_job_name/...
    python scripts/digest_perf_log.py path/to/performance_log.jsonl --last 5
    python scripts/digest_perf_log.py --all --full     # every window + raw reports
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "output"
PERF_NAME = "performance_log.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "path",
        nargs="?",
        help=(
            "performance_log.jsonl, a job folder, a bare job name, or omitted "
            "to use the most recently updated run under output/."
        ),
    )
    parser.add_argument("--last", type=int, default=1,
                        help="Show only the last N windows (default 1). Use --all for every window.")
    parser.add_argument("--all", action="store_true", help="Show every window.")
    parser.add_argument("--full", action="store_true",
                        help="Also print the raw embedded offload reports for each shown window.")
    parser.add_argument("--archived", action="store_true",
                        help="Also include logs/*_performance_log.jsonl archives of older runs.")
    return parser.parse_args()


def resolve_path(supplied: str | None) -> Path:
    if supplied:
        candidate = Path(supplied)
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            inner = candidate / PERF_NAME
            if inner.is_file():
                return inner
            raise SystemExit(f"No {PERF_NAME} in folder: {candidate}")
        # Treat as a bare job name under output/.
        named = DEFAULT_OUTPUT / supplied / PERF_NAME
        if named.is_file():
            return named
        raise SystemExit(f"Could not resolve performance log from: {supplied}")

    if not DEFAULT_OUTPUT.is_dir():
        raise SystemExit(f"No path given and {DEFAULT_OUTPUT} does not exist.")
    candidates = list(DEFAULT_OUTPUT.glob(f"*/{PERF_NAME}"))
    if not candidates:
        raise SystemExit(f"No {PERF_NAME} found under {DEFAULT_OUTPUT}.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                print(f"  (skipped malformed line {line_no}: {error})")
    return records


def g(value, nd: int = 2) -> str:
    """Format an optional number, or an em dash when missing."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def _search(pattern: str, text: str, cast=float):
    if not text:
        return None
    match = re.search(pattern, text)
    if not match:
        return None
    try:
        return tuple(cast(group) for group in match.groups())
    except (TypeError, ValueError):
        return match.groups()


def summarize_record(record: dict, full: bool) -> list[str]:
    lines: list[str] = []
    step = record.get("step")
    window = record.get("window_steps")
    total = record.get("entire_training_step_s")
    lines.append(f"step {step}  (window={window} steps)  total={g(total)}s")

    lines.append(
        "  forward={f}s  backward={b}s  optimizer={o}s  data={d}s  prep={p}s  overhead={x}s".format(
            f=g(record.get("normal_training_forward_s")),
            b=g(record.get("backward_s")),
            o=g(record.get("optimizer_step_s")),
            d=g(record.get("data_loading_s")),
            p=g(record.get("batch_preparation_s")),
            x=g(record.get("other_overhead_s")),
        )
    )

    buckets = record.get("normal_path_by_resolution", {}) or {}
    for size in sorted(buckets, key=lambda s: int(s)):
        bucket = buckets[size]
        if not bucket.get("count"):
            continue
        lines.append(
            "  res {sz}: count={c} path={path}s (fwd {f} / bwd {b})  "
            "peak_alloc={pa}  peak_reserved={pr} GiB".format(
                sz=size,
                c=bucket.get("count"),
                path=g(bucket.get("normal_path_s")),
                f=g(bucket.get("normal_forward_s")),
                b=g(bucket.get("normal_backward_s")),
                pa=g(bucket.get("peak_allocated_gb_max")),
                pr=g(bucket.get("peak_reserved_gb_max")),
            )
        )

    if record.get("dop_cache_hits") or record.get("dop_cache_misses"):
        rate = record.get("dop_cache_hit_rate")
        lines.append(
            "  dop: hits={h} misses={m} hit_rate={r}  prior_gen={pg}s ({pm}/miss)".format(
                h=record.get("dop_cache_hits"),
                m=record.get("dop_cache_misses"),
                r="—" if rate is None else f"{rate * 100:.1f}%",
                pg=g(record.get("dop_prior_generation_s")),
                pm=g(record.get("dop_prior_generation_per_miss_s")),
            )
        )

    offload = record.get("smart_training_offload")
    if offload and "diagnostic_error" not in offload:
        lines.append(
            "  offload: managed={ml} resident={res} offloaded={off} "
            "ring={lr}/{pr} headroom={hu}/{ht} alloc={al} reserved={rv} "
            "peak_reserved={prsv} cached={cg} fp8_fwd={fp8}".format(
                ml=offload.get("managed_layers"),
                res=g(offload.get("planned_resident_gb")),
                off=g(offload.get("offloaded_cpu_gb")),
                lr=g(offload.get("live_ring_gb")),
                pr=g(offload.get("planned_ring_gb")),
                hu=g(offload.get("working_headroom_used_gb")),
                ht=g(offload.get("training_headroom_gb")),
                al=g(offload.get("torch_allocated_gb")),
                rv=g(offload.get("torch_reserved_gb")),
                prsv=g(offload.get("peak_reserved_gb")),
                cg=g(offload.get("allocator_cached_gb")),
                fp8=offload.get("fp8_training_forward_layers"),
            )
        )
    elif offload:
        lines.append(f"  offload: diagnostic_error={offload.get('diagnostic_error')}")

    prefetch = record.get("offload_prefetch", "")
    rate = _search(r"hit_rate=([\d.]+)%", prefetch)
    copy = _search(r"copy=([\d.]+)s@([\d.]+)GB/s", prefetch)
    wait = _search(r"cpu_wait=([\d.]+)s", prefetch)
    inflight = _search(r"inflight=([\d.]+)GiB", prefetch)
    if rate or copy:
        lines.append(
            "  bouncepool: hit_rate={hr}  copy={cs}s@{cg}GB/s  cpu_wait={cw}s  inflight={inf}GiB".format(
                hr="—" if rate is None else f"{rate[0]:.1f}%",
                cs="—" if copy is None else g(copy[0]),
                cg="—" if copy is None else g(copy[1]),
                cw="—" if wait is None else g(wait[0], 1),
                inf="—" if inflight is None else g(inflight[0]),
            )
        )

    profile = record.get("offload_profile", "")
    fetches = _search(r"fetches=(\d+)", profile, int)
    fwd = _search(r"forward=(\d+)", profile, int)
    bwd = _search(r"backward=(\d+)", profile, int)
    cwait = _search(r"compute wait on ready: ([\d.]+)s", profile)
    reuse = _search(r"backward GPU-ring reuse: hits=(\d+) misses=(\d+) hit_rate=([\d.]+)%",
                    profile, lambda s: float(s))
    fp8 = _search(r"native FP8 linear: enabled=(\w+) calls=(\d+) fallbacks=(\d+)",
                  profile, str)
    if fetches or cwait or reuse:
        lines.append(
            "  profile: fetches={ft} (fwd {f} / bwd {b})  compute_wait={cw}s  "
            "bwd_ring_reuse={rr} (miss {rm})  fp8 calls={fc} fallbacks={ff}".format(
                ft="—" if fetches is None else fetches[0],
                f="—" if fwd is None else fwd[0],
                b="—" if bwd is None else bwd[0],
                cw="—" if cwait is None else g(cwait[0], 1),
                rr="—" if reuse is None else f"{reuse[2]:.1f}%",
                rm="—" if reuse is None else int(reuse[1]),
                fc="—" if fp8 is None else fp8[1],
                ff="—" if fp8 is None else fp8[2],
            )
        )

    if full:
        for key in ("offload_profile", "offload_prefetch"):
            text = record.get(key)
            if text:
                lines.append(f"  --- {key} ---")
                lines.extend("  " + ln for ln in text.splitlines())

    return lines


def print_trend(records: list[dict]) -> None:
    if len(records) < 2:
        return
    print("Trend (step -> total step time):")
    for record in records:
        total = record.get("entire_training_step_s")
        # Pick the dominant active resolution for context.
        buckets = record.get("normal_path_by_resolution", {}) or {}
        active = [s for s, b in buckets.items() if b.get("count")]
        res = ",".join(sorted(active, key=lambda s: int(s))) if active else "?"
        print(f"  step {record.get('step'):>7}  {g(total):>7}s   res={res}")
    print()


def main() -> int:
    args = parse_args()
    path = resolve_path(args.path)

    sources = [path]
    if args.archived:
        archives = sorted((path.parent / "logs").glob(f"*_{PERF_NAME}"))
        sources = archives + sources

    records: list[dict] = []
    for source in sources:
        records.extend(load_records(source))

    if not records:
        print(f"No timing windows in {path}")
        return 0

    print(f"Perf log: {path}")
    print(f"Windows: {len(records)} (steps {records[0].get('step')}..{records[-1].get('step')})")
    print()

    shown = records if args.all else records[-max(1, args.last):]
    print_trend(shown)

    for record in shown:
        for line in summarize_record(record, args.full):
            print(line)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
