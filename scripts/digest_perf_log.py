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
import statistics
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
        return "-"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return str(value)


def _dxgi_spill_reserve_gb(offload: dict):
    reserve = offload.get("dxgi_spill_reserve_gb")
    if reserve is not None:
        return reserve
    budget = offload.get("dxgi_non_local_budget_gb")
    usage = offload.get("dxgi_non_local_usage_gb")
    headroom = offload.get("dxgi_non_local_headroom_gb")
    try:
        return max(0.0, float(budget) - float(usage) - float(headroom))
    except (TypeError, ValueError):
        return None


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


def _first(pattern: str, text: str, cast=float, default=None):
    found = _search(pattern, text, cast)
    if not found:
        return default
    return found[0]


def _prefetch_fields(record: dict) -> dict:
    text = record.get("offload_prefetch", "") or ""
    return {
        "acquires": _first(r"acquires=(\d+)", text, int, 0),
        "skipped_gpu_resident": _first(r"skipped_gpu_resident=(\d+)", text, int, 0),
        "resync": _first(r"resync=(\d+)", text, int, 0),
        "mismatch": _first(r"mismatch=(\d+)", text, int, 0),
        "dup_block": _first(r"dup_block=(\d+)", text, int, 0),
        "hit": _first(r"hit=(\d+)", text, int, 0),
        "soft_miss": _first(r"soft_miss=(\d+)", text, int, 0),
        "hard_miss": _first(r"hard_miss=(\d+)", text, int, 0),
        "cpu_wait": _first(r"cpu_wait=([\d.]+)s", text, float, 0.0),
        "copy_s": _first(r"copy=([\d.]+)s@", text, float, 0.0),
        "copy_gbps": _first(r"copy=[\d.]+s@([\d.]+)GB/s", text, float, None),
        "fills": _first(r"fills=(\d+)", text, int, 0),
        "batches": _first(r"batches=(\d+)", text, int, 0),
        "schedule": _first(r"schedule=(\d+)", text, int, None),
        "confidence": _first(r"confidence=([^\s]+)", text, str, None),
    }

def _record_number(record: dict, key: str, default=0.0) -> float:
    value = record.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _active_resolutions(record: dict) -> str:
    buckets = record.get("normal_path_by_resolution", {}) or {}
    active = [s for s, b in buckets.items() if b.get("count")]
    return ",".join(sorted(active, key=lambda s: int(s))) if active else "?"


def _outlier_cutoff(values: list[float]) -> float:
    if not values:
        return 0.0
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    mad = statistics.median(deviations)
    if mad > 0:
        return median + 3.0 * 1.4826 * mad
    if len(values) > 1:
        mean = sum(values) / len(values)
        stddev = statistics.pstdev(values)
        if stddev > 0:
            return mean + 2.0 * stddev
    return max(values)


def _outlier_reason(record: dict, prefetch: dict) -> str:
    parts = []
    total = _record_number(record, "entire_training_step_s")
    forward = _record_number(record, "normal_training_forward_s")
    backward = _record_number(record, "backward_s")
    overhead = _record_number(record, "other_overhead_s")
    if total:
        components = [("fwd", forward), ("bwd", backward), ("overhead", overhead)]
        name, value = max(components, key=lambda item: item[1])
        if value > 0:
            parts.append(f"{name}={g(value)}s")
    if prefetch["hard_miss"] or prefetch["soft_miss"]:
        parts.append(f"misses={prefetch['hard_miss'] + prefetch['soft_miss']}")
    if prefetch["mismatch"] or prefetch["resync"]:
        parts.append(f"align={prefetch['resync']}/{prefetch['mismatch']}")
    if prefetch["cpu_wait"] > 0.05:
        parts.append(f"cpu_wait={g(prefetch['cpu_wait'], 1)}s")
    offload = record.get("smart_training_offload") or {}
    free_peak = offload.get("device_free_peak_gb")
    peak_source = offload.get("device_peak_source", "estimate")
    try:
        if free_peak is not None and float(free_peak) < 1.0:
            parts.append(f"free_peak={g(free_peak)}GiB/{peak_source}")
    except (TypeError, ValueError):
        pass
    return ", ".join(parts) if parts else "no obvious component spike"


def summarize_outliers(records: list[dict], limit: int = 5) -> list[str]:
    scored = []
    totals = []
    for record in records:
        total = record.get("entire_training_step_s")
        if not isinstance(total, (int, float)):
            continue
        total = float(total)
        totals.append(total)
        scored.append((total, record, _prefetch_fields(record)))
    if not scored:
        return []

    cutoff = _outlier_cutoff(totals)
    selected = [item for item in scored if item[0] >= cutoff]
    selected.sort(key=lambda item: item[0], reverse=True)
    if not selected:
        selected = sorted(scored, key=lambda item: item[0], reverse=True)[:min(3, len(scored))]
    else:
        selected = selected[:limit]

    median = statistics.median(totals)
    lines = [f"Outliers: median={g(median)}s cutoff={g(cutoff)}s showing={len(selected)}"]
    for total, record, prefetch in selected:
        lines.append(
            "  step {step}: total={total}s res={res} fwd={fwd}s bwd={bwd}s "
            "hit_rate={hr} hard={hard} soft={soft} reason={reason}".format(
                step=record.get("step"),
                total=g(total),
                res=_active_resolutions(record),
                fwd=g(record.get("normal_training_forward_s")),
                bwd=g(record.get("backward_s")),
                hr=(
                    f"{prefetch['hit'] / max(1, prefetch['hit'] + prefetch['soft_miss'] + prefetch['hard_miss']) * 100:.1f}%"
                    if (prefetch["hit"] or prefetch["soft_miss"] or prefetch["hard_miss"])
                    else "-"
                ),
                hard=prefetch["hard_miss"],
                soft=prefetch["soft_miss"],
                reason=_outlier_reason(record, prefetch),
            )
        )
    return lines

def summarize_records(records: list[dict]) -> list[str]:
    lines: list[str] = []
    totals = [r.get("entire_training_step_s") for r in records]
    totals = [float(v) for v in totals if isinstance(v, (int, float))]
    forwards = [r.get("normal_training_forward_s") for r in records]
    forwards = [float(v) for v in forwards if isinstance(v, (int, float))]
    backwards = [r.get("backward_s") for r in records]
    backwards = [float(v) for v in backwards if isinstance(v, (int, float))]

    if totals:
        lines.append(
            "Run summary: windows={n} total_avg={avg}s min={mn}s max={mx}s "
            "forward_avg={fwd}s backward_avg={bwd}s".format(
                n=len(records),
                avg=g(sum(totals) / len(totals)),
                mn=g(min(totals)),
                mx=g(max(totals)),
                fwd=g(sum(forwards) / len(forwards)) if forwards else "-",
                bwd=g(sum(backwards) / len(backwards)) if backwards else "-",
            )
        )

    prefetch_rows = [_prefetch_fields(r) for r in records if r.get("offload_prefetch")]
    if prefetch_rows:
        hit = sum(row["hit"] for row in prefetch_rows)
        soft = sum(row["soft_miss"] for row in prefetch_rows)
        hard = sum(row["hard_miss"] for row in prefetch_rows)
        accesses = hit + soft + hard
        waits = sum(row["cpu_wait"] for row in prefetch_rows)
        copy_s = sum(row["copy_s"] for row in prefetch_rows)
        fills = sum(row["fills"] for row in prefetch_rows)
        batches = sum(row["batches"] for row in prefetch_rows)
        copy_rates = [row["copy_gbps"] for row in prefetch_rows if row["copy_gbps"] is not None]
        schedules = [row["schedule"] for row in prefetch_rows if row["schedule"] is not None]
        confidence = {}
        for row in prefetch_rows:
            key = row["confidence"] or "unknown"
            confidence[key] = confidence.get(key, 0) + 1
        lines.append(
            "Prefetch summary: windows={n} hit_rate={hr:.1%} hits={hit} soft={soft} hard={hard} "
            "resync={resync} mismatch={mismatch} dup_block={dup} skipped_resident={skip}".format(
                n=len(prefetch_rows),
                hr=hit / max(1, accesses),
                hit=hit,
                soft=soft,
                hard=hard,
                resync=sum(row["resync"] for row in prefetch_rows),
                mismatch=sum(row["mismatch"] for row in prefetch_rows),
                dup=sum(row["dup_block"] for row in prefetch_rows),
                skip=sum(row["skipped_gpu_resident"] for row in prefetch_rows),
            )
        )
        lines.append(
            "Prefetch work: cpu_wait={wait}s copy={copy}s avg_copy={gbps}GB/s "
            "fills={fills} batches={batches} fills_per_batch={fpb} schedule_range={sr} confidence={conf}".format(
                wait=g(waits, 1),
                copy=g(copy_s, 1),
                gbps=g(sum(copy_rates) / len(copy_rates)) if copy_rates else "-",
                fills=fills,
                batches=batches,
                fpb=g(fills / batches, 2) if batches else "-",
                sr=(f"{min(schedules)}..{max(schedules)}" if schedules else "-"),
                conf=" ".join(f"{k}={v}" for k, v in sorted(confidence.items())),
            )
        )
    compile_rows = [r["compile"] for r in records if r.get("compile")]
    if compile_rows:
        new_frames = [int(row.get("new_frames") or 0) for row in compile_rows]
        # The first window that reports any tracing carries the cold compile;
        # a warm run should be flat 0 after it.
        warm = new_frames[1:]
        lines.append(
            "Compile: windows={n} new_frames total={t} first_window={c} "
            "after_first={w} (windows_recompiling={rw}) graphs={g} graph_breaks={gb}".format(
                n=len(compile_rows),
                t=sum(new_frames),
                c=new_frames[0],
                w=sum(warm),
                rw=sum(1 for v in warm if v > 0),
                g=compile_rows[-1].get("graphs_total"),
                gb=compile_rows[-1].get("graph_breaks_total"),
            )
        )
    gc_rows = [
        r.get("smart_training_offload") or {}
        for r in records
        if (r.get("smart_training_offload") or {}).get("alloc_retries_delta") is not None
    ]
    if gc_rows:
        retries = [int(row.get("alloc_retries_delta") or 0) for row in gc_rows]
        frees = [int(row.get("cuda_free_count_delta") or 0) for row in gc_rows]
        mallocs = [int(row.get("cuda_malloc_count_delta") or 0) for row in gc_rows]
        reclaimables = [
            max(0.0, float(row["peak_reserved_gb"]) - float(row["peak_allocated_gb"]))
            for row in gc_rows
            if row.get("peak_reserved_gb") is not None
            and row.get("peak_allocated_gb") is not None
        ]
        lines.append(
            "Allocator GC: windows={n} retries={r} (windows_with_retries={rw}) "
            "cudaFree={f} cudaMalloc={m} reclaimable_at_peak avg={ra} max={rx} GiB".format(
                n=len(gc_rows),
                r=sum(retries),
                rw=sum(1 for v in retries if v > 0),
                f=sum(frees),
                m=sum(mallocs),
                ra=g(sum(reclaimables) / len(reclaimables)) if reclaimables else "-",
                rx=g(max(reclaimables)) if reclaimables else "-",
            )
        )
    dxgi_rows = [
        r.get("smart_training_offload") or {}
        for r in records
        if (r.get("smart_training_offload") or {}).get("dxgi_non_local_budget_gb") is not None
    ]
    if dxgi_rows:
        headrooms = [
            float(row["dxgi_non_local_headroom_gb"])
            for row in dxgi_rows
            if row.get("dxgi_non_local_headroom_gb") is not None
        ]
        usages = [
            float(row["dxgi_non_local_usage_gb"])
            for row in dxgi_rows
            if row.get("dxgi_non_local_usage_gb") is not None
        ]
        reserves = [_dxgi_spill_reserve_gb(row) for row in dxgi_rows]
        reserves = [float(value) for value in reserves if value is not None]
        last = dxgi_rows[-1]
        lines.append(
            "DXGI shared: windows={n} budget={budget} usage_last={usage} usage_max={usage_max} "
            "headroom_last={headroom} headroom_min={headroom_min} reserve={reserve} safe={safe} match={match}".format(
                n=len(dxgi_rows),
                budget=g(last.get("dxgi_non_local_budget_gb")),
                usage=g(last.get("dxgi_non_local_usage_gb")),
                usage_max=g(max(usages)) if usages else "-",
                headroom=g(last.get("dxgi_non_local_headroom_gb")),
                headroom_min=g(min(headrooms)) if headrooms else "-",
                reserve=g(reserves[-1]) if reserves else "-",
                safe=last.get("dxgi_safe_for_control"),
                match=last.get("dxgi_match_method"),
            )
        )
        local_rows = [row for row in dxgi_rows if row.get("dxgi_local_budget_gb") is not None]
        if local_rows:
            local_headrooms = [
                float(row["dxgi_local_headroom_gb"])
                for row in local_rows
                if row.get("dxgi_local_headroom_gb") is not None
            ]
            local_usages = [
                float(row["dxgi_local_usage_gb"])
                for row in local_rows
                if row.get("dxgi_local_usage_gb") is not None
            ]
            local_last = local_rows[-1]
            lines.append(
                "DXGI local: budget={budget} usage_last={usage} usage_max={usage_max} "
                "headroom_last={headroom} headroom_min={headroom_min}".format(
                    budget=g(local_last.get("dxgi_local_budget_gb")),
                    usage=g(local_last.get("dxgi_local_usage_gb")),
                    usage_max=g(max(local_usages)) if local_usages else "-",
                    headroom=g(local_last.get("dxgi_local_headroom_gb")),
                    headroom_min=g(min(local_headrooms)) if local_headrooms else "-",
                )
            )

    lines.extend(summarize_outliers(records))
    return lines


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

    compile_counters = record.get("compile")
    if compile_counters:
        lines.append(
            "  compile: new_frames={nf} new_graphs={ng} new_breaks={nb}  "
            "totals frames={tf} graphs={tg} breaks={tb}".format(
                nf=compile_counters.get("new_frames"),
                ng=compile_counters.get("new_graphs"),
                nb=compile_counters.get("new_graph_breaks"),
                tf=compile_counters.get("frames_total"),
                tg=compile_counters.get("graphs_total"),
                tb=compile_counters.get("graph_breaks_total"),
            )
        )

    if record.get("dop_cache_hits") or record.get("dop_cache_misses"):
        rate = record.get("dop_cache_hit_rate")
        lines.append(
            "  dop: hits={h} misses={m} hit_rate={r}  prior_gen={pg}s ({pm}/miss)".format(
                h=record.get("dop_cache_hits"),
                m=record.get("dop_cache_misses"),
                r="-" if rate is None else f"{rate * 100:.1f}%",
                pg=g(record.get("dop_prior_generation_s")),
                pm=g(record.get("dop_prior_generation_per_miss_s")),
            )
        )

    offload = record.get("smart_training_offload")
    if offload and "diagnostic_error" not in offload:
        # Peak within-step driver footprint / free. New logs carry observed
        # driver-free minima; older logs only have an allocator-derived estimate.
        used_peak = offload.get("device_used_peak_gb")
        free_peak = offload.get("device_free_peak_gb")
        peak_source = offload.get("device_peak_source")
        if used_peak is None and offload.get("peak_reserved_gb") is not None:
            other = max(
                0.0,
                (offload.get("device_used_gb") or 0.0)
                - (offload.get("torch_reserved_gb") or 0.0),
            )
            used_peak = offload["peak_reserved_gb"] + other
            total = offload.get("device_total_gb")
            if total is not None:
                free_peak = max(0.0, total - used_peak)
            peak_source = "estimate"
        if peak_source is None:
            peak_source = "observed" if offload.get("driver_free_samples") else "estimate"
        lines.append(
            "  offload: managed={ml} resident={res} offloaded={off} "
            "ring_peak={rpk} ring_live={lr}/{pr} working_peak={hp} reserve_space={ht} residual={hu} alloc={al} reserved={rv} "
            "peak_reserved={prsv} cached={cg} driver_peak={dup}/{dt} free_peak={dfp} source={src} native_fp8_layers={fp8}".format(
                ml=offload.get("managed_layers"),
                res=g(offload.get("planned_resident_gb")),
                off=g(offload.get("offloaded_cpu_gb")),
                rpk=g(offload.get("ring_peak_gb", offload.get("live_ring_gb"))),
                lr=g(offload.get("live_ring_gb")),
                pr=g(offload.get("planned_ring_gb")),
                # Peak within-step device footprint / free (what the spill cliff
                # sees), derived above so old logs are correct too.
                dup=g(used_peak),
                dt=g(offload.get("device_total_gb")),
                dfp=g(free_peak),
                src=peak_source,
                # Prefer the truthful within-step peak; fall back to the old
                # trough field for logs written before this metric existed.
                hp=g(offload.get("working_reserve_peak_gb",
                                 offload.get("working_reserve_used_gb",
                                             offload.get("working_headroom_used_gb")))),
                hu=g(offload.get("working_reserve_residual_gb",
                                 offload.get("working_reserve_used_gb",
                                             offload.get("working_headroom_used_gb")))),
                ht=g(offload.get("training_working_reserve_gb",
                                 offload.get("training_headroom_gb"))),
                al=g(offload.get("torch_allocated_gb")),
                rv=g(offload.get("torch_reserved_gb")),
                prsv=g(offload.get("peak_reserved_gb")),
                cg=g(offload.get("allocator_cached_gb")),
                fp8=offload.get("fp8_training_forward_layers"),
            )
        )
        if offload.get("alloc_retries_delta") is not None:
            reclaimable = None
            if (
                offload.get("peak_reserved_gb") is not None
                and offload.get("peak_allocated_gb") is not None
            ):
                reclaimable = max(
                    0.0,
                    float(offload["peak_reserved_gb"])
                    - float(offload["peak_allocated_gb"]),
                )
            lines.append(
                "  alloc_gc: retries=+{r} cudaMalloc=+{m} cudaFree=+{f} "
                "reclaimable_at_peak={rec} GiB".format(
                    r=offload.get("alloc_retries_delta"),
                    m=offload.get("cuda_malloc_count_delta"),
                    f=offload.get("cuda_free_count_delta"),
                    rec=g(reclaimable),
                )
            )
        if offload.get("bounce_fill_batches") is not None:
            # Worker-side request count (the "small requests by the workers").
            # fill_batches = worker lock-cycles; group>1 means block-batched.
            lines.append(
                "  worker_fills: fills={f} batches={b} group={grp}({fpb}/batch) "
                "copy={cs}s@{cg}GB/s".format(
                    f=offload.get("bounce_fills"),
                    b=offload.get("bounce_fill_batches"),
                    grp=offload.get("bounce_fill_group_size"),
                    fpb=g(offload.get("bounce_fills_per_batch"), 1),
                    cs=g(offload.get("bounce_copy_s"), 1),
                    cg=g(offload.get("bounce_copy_gbps")),
                )
            )
        if offload.get("dxgi_non_local_budget_gb") is not None:
            lines.append(
                "  dxgi: local usage={local_usage}/{local_budget} GiB headroom={local_headroom} GiB; "
                "shared usage={shared_usage}/{shared_budget} GiB headroom={shared_headroom} GiB "
                "reserve={reserve} GiB safe={safe} manual={manual} match={match}".format(
                    local_usage=g(offload.get("dxgi_local_usage_gb")),
                    local_budget=g(offload.get("dxgi_local_budget_gb")),
                    local_headroom=g(offload.get("dxgi_local_headroom_gb")),
                    shared_usage=g(offload.get("dxgi_non_local_usage_gb")),
                    shared_budget=g(offload.get("dxgi_non_local_budget_gb")),
                    shared_headroom=g(offload.get("dxgi_non_local_headroom_gb")),
                    reserve=g(_dxgi_spill_reserve_gb(offload)),
                    safe=offload.get("dxgi_safe_for_control"),
                    manual=offload.get("dxgi_manual_control"),
                    match=offload.get("dxgi_match_method"),
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
                hr="-" if rate is None else f"{rate[0]:.1f}%",
                cs="-" if copy is None else g(copy[0]),
                cg="-" if copy is None else g(copy[1]),
                cw="-" if wait is None else g(wait[0], 1),
                inf="-" if inflight is None else g(inflight[0]),
            )
        )

    ingraph = record.get("ingraph_stream", "")
    ingraph_fetches = _search(r"fetches=(\d+)", ingraph, int)
    ingraph_h2d = _search(r"h2d_ms=([\d.]+)", ingraph, float)
    ingraph_wait = _search(r"wait_ms=([\d.]+)", ingraph, float)
    if ingraph_fetches:
        lines.append(
            "  ingraph: fetches={f} h2d_ms={h} wait_ms={w}".format(
                f=ingraph_fetches[0],
                h="-" if ingraph_h2d is None else g(ingraph_h2d[0], 3),
                w="-" if ingraph_wait is None else g(ingraph_wait[0], 3),
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
            "bwd_ring_reuse={rr} (miss {rm})  native_fp8 calls={fc} fallbacks={ff}".format(
                ft="-" if fetches is None else fetches[0],
                f="-" if fwd is None else fwd[0],
                b="-" if bwd is None else bwd[0],
                cw="-" if cwait is None else g(cwait[0], 1),
                rr="-" if reuse is None else f"{reuse[2]:.1f}%",
                rm="-" if reuse is None else int(reuse[1]),
                fc="-" if fp8 is None else fp8[1],
                ff="-" if fp8 is None else fp8[2],
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
    for line in summarize_records(records):
        print(line)
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
