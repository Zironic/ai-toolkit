#!/usr/bin/env python3
"""Aggregate a per_file_loss.jsonl log produced by training (train.log_per_file_loss).

Each line: {step, path, name, orig_w, orig_h, train_w, train_h, timestep, is_reg, loss}

Raw diffusion loss depends heavily on timestep/sigma, so per-file absolute loss
mostly answers "how hard was this denoising target under these conditions," not
"is the LoRA getting better."  This tool normalises for that:

  residual = file_loss - global_median_for_same_timestep_bucket

and adds trend tracking (early half vs recent half of training) so you can
distinguish "high but falling" (still learning) from "high and flat" (conflict).

Usage:
    python tools/analyze_per_file_loss.py path/to/per_file_loss.jsonl
    python tools/analyze_per_file_loss.py log.jsonl --min-seen 10 --split 0.6
    python tools/analyze_per_file_loss.py log.jsonl --csv out.csv
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict


# ── stats ──────────────────────────────────────────────────────────────────────

def percentile(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def median(xs):
    return percentile(xs, 50)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def mad(xs):
    if not xs:
        return float("nan")
    m = median(xs)
    return median([abs(x - m) for x in xs])


# ── loading ────────────────────────────────────────────────────────────────────

def dataset_of(r):
    """Extract dataset folder name from path (the directory containing the file)."""
    path = r.get("path", "")
    parts = path.replace("\\", "/").rstrip("/").split("/")
    # look for a "datasets" segment and return what follows it
    try:
        di = next(i for i, p in enumerate(parts) if p == "datasets")
        if di + 2 <= len(parts) - 1:
            return parts[di + 1]
    except StopIteration:
        pass
    # fallback: parent directory of the file
    return parts[-2] if len(parts) >= 2 else "unknown"


def res_label(r):
    return f"{r.get('train_w', '?')}x{r.get('train_h', '?')}"


def load(path, min_step, max_step, include_reg):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print(f"  (skipping malformed line {ln})")
                continue
            if min_step is not None and r.get("step", 0) < min_step:
                continue
            if max_step is not None and r.get("step", 0) > max_step:
                continue
            if not include_reg and r.get("is_reg"):
                continue
            rows.append(r)
    return rows


# ── timestep normalisation ─────────────────────────────────────────────────────

def build_ts_baseline(rows, n_bins):
    """Compute median loss per equal-width timestep bin across all rows."""
    all_ts = [r.get("timestep", 500) for r in rows]
    ts_min, ts_max = min(all_ts), max(all_ts)
    width = (ts_max - ts_min) / n_bins if ts_max > ts_min else 1.0
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        idx = min(int((r.get("timestep", 500) - ts_min) / width), n_bins - 1)
        bins[idx].append(r["loss"])
    edges = [ts_min + i * width for i in range(n_bins + 1)]
    medians = [median(b) if b else float("nan") for b in bins]
    return edges, medians


def ts_residual(r, edges, bin_medians):
    ts = r.get("timestep", 500)
    n = len(bin_medians)
    ts_min = edges[0]
    width = (edges[-1] - edges[0]) / n if n > 0 and edges[-1] > edges[0] else 1.0
    idx = min(int((ts - ts_min) / width), n - 1)
    baseline = bin_medians[idx]
    return r["loss"] - baseline if not math.isnan(baseline) else r["loss"]


# ── trend ──────────────────────────────────────────────────────────────────────

def trend_label(recent_p50, prev_p50, overall_p50, flat_thresh=0.015):
    if math.isnan(recent_p50) or math.isnan(prev_p50):
        return "n/a"
    is_high = recent_p50 > overall_p50
    delta = recent_p50 - prev_p50
    if abs(delta) < flat_thresh:
        direction = "flat"
    elif delta < 0:
        direction = "falling"
    else:
        direction = "rising"
    return ("high" if is_high else "low") + "+" + direction


# ── formatting ─────────────────────────────────────────────────────────────────

def fmt(v, w=8, d=4):
    return " " * (w - 1) + "?" if math.isnan(v) else f"{v:{w}.{d}f}"


def fmt_signed(v, w=8, d=4):
    return " " * (w - 1) + "?" if math.isnan(v) else f"{v:+{w}.{d}f}"


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile")
    ap.add_argument("--top", type=int, default=20, help="worst/best files to show (default 20)")
    ap.add_argument("--min-step", type=int, default=None)
    ap.add_argument("--max-step", type=int, default=None)
    ap.add_argument("--min-seen", type=int, default=8,
                    help="min observations to rank a file (default 8)")
    ap.add_argument("--ts-bins", type=int, default=8,
                    help="timestep buckets for residual normalisation (default 8)")
    ap.add_argument("--split", type=float, default=0.5,
                    help="fraction of step range treated as 'early' for trend (default 0.5)")
    ap.add_argument("--min-bucket-samples", type=int, default=20,
                    help="min samples to show a resolution bucket (default 20)")
    ap.add_argument("--include-reg", action="store_true")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.logfile):
        raise SystemExit(f"No such file: {args.logfile}")

    rows = load(args.logfile, args.min_step, args.max_step, args.include_reg)
    if not rows:
        raise SystemExit("No rows after filtering.")

    all_steps = [r.get("step", 0) for r in rows]
    step_min, step_max = min(all_steps), max(all_steps)
    split_step = step_min + (step_max - step_min) * args.split

    all_losses = [r["loss"] for r in rows]
    overall_p50 = median(all_losses)
    overall_p90 = percentile(all_losses, 90)

    print(f"Loaded {len(rows)} sample-steps | steps {step_min}..{step_max}")
    print(f"Overall  p50={overall_p50:.5f}  p90={overall_p90:.5f}  mean={mean(all_losses):.5f}")

    # ── timestep baseline ──────────────────────────────────────────────────────
    ts_edges, ts_bin_medians = build_ts_baseline(rows, args.ts_bins)
    for r in rows:
        r["_resid"] = ts_residual(r, ts_edges, ts_bin_medians)

    # ── by dataset ─────────────────────────────────────────────────────────────
    ds_early = defaultdict(list)
    ds_recent = defaultdict(list)
    for r in rows:
        ds = dataset_of(r)
        target = ds_early if r.get("step", 0) <= split_step else ds_recent
        target[ds].append(r["loss"])

    all_ds = sorted(set(list(ds_early) + list(ds_recent)))
    print(f"\nLoss by dataset  (split step ~{split_step:.0f})")
    print(f"  {'dataset':<32}  {'seen':>5}  {'p50':>8}  {'p90':>8}  {'prev_p50':>9}  trend")
    print("  " + "-" * 74)
    for ds in all_ds:
        early = ds_early.get(ds, [])
        recent = ds_recent.get(ds, [])
        p50_r = median(recent) if recent else float("nan")
        p50_e = median(early) if early else float("nan")
        p90_r = percentile(recent, 90) if recent else float("nan")
        tl = trend_label(p50_r, p50_e, overall_p50) if early and recent else "n/a"
        print(f"  {ds:<32}  {len(early)+len(recent):>5}  {fmt(p50_r)}  {fmt(p90_r)}  {fmt(p50_e)}  {tl}")

    # ── by resolution ──────────────────────────────────────────────────────────
    by_res = defaultdict(list)
    for r in rows:
        by_res[res_label(r)].append(r["loss"])
    res_filtered = sorted(
        [(k, v) for k, v in by_res.items() if len(v) >= args.min_bucket_samples],
        key=lambda kv: -median(kv[1]),
    )
    print(f"\nLoss by resolution  (>= {args.min_bucket_samples} samples, {len(res_filtered)} buckets shown)")
    print(f"  {'resolution':<14}  {'samples':>7}  {'p50':>8}  {'p90':>8}")
    print("  " + "-" * 46)
    for res, losses in res_filtered:
        print(f"  {res:<14}  {len(losses):>7}  {fmt(median(losses))}  {fmt(percentile(losses, 90))}")

    # ── by file ────────────────────────────────────────────────────────────────
    file_rows = defaultdict(list)
    for r in rows:
        file_rows[r.get("name") or r.get("path")].append(r)

    file_stats = {}
    for name, frows in file_rows.items():
        if len(frows) < args.min_seen:
            continue
        early = [r for r in frows if r.get("step", 0) <= split_step]
        recent = [r for r in frows if r.get("step", 0) > split_step]
        losses = [r["loss"] for r in frows]
        p50_e = median([r["loss"] for r in early]) if early else float("nan")
        p50_r = median([r["loss"] for r in recent]) if recent else float("nan")
        resid = mean([r["_resid"] for r in frows])
        file_stats[name] = {
            "seen": len(frows),
            "p50": median(losses),
            "p90": percentile(losses, 90),
            "p50_early": p50_e,
            "p50_recent": p50_r,
            "resid": resid,
            "trend": trend_label(p50_r, p50_e, overall_p50) if early and recent else "n/a",
        }

    ranked = sorted(file_stats.items(), key=lambda kv: -kv[1]["resid"])
    n = min(args.top, len(ranked))

    col = f"  {'file':<42}  {'seen':>4}  {'p50':>8}  {'p90':>8}  {'resid':>9}  trend"
    sep = "  " + "-" * 84

    print(f"\nFiles ranked by ts-normalised residual  (min-seen={args.min_seen}, {len(ranked)} files)")
    print(f"\n  Worst {n} (highest residual, hardest relative to their timestep):")
    print(col); print(sep)
    for name, s in ranked[:n]:
        print(f"  {name[:42]:<42}  {s['seen']:>4}  {fmt(s['p50'])}  {fmt(s['p90'])}  {fmt_signed(s['resid'])}  {s['trend']}")

    print(f"\n  Best {n} (lowest residual, easiest relative to their timestep):")
    print(col); print(sep)
    for name, s in list(reversed(ranked[-n:])):
        print(f"  {name[:42]:<42}  {s['seen']:>4}  {fmt(s['p50'])}  {fmt(s['p90'])}  {fmt_signed(s['resid'])}  {s['trend']}")

    # ── flags ──────────────────────────────────────────────────────────────────
    flags = []
    for name, s in ranked:
        if not math.isnan(s["resid"]) and s["resid"] > 0.04 and "flat" in s["trend"]:
            flags.append(f"  STUCK   [resid={s['resid']:+.4f}, {s['trend']}]  {name}")
        if s["p50"] < overall_p50 and "rising" in s["trend"]:
            flags.append(f"  RISING  [recent_p50={s['p50_recent']:.4f}, {s['trend']}]  {name}")
        if not math.isnan(s["p90"]) and s["p90"] > 2.5 * overall_p90 and s["seen"] >= args.min_seen:
            flags.append(f"  SPIKE   [p90={s['p90']:.4f}]  {name}")

    print(f"\nFlags ({len(flags)})")
    if flags:
        print("  " + "-" * 60)
        for flag in flags:
            print(flag)
    else:
        print("  none")

    # ── csv ────────────────────────────────────────────────────────────────────
    if args.csv:
        res_of, ds_of = {}, {}
        for r in rows:
            k = r.get("name") or r.get("path")
            res_of.setdefault(k, res_label(r))
            ds_of.setdefault(k, dataset_of(r))
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "dataset", "train_res", "seen",
                        "p50", "p90", "p50_early", "p50_recent", "resid", "trend"])
            for name, s in ranked:
                w.writerow([name, ds_of.get(name, "?"), res_of.get(name, "?"), s["seen"],
                            f"{s['p50']:.6f}", f"{s['p90']:.6f}",
                            f"{s['p50_early']:.6f}", f"{s['p50_recent']:.6f}",
                            f"{s['resid']:.6f}", s["trend"]])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
