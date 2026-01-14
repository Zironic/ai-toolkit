#!/usr/bin/env python3
"""Parse training logs to compute average `normalized_loss` per PNG file.

Behavior:
- Only considers log lines that contain the trailing marker '][CONTROL]' (signifies the last progress bar for the step).
- Extracts the filename ending with `.png` that immediately precedes the '][CONTROL]' token.
- Extracts the numeric value for the `normalized_loss:` field (scientific or decimal notation supported).
- Aggregates values per unique PNG basename and prints averages (optionally CSV).

Usage:
    python scripts/parse_normalized_loss.py /path/to/log.txt
    python scripts/parse_normalized_loss.py "logs/*.log" --out results.csv --min-count 3

"""
from __future__ import annotations
import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

# Regex to capture normalized_loss value (scientific notation supported)
RE_VALUE = re.compile(r"normalized_loss:\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)")
# Regex to capture a filename ending in .png/.jpg/.jpeg immediately before the ][CONTROL] marker
RE_IMG_CONTROL = re.compile(r"([^\s\]]+\.(?:png|jpg|jpeg))\]\s*\[CONTROL\]", re.IGNORECASE)


def parse_paths(paths: List[str]) -> List[str]:
    files: List[str] = []
    for p in paths:
        if p == "-":
            files.append("-")
            continue
        matches = glob.glob(p)
        if matches:
            files.extend(matches)
        elif os.path.exists(p):
            files.append(p)
        else:
            # accept patterns that match nothing but warn
            print(f"[warn] pattern {p} matched no files", file=sys.stderr)
    return files


def parse_file(path: str, key_re: re.Pattern = RE_VALUE) -> List[Tuple[str, float]]:
    results: List[Tuple[str, float]] = []
    if path == "-":
        lines = sys.stdin
    else:
        try:
            f = open(path, "r", encoding="utf-8", errors="replace")
            lines = f
        except Exception as e:
            print(f"[error] could not open {path}: {e}", file=sys.stderr)
            return results

    with (open(path, "r", encoding="utf-8", errors="replace") if path != "-" else sys.stdin) as fh:
        for ln in fh:
            # Only consider lines that mark the end of the progress bar for the step
            if "][CONTROL]" not in ln:
                continue
            m_img = RE_IMG_CONTROL.search(ln)
            if not m_img:
                continue
            raw_name = m_img.group(1)
            name = os.path.basename(raw_name)
            m_val = key_re.search(ln)
            if not m_val:
                continue
            try:
                val = float(m_val.group(1))
            except Exception:
                continue
            results.append((name, val))
    return results


def aggregate(entries: List[Tuple[str, float]]) -> Dict[str, Tuple[int, float]]:
    # returns map: name -> (count, mean)
    acc: Dict[str, List[float]] = defaultdict(list)
    for name, val in entries:
        acc[name].append(val)
    out: Dict[str, Tuple[int, float]] = {}
    for k, v in acc.items():
        out[k] = (len(v), sum(v) / len(v))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Parse logs and average normalized_loss per PNG file")
    p.add_argument("logs", nargs="+", help="Log file paths or glob patterns; use - for stdin")
    p.add_argument("--key", default="normalized_loss", help="Field to extract (default: normalized_loss)")
    p.add_argument("--out", help="Write CSV output to this file (name, count, avg)")
    p.add_argument("--min-count", type=int, default=1, help="Minimum occurrences for inclusion")
    p.add_argument("--precision", type=int, default=6, help="Decimal digits for printed averages")
    args = p.parse_args(argv)

    # Build a regex for key if user passes something else
    key_pattern = re.escape(args.key) + r":\s*([+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)"
    key_re = re.compile(key_pattern)

    files = parse_paths(args.logs)
    all_entries: List[Tuple[str, float]] = []
    for f in files:
        entries = parse_file(f, key_re=key_re)
        all_entries.extend(entries)

    agg = aggregate(all_entries)

    # Filter by count
    filtered = {k: v for k, v in agg.items() if v[0] >= args.min_count}

    # Sort by average descending
    rows = sorted(filtered.items(), key=lambda kv: kv[1][1], reverse=True)

    if args.out:
        try:
            with open(args.out, "w", encoding="utf-8", newline="") as csvf:
                import csv
                w = csv.writer(csvf)
                w.writerow(["name", "count", f"avg_{args.key}"])
                for name, (count, avg) in rows:
                    w.writerow([name, count, f"{avg:.{args.precision}f}"])
            print(f"Wrote {len(rows)} rows to {args.out}")
        except Exception as e:
            print(f"[error] writing CSV {args.out}: {e}", file=sys.stderr)
    else:
        print(f"name,count,avg_{args.key}")
        for name, (count, avg) in rows:
            print(f"{name},{count},{avg:.{args.precision}f}")


if __name__ == "__main__":
    main()
