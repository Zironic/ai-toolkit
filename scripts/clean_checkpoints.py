#!/usr/bin/env python3
"""
Clean up stale training checkpoints.

Rules
-----
1. Aborted run  (max step < ABORTED_THRESHOLD):
   Delete ALL checkpoints if the run hasn't been touched in ABORTED_AGE_HOURS.

2. Completed run (max step >= ABORTED_THRESHOLD):
   Delete checkpoints whose step is below 50% of the run's maximum step.

Safety net: never touch a run whose most recent file was modified within
SAFETY_WINDOW_HOURS hours, regardless of either rule.
"""

import argparse
import re
import time
from collections import defaultdict
from pathlib import Path

CHECKPOINT_RE = re.compile(r'^.+_(\d{9})\.safetensors$')

ABORTED_THRESHOLD   = 1500   # runs that stopped below this step are "aborted"
ABORTED_AGE_HOURS   = 168    # aborted runs must be at least this old to be cleaned (1 week)
SAFETY_WINDOW_HOURS = 2      # never touch anything modified more recently than this


def scan_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    return sorted(
        (int(m.group(1)), f)
        for f in run_dir.iterdir()
        if f.is_file() and (m := CHECKPOINT_RE.match(f.name))
    )


def run_age_hours(run_dir: Path) -> float:
    now = time.time()
    mtimes = [f.stat().st_mtime for f in run_dir.iterdir() if f.is_file()]
    return (now - max(mtimes)) / 3600 if mtimes else float('inf')


def find_deletions(
    run_dir: Path,
    aborted_threshold: int,
    aborted_age_hours: float,
    safety_window_hours: float,
) -> tuple[list[Path], str]:
    checkpoints = scan_checkpoints(run_dir)
    if not checkpoints:
        return [], "no checkpoints found"

    age = run_age_hours(run_dir)
    max_step = checkpoints[-1][0]

    if age < safety_window_hours:
        return [], f"modified {age:.1f}h ago — skipping (active or very recent)"

    if max_step < aborted_threshold:
        if age < aborted_age_hours:
            return [], f"aborted at step {max_step} but only {age:.1f}h old — skipping"
        return (
            [p for _, p in checkpoints],
            f"aborted at step {max_step}, {age:.0f}h old -> delete all {len(checkpoints)} checkpoints",
        )

    cutoff = max_step // 2
    to_delete = [p for step, p in checkpoints if step < cutoff]
    kept = len(checkpoints) - len(to_delete)
    return (
        to_delete,
        f"max step {max_step}, cutoff {cutoff} -> delete {len(to_delete)}, keep {kept}",
    )


def fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1 << 30:
        return f"{n_bytes / (1 << 30):.1f} GB"
    return f"{n_bytes / (1 << 20):.0f} MB"


def main():
    parser = argparse.ArgumentParser(
        description="Remove stale mid-training checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("output_dir", help="Root output directory (contains one subdir per run)")
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually delete files. Without this flag the script only prints what it would do.",
    )
    parser.add_argument(
        "--aborted-threshold", type=int, default=ABORTED_THRESHOLD, metavar="N",
        help=f"Runs with max step < N are treated as aborted (default: {ABORTED_THRESHOLD})",
    )
    parser.add_argument(
        "--aborted-age", type=float, default=ABORTED_AGE_HOURS, metavar="HOURS",
        help=f"Aborted runs must be at least this many hours old to be cleaned (default: {ABORTED_AGE_HOURS} = 1 week)",
    )
    parser.add_argument(
        "--safety-window", type=float, default=SAFETY_WINDOW_HOURS, metavar="HOURS",
        help=f"Never touch runs modified within this many hours (default: {SAFETY_WINDOW_HOURS})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        parser.error(f"Not a directory: {output_dir}")

    aborted_threshold   = args.aborted_threshold
    aborted_age_hours   = args.aborted_age
    safety_window_hours = args.safety_window
    dry_run = not args.execute
    if dry_run:
        print("DRY RUN — pass --execute to actually delete\n")

    total_files = 0
    total_bytes = 0

    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        to_delete, reason = find_deletions(run_dir, aborted_threshold, aborted_age_hours, safety_window_hours)

        if not to_delete:
            print(f"  skip    {run_dir.name}: {reason}")
            continue

        size = sum(p.stat().st_size for p in to_delete)
        print(f"  {'dry-run' if dry_run else 'DELETE '}  {run_dir.name}: {reason} ({fmt_size(size)})")

        for p in to_delete:
            if dry_run:
                print(f"            {p.name}")
            else:
                p.unlink()
                print(f"    deleted {p.name}")

        total_files += len(to_delete)
        total_bytes += size

    print(f"\n{'Would free' if dry_run else 'Freed'} {fmt_size(total_bytes)} across {total_files} files.")
    if dry_run:
        print("Run with --execute to apply.")


if __name__ == "__main__":
    main()
