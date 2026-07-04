#!/usr/bin/env python3
"""Apply exact UTF-8 source edits without shell-escaped multiline strings.

Examples:
    python scripts/exact_edit.py replace path/to/file.py old.txt new.txt
    python scripts/exact_edit.py splice path/to/file.py --start "def f(" --end-before "def g(" new.txt

The script fails when anchors/replacements are not found. It preserves the
original file newline style when normalizing text read from helper files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _newline_style(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def _read_helper(path: Path, newline: str) -> str:
    text = path.read_text(encoding="utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.replace("\n", newline)


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="")


def replace(args: argparse.Namespace) -> int:
    path = Path(args.path)
    text = path.read_text(encoding="utf-8-sig")
    newline = _newline_style(text)
    old = _read_helper(Path(args.old_file), newline)
    new = _read_helper(Path(args.new_file), newline)
    found = text.count(old)
    if found == 0:
        raise SystemExit(f"target text not found in {path}")
    if args.count is not None and found < args.count:
        raise SystemExit(f"requested {args.count} replacements but found {found} in {path}")
    count = args.count if args.count is not None else found
    _write(path, text.replace(old, new, count))
    print(f"replaced {count} occurrence(s) in {path}")
    return 0


def splice(args: argparse.Namespace) -> int:
    path = Path(args.path)
    text = path.read_text(encoding="utf-8-sig")
    newline = _newline_style(text)
    new = _read_helper(Path(args.new_file), newline)
    start = args.start.replace("\\n", newline)
    end_before = args.end_before.replace("\\n", newline)
    start_idx = text.find(start)
    if start_idx < 0:
        raise SystemExit(f"start anchor not found in {path}: {args.start!r}")
    end_idx = text.find(end_before, start_idx + len(start))
    if end_idx < 0:
        raise SystemExit(f"end-before anchor not found in {path}: {args.end_before!r}")
    _write(path, text[:start_idx] + new + text[end_idx:])
    print(f"spliced {path}")
    return 0


def insert(args: argparse.Namespace) -> int:
    path = Path(args.path)
    text = path.read_text(encoding="utf-8-sig")
    newline = _newline_style(text)
    payload = _read_helper(Path(args.new_file), newline)
    anchor = args.anchor.replace("\\n", newline)
    idx = text.find(anchor)
    if idx < 0:
        raise SystemExit(f"anchor not found in {path}: {args.anchor!r}")
    if args.where == "after":
        idx += len(anchor)
        text = text[:idx] + payload + text[idx:]
    else:
        text = text[:idx] + payload + text[idx:]
    _write(path, text)
    print(f"inserted {args.where} anchor in {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("replace", help="replace exact text read from files")
    p.add_argument("path")
    p.add_argument("old_file")
    p.add_argument("new_file")
    p.add_argument("--count", type=int, default=None)
    p.set_defaults(func=replace)

    p = sub.add_parser("splice", help="replace text from --start through before --end-before")
    p.add_argument("path")
    p.add_argument("--start", required=True)
    p.add_argument("--end-before", required=True)
    p.add_argument("new_file")
    p.set_defaults(func=splice)

    p = sub.add_parser("insert", help="insert text from a file before/after an anchor")
    p.add_argument("path")
    p.add_argument("--anchor", required=True)
    p.add_argument("new_file")
    p.add_argument("--where", choices=("before", "after"), default="after")
    p.set_defaults(func=insert)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
