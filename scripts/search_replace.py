#!/usr/bin/env python3
"""Search and replace text within explicitly selected folders.

Dry-run is the default. Pass --apply to write changes atomically.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path, help="Folders to search")
    parser.add_argument("--search", required=True, help="Text or regular expression to find")
    parser.add_argument("--replace", required=True, help="Replacement text")
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="File extension to include; repeat as needed (example: --extension .txt)",
    )
    parser.add_argument("--regex", action="store_true", help="Treat --search as a regular expression")
    parser.add_argument("--ignore-case", action="store_true", help="Match without regard to case")
    parser.add_argument("--no-recursive", action="store_true", help="Only inspect each folder's direct files")
    parser.add_argument("--apply", action="store_true", help="Write changes; otherwise only report them")
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    flags = re.IGNORECASE if args.ignore_case else 0
    pattern = re.compile(args.search if args.regex else re.escape(args.search), flags)
    replacement = args.replace if args.regex else (lambda _match: args.replace)
    extensions = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in args.extension
    }
    changed_files = 0
    replacement_count = 0
    seen: set[Path] = set()

    for supplied_folder in args.folders:
        folder = supplied_folder.resolve()
        if not folder.is_dir():
            raise SystemExit(f"Folder does not exist: {folder}")
        candidates = folder.glob("*") if args.no_recursive else folder.rglob("*")
        for path in sorted(candidate for candidate in candidates if candidate.is_file()):
            resolved = path.resolve()
            if resolved in seen or (extensions and path.suffix.lower() not in extensions):
                continue
            seen.add(resolved)
            try:
                original = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                print(f"SKIP non-UTF-8: {path}")
                continue
            updated, count = pattern.subn(replacement, original)
            if count == 0:
                continue
            changed_files += 1
            replacement_count += count
            action = "WRITE" if args.apply else "WOULD WRITE"
            print(f"{action}: {path} ({count} replacement{'s' if count != 1 else ''})")
            if args.apply:
                atomic_write(path, updated)

    mode = "Applied" if args.apply else "Found"
    suffix = "" if args.apply else "; no files written"
    print(f"{mode} {replacement_count} replacements across {changed_files} files{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
