#!/usr/bin/env python3
"""Rewrite dataset caption files from an editable JSON bundle."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path, PurePosixPath


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite caption .txt files from a caption JSON bundle."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "tmp" / "captions.json",
        help="JSON bundle to read (default: repo/tmp/captions.json).",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=REPO_ROOT / "datasets",
        help="Directory containing the dataset folders (default: repo/datasets).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and list changes without writing files.",
    )
    return parser.parse_args()


def load_captions(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("format") != "ai-toolkit-caption-bundle" or payload.get("version") != 1:
        raise SystemExit("Unsupported caption bundle format or version")
    captions = payload.get("captions")
    if not isinstance(captions, dict):
        raise SystemExit("The JSON field 'captions' must be an object")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in captions.items()):
        raise SystemExit("Every caption path and caption value must be a string")
    return captions


def safe_destination(root: Path, relative: str) -> Path:
    posix_path = PurePosixPath(relative)
    if posix_path.is_absolute() or ".." in posix_path.parts or posix_path.suffix.lower() != ".txt":
        raise SystemExit(f"Unsafe or non-caption path in bundle: {relative!r}")
    destination = root.joinpath(*posix_path.parts).resolve()
    if os.path.commonpath((str(root), str(destination))) != str(root):
        raise SystemExit(f"Caption path escapes destination root: {relative!r}")
    return destination


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(text.rstrip("\r\n"))
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    destination_root = args.destination_root.resolve()
    captions = load_captions(input_path)

    destinations = [
        (relative, safe_destination(destination_root, relative), text)
        for relative, text in sorted(captions.items())
    ]
    if args.dry_run:
        for relative, destination, _ in destinations:
            print(f"Would write {relative} -> {destination}")
        print(f"Validated {len(destinations)} captions; no files written")
        return 0

    for _, destination, text in destinations:
        atomic_write_text(destination, text)
    print(f"Wrote {len(destinations)} captions beneath {destination_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
