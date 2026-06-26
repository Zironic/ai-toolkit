#!/usr/bin/env python3
"""Collect dataset caption files into one editable JSON bundle."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = (
    "jinx_references_natural",
    "jinx_nobg_mini_natural",
    "live_action_jinx",
    "studio_alts",
    "live_action_jinx",
    "live_action_body",
    "live_action_jinx_canon",
    "live_action_krea_synths"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile caption .txt files into a single UTF-8 JSON file."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPO_ROOT / "datasets",
        help="Directory containing the dataset folders (default: repo/datasets).",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset folder to include; repeat as needed (default: the four Jinx datasets).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "tmp" / "captions.json",
        help="JSON bundle to create (default: repo/tmp/captions.json).",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
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
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    try:
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    datasets = tuple(args.datasets or DEFAULT_DATASETS)
    captions: dict[str, str] = {}

    for dataset in datasets:
        dataset_path = source_root / dataset
        if not dataset_path.is_dir():
            raise SystemExit(f"Dataset directory does not exist: {dataset_path}")
        for caption_path in sorted(dataset_path.rglob("*.txt")):
            relative_path = caption_path.relative_to(source_root).as_posix()
            captions[relative_path] = caption_path.read_text(encoding="utf-8").rstrip("\r\n")

    payload: dict[str, object] = {
        "format": "ai-toolkit-caption-bundle",
        "version": 1,
        "datasets": list(datasets),
        "captions": captions,
    }
    atomic_write_json(args.output.resolve(), payload)
    print(f"Wrote {len(captions)} captions to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
