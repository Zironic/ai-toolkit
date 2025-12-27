"""Precompute Canny control images for a dataset.

Saves canny images into <dataset_root>/canny/ with filenames matching the source image
(e.g., image_0001.jpg -> image_0001_canny.png) and writes an atomic manifest JSON.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def debug(msg: str, *args):
    print(f"[PRECOMPUTE] {msg}".format(*args))


def make_canny_image(pil_img: Image.Image, threshold1: int, threshold2: int, blur: int) -> Image.Image:
    # convert to grayscale numpy
    arr = np.array(pil_img.convert("L"))
    if blur and blur > 0:
        arr = cv2.GaussianBlur(arr, (blur | 1, blur | 1), 0)
    edges = cv2.Canny(arr, threshold1, threshold2)
    # convert back to PIL (single-channel)
    return Image.fromarray(edges)


def atomic_write_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def precompute_dataset(
    dataset_root: Path,
    out_dir: Path = None,
    threshold1: int = 100,
    threshold2: int = 200,
    blur: int = 3,
    control_size: int = None,
    overwrite: bool = False,
) -> dict:
    dataset_root = Path(dataset_root)
    if out_dir is None:
        out_dir = dataset_root / "canny"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_root": str(dataset_root),
        "canny_folder": str(out_dir),
        "params": {
            "threshold1": threshold1,
            "threshold2": threshold2,
            "blur": blur,
            "control_size": control_size,
        },
        "files": {},
    }

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = [p for p in dataset_root.iterdir() if p.suffix.lower() in img_exts and p.is_file()]

    for src in sorted(files):
        name = src.stem
        out_path = out_dir / f"{name}_canny.png"
        if out_path.exists() and not overwrite:
            debug("Skipping existing %s", out_path)
            manifest["files"][str(src.name)] = str(out_path.name)
            continue

        debug("Processing %s -> %s", src.name, out_path.name)
        with Image.open(src) as im:
            canny = make_canny_image(im, threshold1, threshold2, blur)
            if control_size is not None:
                canny = canny.resize((control_size, control_size), Image.BICUBIC)
            canny.save(out_path)
            manifest["files"][str(src.name)] = str(out_path.name)

    manifest_path = out_dir / "canny_manifest.json"
    atomic_write_json(manifest_path, manifest)
    debug("Wrote manifest to %s", manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Precompute Canny control images for a dataset")
    parser.add_argument("dataset_root", help="Path to dataset root containing images")
    parser.add_argument("--threshold1", type=int, default=100)
    parser.add_argument("--threshold2", type=int, default=200)
    parser.add_argument("--blur", type=int, default=3)
    parser.add_argument("--control-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = precompute_dataset(
        Path(args.dataset_root),
        threshold1=args.threshold1,
        threshold2=args.threshold2,
        blur=args.blur,
        control_size=args.control_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
