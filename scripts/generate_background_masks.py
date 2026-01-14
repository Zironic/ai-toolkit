#!/usr/bin/env python
"""Generate background masks by matching an exact RGB color and inverting.

Usage examples:
  python scripts/generate_background_masks.py --dataset datasets/my_dataset --color #ffffff --output datasets/my_dataset/masks
  python scripts/generate_background_masks.py --image datasets/my_dataset/imgs/foo.png --color ffffff --output datasets/my_dataset/masks

This script prints a JSON object on success: {"success": true, "masks": ["relative/path/to/mask1.png", ...]}
On failure it exits non-zero and prints an error message to stderr.
"""

import argparse
import json
import os
from pathlib import Path
from PIL import Image

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}


def hex_to_rgb(hexstr: str):
    h = hexstr.strip().lstrip('#')
    if len(h) != 6:
        raise ValueError('Invalid color hex string: expected 6 hex digits')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def find_images_recursive(root: Path):
    out = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            # skip control or mask folders
            if any(part in ('_controls', 'masks') for part in p.parts):
                continue
            out.append(p)
    return out


def process_one_image(img_path: Path, rgb_match, out_dir: Path):
    img = Image.open(img_path).convert('RGBA')
    w, h = img.size
    px = img.load()

    mask = Image.new('L', (w, h))
    mask_px = mask.load()

    rM, gM, bM = rgb_match
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            match = (r == rM and g == gM and b == bM)
            # invert: mask 255 when NOT matching background color
            mask_px[x, y] = 0 if match else 255

    out_name = img_path.stem + '.png'
    out_path = out_dir / out_name
    mask.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dataset', help='Path to dataset folder containing images')
    group.add_argument('--image', help='Single image file to process')
    parser.add_argument('--color', required=True, help='Hex color to match as background (e.g., #ffffff)')
    parser.add_argument('--output', required=False, help='Output directory for masks; defaults to <dataset>/masks or image parent/masks')
    args = parser.parse_args()

    try:
        rgb = hex_to_rgb(args.color)
    except Exception as e:
        print(f'Invalid color: {e}', flush=True)
        raise SystemExit(2)

    if args.dataset:
        dataset_path = Path(args.dataset)
        if not dataset_path.exists() or not dataset_path.is_dir():
            print(f'Dataset path not found or not a directory: {args.dataset}', flush=True)
            raise SystemExit(3)
        images = find_images_recursive(dataset_path)
        out_dir = Path(args.output) if args.output else dataset_path / 'masks'
    else:
        img_path = Path(args.image)
        if not img_path.exists() or not img_path.is_file():
            print(f'Image not found: {args.image}', flush=True)
            raise SystemExit(4)
        images = [img_path]
        out_dir = Path(args.output) if args.output else img_path.parent / 'masks'

    out_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for img in images:
        try:
            out_mask = process_one_image(img, rgb, out_dir)
            rel = os.path.relpath(out_mask, start=dataset_path if args.dataset else img.parent)
            created.append(rel)
        except Exception as e:
            print(f'Failed processing {img}: {e}', flush=True)

    # Print JSON result to stdout for the caller to parse
    print(json.dumps({'success': True, 'masks': created}), flush=True)


if __name__ == '__main__':
    main()
