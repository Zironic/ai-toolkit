#!/usr/bin/env python3
"""
Auto-crop images by detecting white/black backgrounds.

Detects pure white (#FFFFFF) or pure black (#000000) backgrounds, finds the
bounding box of content, and crops to that size (snapped to divisibility).
The dataloader will handle bucketing and any necessary padding/scaling.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def detect_background(img_array: np.ndarray) -> np.ndarray:
    """Detect if background is white or black by sampling corners."""
    h, w = img_array.shape[:2]
    corners = [
        img_array[0, 0],
        img_array[0, w - 1],
        img_array[h - 1, 0],
        img_array[h - 1, w - 1],
    ]
    avg = np.mean(corners, axis=0)
    # If average brightness > 128, assume white background
    if np.mean(avg) > 128:
        return np.array([255, 255, 255], dtype=np.uint8)
    else:
        return np.array([0, 0, 0], dtype=np.uint8)


def get_content_bbox(
    img_array: np.ndarray, bg_color: np.ndarray, threshold: int = 0
) -> tuple[int, int, int, int]:
    """Find bounding box of non-background content.

    Returns (x_min, y_min, x_max, y_max) - x_max/y_max are exclusive.
    """
    # Difference from background
    diff = np.abs(img_array.astype(np.float32) - bg_color.astype(np.float32))
    # Mask where any channel differs by more than threshold
    mask = diff.max(axis=-1) > threshold

    if not mask.any():
        # No content found, return full image
        return 0, 0, img_array.shape[1], img_array.shape[0]

    # Find bounding rows/cols
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    y_min, y_max = y_indices[0], y_indices[-1] + 1
    x_min, x_max = x_indices[0], x_indices[-1] + 1

    return x_min, y_min, x_max, y_max


def snap_to_divisibility(value: int, divisibility: int, round_up: bool = True) -> int:
    """Snap a value to the nearest multiple of divisibility."""
    if value % divisibility == 0:
        return value
    if round_up:
        return value + (divisibility - value % divisibility)
    else:
        return value - (value % divisibility)


def process_image(
    input_path: Path,
    output_path: Path,
    threshold: int = 0,
    background: str = "auto",
    divisibility: int = 16,
) -> dict:
    """Process a single image. Returns info dict."""
    # Load image
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    img_w, img_h = img.size

    # Detect or set background color
    if background == "auto":
        bg_color = detect_background(img_array)
    elif background == "white":
        bg_color = np.array([255, 255, 255], dtype=np.uint8)
    else:  # black
        bg_color = np.array([0, 0, 0], dtype=np.uint8)

    # Find content bounding box
    x_min, y_min, x_max, y_max = get_content_bbox(img_array, bg_color, threshold)
    content_w = x_max - x_min
    content_h = y_max - y_min

    # Snap dimensions up to divisibility
    target_w = snap_to_divisibility(content_w, divisibility, round_up=True)
    target_h = snap_to_divisibility(content_h, divisibility, round_up=True)

    # Calculate how much extra we need on each side
    extra_w = target_w - content_w
    extra_h = target_h - content_h

    # Distribute extra space, preferring to expand into existing image
    # (to capture more context rather than adding background)
    left_extra = extra_w // 2
    right_extra = extra_w - left_extra
    top_extra = extra_h // 2
    bottom_extra = extra_h - top_extra

    # Adjust bbox, clamping to image bounds
    new_x_min = max(0, x_min - left_extra)
    new_x_max = min(img_w, x_max + right_extra)
    new_y_min = max(0, y_min - top_extra)
    new_y_max = min(img_h, y_max + bottom_extra)

    # If we couldn't expand enough (hit image edge), expand the other direction
    actual_w = new_x_max - new_x_min
    actual_h = new_y_max - new_y_min

    if actual_w < target_w:
        shortfall = target_w - actual_w
        if new_x_min > 0:
            expand = min(new_x_min, shortfall)
            new_x_min -= expand
            shortfall -= expand
        if shortfall > 0 and new_x_max < img_w:
            new_x_max = min(img_w, new_x_max + shortfall)

    if actual_h < target_h:
        shortfall = target_h - actual_h
        if new_y_min > 0:
            expand = min(new_y_min, shortfall)
            new_y_min -= expand
            shortfall -= expand
        if shortfall > 0 and new_y_max < img_h:
            new_y_max = min(img_h, new_y_max + shortfall)

    # Final dimensions
    final_w = new_x_max - new_x_min
    final_h = new_y_max - new_y_min

    # Crop
    output_img = img.crop((new_x_min, new_y_min, new_x_max, new_y_max))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_img.save(output_path, quality=95)

    # Calculate stats
    original_pixels = img_w * img_h
    final_pixels = final_w * final_h
    reduction_pct = (1 - final_pixels / original_pixels) * 100

    return {
        "input": input_path.name,
        "original_size": (img_w, img_h),
        "content_size": (content_w, content_h),
        "final_size": (final_w, final_h),
        "reduction_pct": reduction_pct,
        "background": "white" if bg_color[0] == 255 else "black",
        "divisible": final_w % divisibility == 0 and final_h % divisibility == 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Auto-crop images by detecting white/black backgrounds."
    )
    parser.add_argument("input_folder", type=Path, help="Source folder with images")
    parser.add_argument("output_folder", type=Path, help="Destination folder for cropped images")
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Tolerance for background detection - 0 means exact match only (default: 0)",
    )
    parser.add_argument(
        "--background",
        choices=["auto", "white", "black"],
        default="auto",
        help="Background color: auto-detect, white, or black (default: auto)",
    )
    parser.add_argument(
        "--divisibility",
        type=int,
        default=16,
        help="Snap dimensions to multiples of this value (default: 16 for FLUX)",
    )
    parser.add_argument(
        "--copy-captions",
        action="store_true",
        help="Copy matching .txt caption files",
    )

    args = parser.parse_args()

    # Find all images
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    input_files = [
        f for f in args.input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not input_files:
        print(f"No images found in {args.input_folder}")
        return

    print(f"Processing {len(input_files)} images (divisibility={args.divisibility})...\n")

    total_original = 0
    total_final = 0

    for input_path in input_files:
        output_path = args.output_folder / input_path.name

        try:
            info = process_image(
                input_path,
                output_path,
                threshold=args.threshold,
                background=args.background,
                divisibility=args.divisibility,
            )

            ow, oh = info["original_size"]
            cw, ch = info["content_size"]
            fw, fh = info["final_size"]

            total_original += ow * oh
            total_final += fw * fh

            # Format output
            div_ok = "OK" if info["divisible"] else "!!"
            print(f"{info['input']}:")
            print(f"  {ow}x{oh} -> {fw}x{fh} (content: {cw}x{ch}) [{div_ok}]")
            print(f"  Reduction: {info['reduction_pct']:.1f}% | bg: {info['background']}")
            print()

            # Copy caption file if requested
            if args.copy_captions:
                caption_path = input_path.with_suffix(".txt")
                if caption_path.exists():
                    output_caption = args.output_folder / caption_path.name
                    shutil.copy2(caption_path, output_caption)

        except Exception as e:
            print(f"  ERROR processing {input_path.name}: {e}\n")

    # Summary
    if total_original > 0:
        total_reduction = (1 - total_final / total_original) * 100
        print(f"Total reduction: {total_reduction:.1f}% ({total_original:,} -> {total_final:,} pixels)")

    print("Done!")


if __name__ == "__main__":
    main()
