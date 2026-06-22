"""
Generate one image per caption file using a toolkit diffusion model.

Reads every caption (`.txt` by default) in a directory, treats the text of each
file as a prompt, and renders an image for it. Images are written to
    C:\\GenAI\\ai-toolkit\\tmp\\<captions-dir-name>\\<caption-stem>.png
so a caption file named `cat_on_a_roof.txt` produces `cat_on_a_roof.png`.

This reuses the toolkit's own model loading + sampling path, so it honors the
project's custom flow-match sampler and quantization handling.

Defaults: Z-Image base, 8-bit (qfloat8) quant, 512x512, 25 steps, cfg 4.

Usage:
    python scripts/generate_from_captions.py path/to/captions
    python scripts/generate_from_captions.py path/to/captions --steps 30 --cfg 3.5
    python scripts/generate_from_captions.py path/to/captions --no-quant --width 768 --height 768
"""

import argparse
import os
import sys

# add project root to sys path so `toolkit` imports resolve when run directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import torch

from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.util.get_model import get_model_class

# default output root requested for this project
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "tmp")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one image per caption file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "captions_dir",
        type=str,
        help="Directory containing caption files (one prompt per file).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Tongyi-MAI/Z-Image",
        help="Model repo id or local path.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="zimage",
        help="Model architecture key (see toolkit/util/get_model.py).",
    )
    parser.add_argument(
        "--caption-ext",
        type=str,
        default="txt",
        help="Caption file extension (without the dot).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output dir. Images land in <root>/<captions-dir-name>/.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=4.0, help="guidance_scale")
    parser.add_argument(
        "--negative",
        type=str,
        default="",
        help="Negative prompt applied to every generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for every image. Use -1 for a random seed per image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="Compute dtype for the model.",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="Disable quantization (default is 8-bit qfloat8 on transformer + TE).",
    )
    parser.add_argument(
        "--qtype",
        type=str,
        default="qfloat8",
        help="Quantization type when quantizing (qfloat8 = 8-bit float).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if the output image already exists.",
    )
    return parser.parse_args()


def collect_captions(captions_dir, caption_ext):
    """Return a sorted list of (stem, prompt) for every non-empty caption file."""
    ext = caption_ext.lstrip(".").lower()
    captions = []
    for entry in sorted(os.listdir(captions_dir)):
        path = os.path.join(captions_dir, entry)
        if not os.path.isfile(path):
            continue
        stem, file_ext = os.path.splitext(entry)
        if file_ext.lstrip(".").lower() != ext:
            continue
        with open(path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        if not prompt:
            print(f"  skipping empty caption: {entry}")
            continue
        captions.append((stem, prompt))
    return captions


def main():
    args = parse_args()

    captions_dir = os.path.abspath(args.captions_dir)
    if not os.path.isdir(captions_dir):
        raise NotADirectoryError(f"captions_dir is not a directory: {captions_dir}")

    output_dir = os.path.join(args.output_root, os.path.basename(captions_dir.rstrip("/\\")))
    os.makedirs(output_dir, exist_ok=True)

    captions = collect_captions(captions_dir, args.caption_ext)
    if not captions:
        raise ValueError(
            f"No '.{args.caption_ext}' caption files found in {captions_dir}"
        )
    print(f"Found {len(captions)} caption(s) in {captions_dir}")
    print(f"Output dir: {output_dir}")

    # ── Build the model ──────────────────────────────────────────────────────
    quantize = not args.no_quant
    model_config = ModelConfig(
        name_or_path=args.model,
        arch=args.arch,
        dtype=args.dtype,
        quantize=quantize,
        quantize_te=quantize,
        qtype=args.qtype,
        qtype_te=args.qtype,
    )

    print(
        f"Loading model '{args.model}' (arch={model_config.arch}, "
        f"quant={'on:' + args.qtype if quantize else 'off'}, dtype={args.dtype})"
    )
    ModelClass = get_model_class(model_config)
    model = ModelClass(device=args.device, model_config=model_config, dtype=args.dtype)
    model.load_model()

    # ── Build a generation config per caption ────────────────────────────────
    gen_configs = []
    for stem, prompt in captions:
        out_path = os.path.join(output_dir, f"{stem}.png")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  exists, skipping: {stem}.png (use --overwrite to force)")
            continue
        gen_configs.append(
            GenerateImageConfig(
                prompt=prompt,
                width=args.width,
                height=args.height,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                negative_prompt=args.negative,
                seed=args.seed,
                output_path=out_path,
            )
        )

    if not gen_configs:
        print("Nothing to generate (all outputs already exist).")
        return

    print(
        f"Generating {len(gen_configs)} image(s): "
        f"{args.width}x{args.height}, steps={args.steps}, cfg={args.cfg}, seed={args.seed}"
    )
    model.generate_images(gen_configs)
    print(f"Done. Images saved to {output_dir}")


if __name__ == "__main__":
    main()
