"""CLI to split a combined LoRA into content/style LoRAs by block lists.

Usage examples:
  python tools/split_lora.py in.safetensors --content-out content.safetensors --style-out style.safetensors --content-blocks 20-29 --style-blocks 30-57
"""
from __future__ import annotations
import argparse
import sys
from typing import List, Optional
from toolkit.split_lora import save_split_lora_from_file


def parse_block_range(s: str) -> List[int]:
    # supports comma separated values and ranges like 20-29
    out = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(list(range(int(a), int(b) + 1)))
        elif part == '':
            continue
        else:
            out.append(int(part))
    return out


def run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Split a combined LoRA into content and style LoRAs by block indices')
    parser.add_argument('in_file')
    parser.add_argument('--content-out', required=True)
    parser.add_argument('--style-out', required=True)
    parser.add_argument('--content-blocks', required=True, help='Comma separated blocks or ranges (e.g., 20-29,33)')
    parser.add_argument('--style-blocks', required=True, help='Comma separated blocks or ranges (e.g., 30-57)')
    parser.add_argument('--dtype', default=None, help='Optional dtype for output tensors (e.g., float32, float16)')
    parser.add_argument('--content-block-dims', default=None, help='Optional CSV of ints saved into metadata.block_dims for content')
    parser.add_argument('--style-block-dims', default=None, help='Optional CSV of ints saved into metadata.block_dims for style')

    args = parser.parse_args(argv)

    content_blocks = parse_block_range(args.content_blocks)
    style_blocks = parse_block_range(args.style_blocks)

    content_block_dims = None
    style_block_dims = None
    if args.content_block_dims:
        content_block_dims = [int(x) for x in args.content_block_dims.split(',') if x.strip()]
    if args.style_block_dims:
        style_block_dims = [int(x) for x in args.style_block_dims.split(',') if x.strip()]

    dtype = None
    if args.dtype is not None:
        import torch
        if args.dtype.lower() in ('fp16', 'float16'):
            dtype = torch.float16
        elif args.dtype.lower() in ('fp32', 'float32'):
            dtype = torch.float32
        elif args.dtype.lower() in ('bf16',):
            dtype = torch.bfloat16

    save_split_lora_from_file(
        args.in_file,
        args.content_out,
        args.style_out,
        content_blocks,
        style_blocks,
        metadata={'cli_invoked': 'split_lora'},
        dtype=dtype,
        content_block_dims=content_block_dims,
        style_block_dims=style_block_dims,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(run())