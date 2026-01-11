#!/usr/bin/env python3
"""Inspect a .safetensors file and summarize LoRA modules by block index.

Usage:
  python inspect_safetensors_lora.py <path.safetensors> [--json out.json]

Outputs summary to stdout and optional JSON file for further inspection.
"""
import sys
import json
from pathlib import Path
try:
    from safetensors.torch import load_file
except Exception as exc:
    print("ERROR: safetensors.torch not available:", exc)
    sys.exit(2)


def main():
    if len(sys.argv) < 2:
        print("Usage: inspect_safetensors_lora.py <path.safetensors> [--json out.json]")
        sys.exit(2)
    path = Path(sys.argv[1])
    out_json = None
    if len(sys.argv) >= 3 and sys.argv[2] == "--json" and len(sys.argv) >= 4:
        out_json = Path(sys.argv[3])

    if not path.exists():
        print("File not found:", path)
        sys.exit(1)

    sd = load_file(str(path))
    keys = list(sd.keys())
    print(f"Loaded {path} ({len(keys)} keys)")

    tops = sorted(set(k.split('.')[0] for k in keys))
    print(f"Top-level modules: {len(tops)}")

    # Candidate LoRA module prefixes
    lora_tops = [t for t in tops if t.startswith('lora_unet') or t.startswith('lora_') or t.startswith('lora_te')]
    print(f"Candidate LoRA modules: {len(lora_tops)} (sample: {lora_tops[:10]})")

    # Attempt to import get_block_index for mapping names to block indices
    try:
        from toolkit.kohya_lora import get_block_index
    except Exception as exc:
        print("Warning: couldn't import get_block_index:", exc)
        def get_block_index(name):
            return -1

    module_details = {t: {'block': get_block_index(t), 'keys': []} for t in lora_tops}

    for k in keys:
        top = k.split('.')[0]
        if top in module_details:
            module_details[top]['keys'].append(k)

    by_block = {}
    for top, d in module_details.items():
        by_block.setdefault(d['block'], []).append(top)

    print(f"Distinct block indices found: {len(by_block)}")
    for b in sorted(by_block.keys(), key=lambda x: (x if isinstance(x, int) else -1)):
        items = by_block[b]
        print(f"block {b}: {len(items)} modules, sample: {items[:5]}")

    # Print up to 40 lora_down weight shapes
    printed = 0
    for k in keys:
        top = k.split('.')[0]
        if top in module_details and 'lora_down' in k:
            arr = sd[k]
            print(f"{top} {k} shape: {tuple(arr.shape)}")
            printed += 1
            if printed >= 40:
                break

    # Print counts for indices commonly discussed (internal 19..56 -> user-facing 20..57)
    print('\nCounts for internal indices 19..56 (user blocks 20..57):')
    for i in range(19, 57):
        print(f"index {i}: {len(by_block.get(i, []))}")

    if out_json is not None:
        out = {
            'path': str(path),
            'num_keys': len(keys),
            'modules': module_details,
            'by_block_counts': {str(k): len(v) for k, v in by_block.items()},
        }
        out_json.write_text(json.dumps(out, indent=2))
        print(f"Wrote JSON summary to {out_json}")

    print('\nDone')


if __name__ == '__main__':
    main()
