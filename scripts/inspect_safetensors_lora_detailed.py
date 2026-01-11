#!/usr/bin/env python3
"""Detailed inspector for LoRA-style entries in a safetensors file.
Usage: python inspect_safetensors_lora_detailed.py <path.safetensors>
Prints: first keys, counts of 'lora_up'/'lora_down', per-block key counts for blocks 20..57, and sample nonzero stats.
"""
import sys
from pathlib import Path
try:
    from safetensors.torch import load_file
except Exception as exc:
    print("ERROR: safetensors.torch not available:", exc)
    sys.exit(2)

if len(sys.argv) < 2:
    print('Usage: inspect_safetensors_lora_detailed.py <path.safetensors>')
    sys.exit(2)

p = Path(sys.argv[1])
if not p.exists():
    print('File not found', p)
    sys.exit(1)

sd = load_file(str(p))
keys = list(sd.keys())
print(f"Loaded {p} ({len(keys)} keys)")
print('\nFirst 200 keys:')
for k in keys[:200]:
    print(k)

has_lora_down = [k for k in keys if 'lora_down' in k]
has_lora_up = [k for k in keys if 'lora_up' in k]
has_down_blocks_explicit = [k for k in keys if 'down_blocks_' in k]
has_up_blocks_explicit = [k for k in keys if 'up_blocks_' in k]

print('\nCounts:')
print('keys containing lora_down:', len(has_lora_down))
print('keys containing lora_up:  ', len(has_lora_up))
print('keys containing down_blocks_: ', len(has_down_blocks_explicit))
print('keys containing up_blocks_:   ', len(has_up_blocks_explicit))

print('\nSample lora_down keys (up to 20):')
for k in has_lora_down[:20]:
    print(k)

# Check per-block presence for user-facing blocks 20..57 (internal 19..56)
print('\nPer-block counts for user-facing blocks 20..57')
for user_block in range(20, 58):
    substr1 = f"down_blocks_{user_block}_"
    substr2 = f"up_blocks_{user_block}_"
    cnt = sum(1 for k in keys if substr1 in k or substr2 in k)
    print(f"block {user_block}: {cnt} keys")

# For lora_down keys, inspect nonzero stats for those that mention block >= 30
print('\nNonzero stats for lora_down keys mentioning blocks >= 30:')
printed = 0
for k in has_lora_down:
    # attempt to find a block number in the string
    import re
    m = re.search(r"_(down|up)_blocks_(\d+)_", k)
    if not m:
        # try alternative pattern
        m = re.search(r"down_blocks_(\d+)_", k)
    if m:
        blk = int(m.groups()[-1])
        if blk >= 30:
            arr = sd[k]
            nz = (arr != 0).sum()
            total = arr.size
            print(f"{k} -> block {blk}: shape={tuple(arr.shape)}, nonzero={int(nz)}/{total}")
            printed += 1
            if printed >= 30:
                break

# If no explicit 'lora_down' keys (they may be stored in merged weights), try to detect merged weight changes
if len(has_lora_down) == 0:
    print('\nNo explicit lora_down keys found. Possible explanations:')
    print('- LoRA weights were merged into base model weights before saving, or')
    print('- naming scheme differs (e.g., prefixes removed).')
    print('Listing keys that contain numeric block patterns:')
    num_keys = [k for k in keys if 'blocks_' in k]
    for k in num_keys[:100]:
        print(k)

print('\nDone')
