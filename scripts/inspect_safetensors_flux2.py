#!/usr/bin/env python3
"""Inspect safetensors for Flux/Flux2-style transformer block LoRA entries.

Searches for keys matching common Flux2 naming conventions (transformer_blocks, single_transformer_blocks,
or layer lists like diffusion_model.layers.<i>) and reports counts and nonzero statistics for blocks >= 30.
"""
import sys
import re
from pathlib import Path
try:
    from safetensors.torch import load_file
except Exception as exc:
    print("ERROR: safetensors.torch not available:", exc)
    sys.exit(2)

if len(sys.argv) < 2:
    print('Usage: inspect_safetensors_flux2.py <path.safetensors>')
    sys.exit(2)

p = Path(sys.argv[1])
if not p.exists():
    print('File not found', p)
    sys.exit(1)

sd = load_file(str(p))
keys = list(sd.keys())
print(f"Loaded {p} ({len(keys)} keys)")

# patterns to detect Flux2/DiT block naming
regexes = [
    (re.compile(r"transformer_blocks\.(\d+)\."), 'transformer_blocks'),
    (re.compile(r"single_transformer_blocks\.(\d+)\."), 'single_transformer_blocks'),
    (re.compile(r"diffusion_model\.transformer_blocks\.(\d+)\."), 'diffusion_model.transformer_blocks'),
    (re.compile(r"diffusion_model\.single_transformer_blocks\.(\d+)\."), 'diffusion_model.single_transformer_blocks'),
    (re.compile(r"diffusion_model\.layers\.(\d+)\."), 'diffusion_model.layers'),
    (re.compile(r"layers\.(\d+)\."), 'layers'),
]

block_map = {}
for k in keys:
    for rx, name in regexes:
        m = rx.search(k)
        if m:
            idx = int(m.group(1))
            block_map.setdefault((name, idx), []).append(k)

# summarize
print('\nDetected block groups:')
for (name, idx), ks in sorted(block_map.items(), key=lambda x: (x[0][0], x[0][1])):
    print(f"{name}[{idx}]: {len(ks)} keys (sample: {ks[:3]})")

# focus on blocks >= 30 (user asked about >30)
threshold = 30
print(f"\nSummary for blocks with index >= {threshold}:")
found_any = False
for (name, idx), ks in sorted(block_map.items(), key=lambda x: (x[0][0], x[0][1])):
    if idx >= threshold:
        found_any = True
        # count keys that look like LoRA params (lora_A/B/lora.weight etc)
        lora_like = [k for k in ks if 'lora' in k or '.lora_' in k]
        # compute nonzero counts for any lora_A/B entries
        nonzero_stats = []
        for k in lora_like:
            try:
                arr = sd[k]
                nz = int((arr != 0).sum())
                total = int(arr.size)
                nonzero_stats.append((k, nz, total))
            except Exception:
                continue
        print(f"{name}[{idx}] keys={len(ks)} lora_like={len(lora_like)} nonzero_entries={len(nonzero_stats)}")
        if nonzero_stats:
            for k,nz,total in nonzero_stats[:5]:
                print(f"  {k}: nonzero {nz}/{total}")

if not found_any:
    print('No blocks >= threshold detected using the configured Flux2 patterns.')

# Also check top-level keys patterns that indicate double vs single stream presence
print('\nTop-level name samples:')
top_samples = sorted(set(k.split('.')[0] for k in keys))[:50]
print(top_samples)

print('\nDone')
