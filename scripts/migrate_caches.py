"""Migration helper to propose & apply renames for legacy caches to new hashed format.

Usage:
    python scripts/migrate_caches.py --dir <image_dir> [--apply]

This tool runs in dry-run mode by default and prints suggested renames.
"""
import argparse
from pathlib import Path
from toolkit.cache_utils import compute_file_sha256
import os

CACHE_DIRS = ['_latent_cache', '_t_e_cache', '_controls', '_context_cache']


def propose_renames(image_dir: Path):
    image_dir = Path(image_dir)
    suggestions = []
    for root, dirs, files in os.walk(image_dir):
        for d in dirs:
            if d in CACHE_DIRS:
                cache_dir = Path(root) / d
                img_parent = cache_dir.parent
                # try to find the source image for this folder
                for f in cache_dir.iterdir():
                    if not f.is_file():
                        continue
                    # simple heuristic: extract base (prefix before first underscore)
                    name = f.name
                    if '_' not in name:
                        continue
                    base = name.split('_', 1)[0]
                    # find image with that base in parent
                    candidates = list(img_parent.glob(base + '.*'))
                    source = None
                    if len(candidates) > 0:
                        source = candidates[0]
                        content = compute_file_sha256(source)
                        new_name = f"{base}_legacy_{content}{f.suffix}"
                        suggestions.append((f, cache_dir / new_name))
    return suggestions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    s = propose_renames(Path(args.dir))
    if not s:
        print("No candidate legacy cache files found")
        return
    for old, new in s:
        print(f"{old} -> {new}")
    if args.apply:
        for old, new in s:
            os.replace(str(old), str(new))
        print("Applied renames")

if __name__ == '__main__':
    main()