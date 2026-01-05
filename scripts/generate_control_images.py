#!/usr/bin/env python
"""Simple CLI wrapper that uses ControlGenerator to generate control images for a folder.

This script is a convenience wrapper for dataset owners to pre-generate controls.
"""
import argparse
import os
from toolkit.control_generator import ControlGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', help='Root image directory')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--controls', default='depth,pose,line', help='Comma-separated control types')
    parser.add_argument('--regen', action='store_true')
    args = parser.parse_args()

    types = [t.strip() for t in args.controls.split(',') if t.strip()]
    cg = ControlGenerator(device=args.device)
    cg.regen = args.regen
    for root, dirs, files in os.walk(args.img_dir):
        for file in files:
            if file.startswith('.'):
                continue
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                p = os.path.join(root, file)
                for t in types:
                    try:
                        cg.get_control_path(p, t)
                    except Exception as e:
                        print(f"Failed to generate control {t} for {p}: {e}")


if __name__ == '__main__':
    main()
