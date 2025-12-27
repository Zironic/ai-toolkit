import json
import time
from pathlib import Path
from PIL import Image
import os

from tools.precompute_control import precompute_dataset


def create_dummy_images(tmp_path, n=3):
    for i in range(n):
        img = Image.new('RGB', (64, 64), color=(i * 40, i * 40, i * 40))
        p = tmp_path / f"image_{i:03d}.jpg"
        img.save(p)
    return [tmp_path / f"image_{i:03d}.jpg" for i in range(n)]


def test_precompute_idempotence(tmp_path):
    images = create_dummy_images(tmp_path, n=3)

    out_dir = tmp_path / "canny"
    manifest1 = precompute_dataset(tmp_path, out_dir=out_dir, threshold1=50, threshold2=150, blur=1, overwrite=False)
    assert (out_dir / "canny_manifest.json").exists()
    with open(out_dir / "canny_manifest.json") as f:
        m1 = json.load(f)
    assert len(m1['files']) == 3

    times = {}
    for f in images:
        times[f.name] = (out_dir / f"{f.stem}_canny.png").stat().st_mtime

    # Re-run with overwrite=False should not change timestamps
    time.sleep(0.1)
    manifest2 = precompute_dataset(tmp_path, out_dir=out_dir, threshold1=50, threshold2=150, blur=1, overwrite=False)
    for f in images:
        assert (out_dir / f"{f.stem}_canny.png").stat().st_mtime == times[f.name]

    # Re-run with overwrite=True should update timestamps
    time.sleep(0.1)
    manifest3 = precompute_dataset(tmp_path, out_dir=out_dir, threshold1=50, threshold2=150, blur=1, overwrite=True)
    for f in images:
        assert (out_dir / f"{f.stem}_canny.png").stat().st_mtime != times[f.name]


def test_manifest_contents(tmp_path):
    create_dummy_images(tmp_path, n=1)
    out_dir = tmp_path / "canny"
    manifest = precompute_dataset(tmp_path, out_dir=out_dir, threshold1=12, threshold2=34, blur=0)
    p = out_dir / "canny_manifest.json"
    assert p.exists()
    with open(p) as f:
        m = json.load(f)
    assert m['params']['threshold1'] == 12
    assert m['params']['threshold2'] == 34
    assert list(m['files'].keys())[0] == 'image_000.jpg'
