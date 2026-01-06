import os
from PIL import Image, ImageDraw
import numpy as np
import torch
import pytest

from scripts.preview_mask import make_control_mask, make_pose_coverage_mask


def _make_synthetic_control(path, size=(64, 64)):
    im = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(im)
    W, H = size
    # simple stick-figure: body line + two arms + legs
    draw.line((W//2, H//4, W//2, H*3//4), fill=(255,255,255), width=1)
    draw.line((W//2, H//3, W//3, H//2), fill=(255,255,255), width=1)
    draw.line((W//2, H//3, W*2//3, H//2), fill=(255,255,255), width=1)
    draw.line((W//2, H*3//4, W//3, H), fill=(255,255,255), width=1)
    draw.line((W//2, H*3//4, W*2//3, H), fill=(255,255,255), width=1)
    im.save(path)


def test_pose_coverage_mask_shape_and_area(tmp_path):
    ctrl = tmp_path / 'ctrl.png'
    _make_synthetic_control(str(ctrl), size=(64, 64))

    base = make_control_mask(str(ctrl), target_size=(64, 64), threshold=0.01, dilate=1, blur=1, ref_max_dim=128)
    cov = make_pose_coverage_mask(str(ctrl), target_size=(64, 64), threshold=0.01, base_dilate=6, auto_scale_factor=2.0, blur=3, ref_max_dim=128)

    assert base.shape == cov.shape == (64, 64)

    area_base = (base > 0.01).float().sum().item()
    area_cov = (cov > 0.01).float().sum().item()

    # coverage mask should be larger or equal to base mask
    assert area_cov >= area_base

    # but not excessively large; for this tiny synthetic case expect less than 80% of image
    assert area_cov <= 64*64*0.85


def test_pose_coverage_mask_empty_fallback(tmp_path):
    # empty control should fall back to control mask behavior and not crash
    ctrl = tmp_path / 'ctrl_empty.png'
    Image.new('RGB', (64, 64), (0, 0, 0)).save(str(ctrl))

    cov = make_pose_coverage_mask(str(ctrl), target_size=(64, 64), threshold=0.01, base_dilate=6, auto_scale_factor=2.0, blur=3, ref_max_dim=128)
    assert cov.shape == (64, 64)
    assert cov.max() <= 1.0 and cov.min() >= 0.0


def test_pose_coverage_dense_control_fallback(tmp_path):
    # dense control (photo-like) should not produce an all-1 mask; we expect edge fallback
    ctrl = tmp_path / 'ctrl_dense.png'
    # make a gradient image which would be dense after activation
    im = Image.linear_gradient('L').resize((64,64)).convert('RGB')
    im.save(str(ctrl))

    cov = make_pose_coverage_mask(str(ctrl), target_size=(64, 64), threshold=0.01, base_dilate=6, auto_scale_factor=2.0, blur=3, ref_max_dim=128)
    assert cov.shape == (64, 64)
    area_cov = (cov > 0.01).float().sum().item()
    assert area_cov <= 64*64*0.95  # not 100%
    assert area_cov > 0  # still non-empty


def test_pose_coverage_respects_controls_dir(tmp_path):
    # If the user passes the base image, prefer a control file in _controls/ with the same basename
    base = tmp_path / 'img001.jpg'
    Image.new('RGB', (64,64), (128,128,128)).save(str(base))
    controls_dir = tmp_path / '_controls'
    controls_dir.mkdir()
    ctrl = controls_dir / 'img001.pose.png'
    # make a simple skeleton control
    im = Image.new('RGB', (64, 64), (0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.line((32, 12, 32, 52), fill=(255,255,255), width=1)
    im.save(str(ctrl))

    # call with base path; function should find control and produce similar mask to direct control
    cov_via_base = make_pose_coverage_mask(str(base), target_size=(64, 64), threshold=0.01, base_dilate=4, auto_scale_factor=1.0, blur=1, ref_max_dim=128)
    cov_via_ctrl = make_pose_coverage_mask(str(ctrl), target_size=(64, 64), threshold=0.01, base_dilate=4, auto_scale_factor=1.0, blur=1, ref_max_dim=128)

    area_base = (cov_via_base > 0.01).float().sum().item()
    area_ctrl = (cov_via_ctrl > 0.01).float().sum().item()

    assert cov_via_base.shape == cov_via_ctrl.shape == (64, 64)
    # Using control should not produce full-image mask, and base-based lookup should match control-based
    assert area_ctrl > 0 and area_ctrl < 64*64
    assert abs(area_base - area_ctrl) <= 200
