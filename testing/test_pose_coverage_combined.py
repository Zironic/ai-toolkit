from PIL import Image, ImageDraw
import numpy as np
from scripts.preview_mask import make_pose_coverage_mask


def test_pose_coverage_uses_base_edges(tmp_path):
    # base image with a clear black rectangle on white background (the subject)
    base = tmp_path / 'img.jpg'
    W, H = 64, 64
    im = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    # rectangle region that's the subject
    rect = (12, 12, 52, 50)
    draw.rectangle(rect, fill=(0, 0, 0))
    im.save(str(base))

    # control image: simple tiny skeleton in center of rectangle
    controls_dir = tmp_path / '_controls'
    controls_dir.mkdir()
    ctrl = controls_dir / 'img.pose.png'
    c = Image.new('RGB', (W, H), (0, 0, 0))
    cd = ImageDraw.Draw(c)
    cd.ellipse((30, 26, 34, 30), fill=(255, 255, 255))
    c.save(str(ctrl))

    # generate mask using both control and base
    # explicitly enable edge-based refinement for this test
    mask = make_pose_coverage_mask(str(ctrl), target_size=(H, W), threshold=0.01, base_dilate=6, auto_scale_factor=1.0, blur=1, ref_max_dim=128, base_img_path=str(base), use_edges=True)

    # mask area should be close to rectangle area
    area = (mask > 0.1).sum().item()
    rect_area = (rect[2]-rect[0]+1) * (rect[3]-rect[1]+1)

    assert area > rect_area * 0.7
    assert area < rect_area * 1.3
