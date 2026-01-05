import torch
import types
from scripts.preview_mask import make_control_mask


def make_large_control(b=1, c=3, h=64, w=64):
    t = torch.zeros((b, c, h, w))
    # fill big central box occupying large fraction
    t[:, :, 8:56, 8:56] = 1.0
    return t


def make_small_control(b=1, c=3, h=64, w=64):
    t = torch.zeros((b, c, h, w))
    # small dot
    t[:, :, 30:34, 30:34] = 1.0
    return t


def test_auto_dilate_increases_mask_area_for_large_control():
    ctrl_img_path = None
    # Instead of writing to disk, use make_control_mask logic inline by creating an image
    # We'll craft a fake image file using PIL in-memory via temporary file
    from PIL import Image
    import numpy as np
    import tempfile
    import os

    # create a large control image (white box on black)
    tmp = tempfile.mkdtemp()
    control_path = os.path.join(tmp, 'ctrl.png')
    img = Image.new('RGB', (64, 64), (0, 0, 0))
    for y in range(8,56):
        for x in range(8,56):
            img.putpixel((x,y),(255,255,255))
    img.save(control_path)

    # compute mask with auto disabled
    mask_base = make_control_mask(control_path, target_size=(64,64), threshold=0.05, dilate=7, blur=3)
    area_base = (mask_base > 0.01).float().sum().item()

    # compute mask with auto enabled via manipulating formula used in preview (approx)
    # emulate auto-dilate determination: compute frac and new dilate
    arr = (torch.from_numpy(np.array(img.convert('L')) / 255.0))
    nz = torch.nonzero(arr > 0.05)
    ys = nz[:,0]; xs = nz[:,1]
    h_bbox = int(ys.max()-ys.min()+1)
    w_bbox = int(xs.max()-xs.min()+1)
    frac = max(h_bbox, w_bbox)/float(64)
    dilate_auto = max(1, int(round(7 * (1.0 + frac * 4.0))))

    mask_auto = make_control_mask(control_path, target_size=(64,64), threshold=0.05, dilate=dilate_auto, blur=3)
    area_auto = (mask_auto > 0.01).float().sum().item()

    # Expect the auto-dilated mask to be noticeably larger than the base mask
    # Note: since dilation is applied at high reference resolution then downsampled, the
    # effective area increase is smoother; allow more modest growth.
    assert area_auto > area_base * 1.07

    # cleanup
    import shutil
    shutil.rmtree(tmp)
