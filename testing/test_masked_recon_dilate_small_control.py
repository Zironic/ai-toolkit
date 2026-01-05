import torch
from scripts.preview_mask import make_control_mask


def test_auto_dilate_small_control_no_excessive_growth():
    from PIL import Image
    import numpy as np
    import tempfile
    import os

    tmp = tempfile.mkdtemp()
    control_path = os.path.join(tmp, 'ctrl.png')
    img = Image.new('RGB', (64, 64), (0, 0, 0))
    # small dot
    for y in range(30,34):
        for x in range(30,34):
            img.putpixel((x,y),(255,255,255))
    img.save(control_path)

    mask_base = make_control_mask(control_path, target_size=(64,64), threshold=0.05, dilate=7, blur=3)
    area_base = (mask_base > 0.01).float().sum().item()

    # small bbox fraction -> dilate_auto should be close to base
    arr = (torch.from_numpy(np.array(img.convert('L')) / 255.0))
    nz = torch.nonzero(arr > 0.05)
    ys = nz[:,0]; xs = nz[:,1]
    h_bbox = int(ys.max()-ys.min()+1)
    w_bbox = int(xs.max()-xs.min()+1)
    frac = max(h_bbox, w_bbox)/float(64)
    dilate_auto = max(1, int(round(7 * (1.0 + frac * 4.0))))

    mask_auto = make_control_mask(control_path, target_size=(64,64), threshold=0.05, dilate=dilate_auto, blur=3)
    area_auto = (mask_auto > 0.01).float().sum().item()

    # Expect little or modest growth for small controls
    assert area_auto <= area_base * 1.5

    import shutil
    shutil.rmtree(tmp)
