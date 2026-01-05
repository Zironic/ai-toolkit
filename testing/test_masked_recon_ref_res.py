import os
import tempfile
import numpy as np
from PIL import Image
from scripts.preview_mask import make_control_mask


def _write_control_img(box_coords, size=(64,64)):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'ctrl.png')
    img = Image.new('RGB', size, (0, 0, 0))
    for y in range(box_coords[1], box_coords[3]):
        for x in range(box_coords[0], box_coords[2]):
            img.putpixel((x,y),(255,255,255))
    img.save(path)
    return path, tmp


def test_make_control_mask_downsamples_and_has_expected_shape():
    path, tmp = _write_control_img((8,8,56,56), size=(64,64))
    mask = make_control_mask(path, target_size=(64,64), threshold=0.01, dilate=7, blur=3, ref_max_dim=256)
    assert mask.shape == (64, 64)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    import shutil
    shutil.rmtree(tmp)


def test_ref_res_changes_mask_values_compared_to_native():
    path, tmp = _write_control_img((8,8,56,56), size=(64,64))
    mask_native = make_control_mask(path, target_size=(64,64), threshold=0.01, dilate=7, blur=3, ref_max_dim=64)
    mask_ref = make_control_mask(path, target_size=(64,64), threshold=0.01, dilate=7, blur=3, ref_max_dim=512)

    # masks should differ when the reference resolution changes
    assert np.abs(mask_ref.numpy() - mask_native.numpy()).sum() > 0.0

    import shutil
    shutil.rmtree(tmp)
