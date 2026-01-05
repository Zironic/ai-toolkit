import torch
from toolkit.losses import masked_mse, luminance_mask_from_images


def test_masked_mse_basic():
    b, c, h, w = 2, 3, 8, 8
    pred = torch.zeros((b, c, h, w))
    target = torch.zeros((b, c, h, w))
    mask = torch.zeros((b, 1, h, w))
    mask[:, :, 2:6, 2:6] = 1.0
    # set a difference in masked region
    pred[:, :, 3:5, 3:5] = 1.0
    loss = masked_mse(pred, target, mask)
    assert loss > 0


def test_luminance_mask_range():
    b, c, h, w = 1, 3, 16, 16
    target = torch.zeros((b, c, h, w))
    pred = target.clone()
    # add a bright patch
    pred[:, :, 6:10, 6:10] = 1.0
    mask = luminance_mask_from_images(pred, target, blur_kernel=3)
    assert mask.shape == (b, 1, h, w)
    assert mask.max() <= 1.0 and mask.min() >= 0.0
    # check mask highlights the patch
    assert mask[0, 0, 7, 7] > 0.1
