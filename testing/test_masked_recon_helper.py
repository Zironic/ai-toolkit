import torch
from types import SimpleNamespace
from toolkit.masked_recon import build_control_mask


class DummyConfig:
    def __init__(self):
        self.masked_recon_control_ref_max_size = 1024
        self.masked_recon_control_threshold = 0.05
        self.masked_recon_control_dilate = 3
        self.masked_recon_control_dilate_auto = True
        self.masked_recon_control_dilate_scale_factor = 2.0
        self.masked_recon_control_blur = 3


def test_build_control_mask_basic():
    cfg = DummyConfig()
    ctrl = torch.zeros((1, 3, 32, 32))
    # draw a central square
    ctrl[0, :, 8:24, 8:24] = 1.0
    mask = build_control_mask(ctrl, cfg, target_size=(16, 16), device_torch=ctrl.device)
    assert mask is not None
    assert mask.shape == (1, 1, 16, 16)
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    assert mask.sum() > 0


def test_build_control_mask_list_input():
    cfg = DummyConfig()
    a = torch.zeros((3, 16, 16))
    a[:, 4:12, 4:12] = 1.0
    b = a.clone()
    mask = build_control_mask([a, b], cfg, target_size=(8, 8), device_torch=a.device)
    assert mask is not None
    assert mask.shape == (2, 1, 8, 8)
    assert mask.sum() > 0