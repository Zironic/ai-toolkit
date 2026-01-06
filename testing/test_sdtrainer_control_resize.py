import torch
from PIL import Image
from toolkit.buckets import get_bucket_for_image_size
from extensions_built_in.sd_trainer.SDTrainer import _resize_batch_to_bucket


def make_dummy_batch(h, w, c=3):
    # create a batch tensor [1, C, H, W]
    t = torch.randint(0, 255, (1, c, h, w), dtype=torch.uint8)
    return t


def test_resize_non_square_control_to_bucket():
    # example from logs: orig 1285x407 (H x W)
    H, W = 1285, 407
    size = 512
    batch = make_dummy_batch(H, W)
    batch_resized, used_dataset_control, meta = _resize_batch_to_bucket(batch, size, full_size_control_images=False)
    assert not used_dataset_control
    # verify that final resized matches bucket target
    bucket = get_bucket_for_image_size(W, H, resolution=size)
    target_w, target_h = bucket['width'], bucket['height']
    _, C, H2, W2 = batch_resized.shape
    assert (H2, W2) == (target_h, target_w)


def test_resize_full_size_preserves_and_pads():
    # full-size path where long side equals size, but not multip of 16 -> will be padded
    H, W = 300, 400
    size = 400
    batch = make_dummy_batch(H, W)
    batch_resized, used_dataset_control, meta = _resize_batch_to_bucket(batch, size, full_size_control_images=True)
    # used_dataset_control True only when sizes already match and are mult of 16
    assert not used_dataset_control
    # resized should be padded to multiple of 16
    _, C, H2, W2 = batch_resized.shape
    assert H2 % 16 == 0 and W2 % 16 == 0
