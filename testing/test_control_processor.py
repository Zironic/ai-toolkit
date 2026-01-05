import numpy as np
from PIL import Image
from toolkit.control_processor import ControlImageProcessor


def test_canny_processor_returns_single_channel_image():
    # create a simple image with a white square on black background
    arr = np.zeros((128, 128, 3), dtype=np.uint8)
    arr[32:96, 32:96, :] = 255
    img = Image.fromarray(arr)

    out = ControlImageProcessor.process(img, 'canny')
    assert out.mode == 'L'
    out_arr = np.array(out)
    assert out_arr.dtype == np.uint8
    # edges should be present (non-zero)
    assert out_arr.max() > 0


def test_identity_returns_rgb():
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    out = ControlImageProcessor.process(img, 'identity')
    assert out.mode == 'RGB'
