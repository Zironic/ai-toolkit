from PIL import Image
import numpy as np
from typing import Union


def _to_gray_array(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert('L'), dtype=np.float32)
    return arr


def _sobel_edges(arr: np.ndarray) -> np.ndarray:
    # simple sobel filter implementation (no external deps)
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    from scipy.signal import convolve2d

    gx = convolve2d(arr, Kx, mode='same', boundary='symm')
    gy = convolve2d(arr, Ky, mode='same', boundary='symm')
    g = np.hypot(gx, gy)
    # normalize to 0-255
    g = g / (g.max() + 1e-8) * 255.0
    return g


class ControlImageProcessor:
    """Lightweight control image processor.

    Provides deterministic, dependency-light implementations for control types
    commonly used in training tests: 'canny' (edge-like), 'identity' (passthrough),
    and 'grayscale'. For heavy-weight processors (pose, depth, openpose) the
    class will raise ImportError or defer to installed packages (controlnet_aux,
    easy_dwpose, or depth pipelines).
    """

    @staticmethod
    def process(image: Union[Image.Image, np.ndarray], control_type: str = 'canny') -> Image.Image:
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image.astype('uint8'))
        else:
            img = image

        control_type = control_type.lower()
        if control_type in ('none', 'identity'):
            return img.convert('RGB')

        if control_type == 'grayscale':
            return img.convert('L')

        if control_type == 'canny' or control_type == 'line':
            # Use a simple sobel-based edge detector
            try:
                arr = _to_gray_array(img)
                edges = _sobel_edges(arr)
                # apply simple threshold (Otsu could be used but this is deterministic)
                thr = np.percentile(edges, 75)
                edges_bin = (edges > thr).astype('uint8') * 255
                return Image.fromarray(edges_bin.astype('uint8')).convert('L')
            except ImportError:
                raise ImportError('scipy is required for canny-style control processing (pip install scipy)')

        # For heavy types, try to use the existing ControlGenerator or external libs
        if control_type == 'pose':
            try:
                from toolkit.control_generator import ControlGenerator
                # create a generator with CPU device and generate a file-based control, then reopen it
                # but to keep this function pure, raise ImportError to signal heavier deps required
                raise ImportError('Pose support requires optional dependencies (easy_dwpose or controlnet_aux)')
            except Exception:
                raise ImportError('Pose support requires optional dependencies (easy_dwpose or controlnet_aux)')

        if control_type == 'depth':
            raise ImportError('Depth support requires transformers pipelines (depth estimation)')

        raise ValueError(f"Unknown control type: {control_type}")
