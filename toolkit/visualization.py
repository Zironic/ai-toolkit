import os
from PIL import Image
import numpy as np
import torch


def save_mask_preview(mask: torch.Tensor, save_path: str):
    """Save a single-channel mask tensor as a PNG to save_path.

    mask: torch.Tensor in shape (H, W) or (1, H, W), values in [0,1]
    save_path: full path to save PNG
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().float()
        if m.dim() == 3 and m.size(0) == 1:
            m = m[0]
        arr = (m.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    else:
        arr = (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(save_path)
