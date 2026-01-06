import sys
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF

# Optional OpenCV fallback for faster morphological ops
try:
    import cv2
except Exception:
    cv2 = None

# Ensure the repository root is on sys.path so local packages (e.g., `toolkit`) can
# be imported when running this script directly (python scripts/preview_mask.py ...)
try:
    import toolkit  # type: ignore
except Exception:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def make_control_mask(control_img_path, target_size, threshold=0.05, dilate=7, blur=9, ref_max_dim: int = 1024):
    """Delegate to the unified generator in `toolkit.masked_recon.generate_control_mask`.

    This ensures preview masks match training masks exactly and removes duplicated code.
    """
    from toolkit.masked_recon import generate_control_mask
    mask = generate_control_mask(control_img_path, base_img_path=None, target_size=target_size, threshold=threshold, base_dilate=dilate, blur=blur, ref_max_dim=ref_max_dim)
    return mask.squeeze(0).squeeze(0)  # [H,W]


    if cv2 is not None:
        # Cap dilation so kernel does not become larger than image
        r_px = min(r_px, int(max_dilate) if max_dilate is not None else r_px, max(ref_H, ref_W)//2)
        # Cap dilation so kernel does not become larger than image
        r_px = min(r_px, int(max_dilate) if max_dilate is not None else r_px, max(ref_H, ref_W)//2)
        # Use elliptical kernel for more natural coverage
        k = max(1, 2 * r_px + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(mask_np, kernel, iterations=1)
        # close small holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 3), max(3, k // 3)))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        if blur and blur > 0:
            bsz = max(1, int(blur) * 2 + 1)
            blurred = cv2.GaussianBlur(closed, (bsz, bsz), 0)
        else:
            blurred = closed
        mask_ref = torch.from_numpy(blurred.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    else:
        # Fallback: use torch max-pool as dilation
        # cap r_px similar to cv2 branch
        r_px = min(r_px, int(max_dilate) if max_dilate is not None else r_px, max(ref_H, ref_W)//2)
        d = max(1, 2 * r_px + 1)
        mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        mask_tensor = F.max_pool2d(mask_tensor, kernel_size=d, stride=1, padding=d // 2)
        # morphological close via dilation then erosion approximation
        mask_tensor = F.max_pool2d(mask_tensor, kernel_size=3, stride=1, padding=1)
        mask_ref = mask_tensor

    # normalize and clamp at reference size
    max_per_sample = mask_ref.view(mask_ref.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
    mask_ref = mask_ref / (max_per_sample + 1e-9)
    mask_ref = mask_ref.clamp(0.0, 1.0)

    # Downsample to target size using area mode for good averaging
    mask = F.interpolate(mask_ref, size=(Ht, Wt), mode='area')
    mask = mask.clamp(0.0, 1.0)
    return mask.squeeze(0).squeeze(0)  # [H,W]


def make_pose_coverage_mask(control_img_path, target_size, threshold=0.02, base_dilate=10, auto_scale_factor=3.0, blur=5, ref_max_dim: int = 1024, max_dilate: int = 200, base_img_path: str = None, use_edges: bool = False):
    """Wrapper that calls the unified generator in `toolkit.masked_recon` so preview masks
    match training masks exactly. `use_edges` opts into base-image edge refinement (disabled
    by default to preserve the pre-refactor behaviour)."""
    from toolkit.masked_recon import generate_control_mask
    mask = generate_control_mask(control_img_path, base_img_path=base_img_path, target_size=target_size, threshold=threshold, base_dilate=base_dilate, auto_scale_factor=auto_scale_factor, blur=blur, ref_max_dim=ref_max_dim, max_dilate=max_dilate, use_edges=use_edges)
    return mask.squeeze(0).squeeze(0)  # [H,W]


def save_mask_png(mask_tensor, path):
    arr = (mask_tensor.detach().cpu().numpy() * 255.0).astype(np.uint8)
    im = Image.fromarray(arr, mode='L')
    im.save(path)


def overlay_mask_on_image(image_path, mask_tensor, out_path, color=(255, 0, 0), alpha=0.6):
    base = Image.open(image_path).convert('RGBA')
    arr = (mask_tensor.detach().cpu().numpy() * 255.0).astype(np.uint8)
    mask_im = Image.fromarray(arr, mode='L')
    # create color overlay
    overlay = Image.new('RGBA', base.size, color + (0,))
    # use mask as alpha scaled by alpha
    alpha_mask = (arr.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    alpha_im = Image.fromarray(alpha_mask, mode='L')
    overlay.putalpha(alpha_im)
    out = Image.alpha_composite(base, overlay)
    out.save(out_path)


def main():
    if len(sys.argv) < 3:
        print('Usage: python preview_mask.py <control_image> <target_image> [out_dir]')
        sys.exit(1)
    control_path = sys.argv[1]
    target_path = sys.argv[2]
    out_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.join('output', 'mask_previews')
    os.makedirs(out_dir, exist_ok=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.05)
    parser.add_argument('--dilate', type=int, default=7)
    parser.add_argument('--blur', type=int, default=9)
    parser.add_argument('--auto-dilate', action='store_true')
    parser.add_argument('--dilate-scale-factor', type=float, default=4.0)
    parser.add_argument('--ref-max-size', type=int, default=1024)
    parser.add_argument('--pose-coverage', action='store_true', help='Generate a pose-coverage mask (thickened skeleton + closing)')
    parser.add_argument('--use-edges', action='store_true', help='Allow edge-based refinement using the base image (opt-in)')
    args = parser.parse_args(sys.argv[4:])

    ref_max_size = args.ref_max_size

    # ... later when computing mask0 and mask, pass ref_max_size into make_control_mask calls

    tgt = Image.open(target_path).convert('RGB')
    W, H = tgt.size

    threshold = args.threshold
    dilate = args.dilate
    blur = args.blur

    if args.auto_dilate:
        # compute mask0 first at threshold to get bbox fraction then compute dilate per formula
        mask0 = make_control_mask(control_path, target_size=(H, W), threshold=threshold, dilate=1, blur=1, ref_max_dim=ref_max_size)
        nz = np.argwhere(mask0.numpy() > 0.0)
        if nz.size == 0:
            frac = 0.0
        else:
            ys = nz[:, 0]
            xs = nz[:, 1]
            h_bbox = int(ys.max() - ys.min() + 1)
            w_bbox = int(xs.max() - xs.min() + 1)
            frac = max(h_bbox, w_bbox) / float(max(H, W))
        dilate = max(1, int(round(dilate * (1.0 + frac * args.dilate_scale_factor))))

    if args.pose_coverage:
        # If user supplied the base image (not the skeleton control), check for a control
        # image in a sibling `_controls/` directory named `<basename>.pose.jpg` or `.pose.png`.
        base_img = None
        cp = control_path
        dirname = os.path.dirname(cp)
        base_name = os.path.basename(cp).rsplit('.', 1)[0]
        controls_dir = os.path.join(dirname, '_controls')
        found = None
        for ext in ('.pose.jpg', '.pose.png', '.jpg', '.png'):
            cand = os.path.join(controls_dir, base_name + ext)
            if os.path.exists(cand):
                found = cand
                break
        if found is not None:
            # use control from _controls and remember base for edge fallback
            base_img = control_path
            control_for_mask = found
        else:
            control_for_mask = control_path
            base_img = None
            mask = make_pose_coverage_mask(control_for_mask, target_size=(H, W), threshold=threshold, base_dilate=dilate, blur=blur, auto_scale_factor=args.dilate_scale_factor, ref_max_dim=ref_max_size, base_img_path=base_img, use_edges=args.use_edges)
