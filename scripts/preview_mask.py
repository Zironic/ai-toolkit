import sys
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF


def make_control_mask(control_img_path, target_size, threshold=0.05, dilate=7, blur=9, ref_max_dim: int = 1024):
    """Create a control-derived mask by building it at a high reference resolution (ref_max_dim)
    and then downsampling to the requested target_size. This reduces blocky artifacts when
    controls are low-resolution by applying dilate/blur at higher resolution.
    """
    # Load control image as RGB tensor [1, C, H, W]
    ctrl = Image.open(control_img_path).convert('RGB')
    ctrl_t = TF.to_tensor(ctrl).unsqueeze(0)  # [1,C,H,W], values 0..1

    # Activation: abs sum across channels
    act = ctrl_t.abs().sum(dim=1, keepdim=True)  # [1,1,Hc,Wc]
    # Compute reference size based on ref_max_dim and the target size to preserve aspect ratio
    Ht, Wt = target_size
    ref_max = int(ref_max_dim) if ref_max_dim is not None else max(Ht, Wt)
    scale = float(ref_max) / float(max(Ht, Wt)) if max(Ht, Wt) > 0 else 1.0
    ref_H = max(1, int(round(Ht * scale)))
    ref_W = max(1, int(round(Wt * scale)))

    # Upsample to reference size, perform ops there, then downsample
    act = F.interpolate(act, size=(ref_H, ref_W), mode='bilinear', align_corners=False)
    # normalize per-sample
    max_per_sample = act.view(act.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
    act = act / (max_per_sample + 1e-9)
    # threshold
    mask0 = (act > threshold).float()
    # dilate via max pool
    d = dilate
    if d > 1:
        mask1 = F.max_pool2d(mask0, kernel_size=d, stride=1, padding=d // 2)
    else:
        mask1 = mask0
    # blur via avg pool
    b = blur
    if b > 1:
        mask_ref = F.avg_pool2d(mask1, kernel_size=b, stride=1, padding=b // 2)
    else:
        mask_ref = mask1

    # normalize and clamp at reference size
    max_per_sample = mask_ref.view(mask_ref.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
    mask_ref = mask_ref / (max_per_sample + 1e-9)
    mask_ref = mask_ref.clamp(0.0, 1.0)

    # Downsample to target size using area mode for good averaging
    mask = F.interpolate(mask_ref, size=(Ht, Wt), mode='area')
    mask = mask.clamp(0.0, 1.0)
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

    mask = make_control_mask(control_path, target_size=(H, W), threshold=threshold, dilate=dilate, blur=blur, ref_max_dim=ref_max_size)

    mask_path = os.path.join(out_dir, os.path.basename(target_path).rsplit('.', 1)[0] + '_mask.png')
    overlay_path = os.path.join(out_dir, os.path.basename(target_path).rsplit('.', 1)[0] + '_overlay.png')

    save_mask_png(mask, mask_path)
    overlay_mask_on_image(target_path, mask, overlay_path)

    print('Saved mask to', mask_path)
    print('Saved overlay to', overlay_path)


if __name__ == '__main__':
    main()
