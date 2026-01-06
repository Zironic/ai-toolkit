import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import numpy as np
import torch
from toolkit.masked_recon import generate_control_mask

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('control')
    p.add_argument('--base', default=None)
    p.add_argument('--size', type=int, nargs=2, default=None)
    p.add_argument('--threshold', type=float, default=0.02)
    p.add_argument('--dilate', type=int, default=10)
    p.add_argument('--blur', type=int, default=5)
    args = p.parse_args()

    target = None
    if args.size is None:
        # try to infer size from base or control
        img_path = args.base if args.base else args.control
        im = Image.open(img_path).convert('RGB')
        W, H = im.size
        target = (H, W)
    else:
        target = tuple(args.size)

    # Also print the activation / skeleton (mask0) at reference size to see where the generator
    # starts from. This duplicates a few ops from generate_control_mask so we can inspect intermediate state.
    from PIL import Image as PILImage
    import torchvision.transforms.functional as TF
    import torch.nn.functional as F

    ctrl_pil = PILImage.open(args.control).convert('RGB')
    # match default ref_max used by preview when ref-max-size isn't provided (1024)
    ref_max = 1024
    Ht, Wt = target
    scale = float(ref_max) / float(max(Ht, Wt)) if max(Ht, Wt) > 0 else 1.0
    ref_H = max(1, int(round(Ht * scale)))
    ref_W = max(1, int(round(Wt * scale)))
    ctrl_t = TF.to_tensor(ctrl_pil).unsqueeze(0)
    act = ctrl_t.abs().sum(dim=1, keepdim=True)
    act = F.interpolate(act, size=(ref_H, ref_W), mode='bilinear', align_corners=False)
    max_per_sample = act.view(act.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
    act = act / (max_per_sample + 1e-9)
    mask0 = (act > args.threshold).float()
    mask0_np = (mask0[0,0].cpu().numpy() * 255.0).astype('uint8')
    m0_nz = int(np.count_nonzero(mask0_np))
    m0_frac = float(m0_nz) / float(mask0_np.size)
    if m0_nz > 0:
        ys0, xs0 = np.nonzero(mask0_np)
        m0_bbox = (int(xs0.min()), int(ys0.min()), int(xs0.max()), int(ys0.max()))
    else:
        m0_bbox = None
    print(f'[debug] mask0 nonzero: {m0_nz}, frac: {m0_frac:.4f}, mask0_bbox: {m0_bbox}')
    # save mask0 view for inspection
    Image.fromarray(mask0_np, mode='L').save('tmp_debug_mask0.png')
    print('wrote tmp_debug_mask0.png')

    mask = generate_control_mask(args.control, base_img_path=args.base, target_size=target, threshold=args.threshold, base_dilate=args.dilate, blur=args.blur)
    # mask is [B,1,H,W] torch tensor
    m = mask.squeeze(0).squeeze(0).cpu().numpy()

    nz = np.argwhere(m > 0.05)
    if nz.size == 0:
        print('MASK EMPTY')
        sys.exit(0)
    ys = nz[:,0]
    xs = nz[:,1]
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    area = nz.shape[0]
    H, W = m.shape
    # compute centroid of mask
    cy = int(np.round(ys.mean()))
    cx = int(np.round(xs.mean()))
    print(f'mask shape: {m.shape}, nonzero area: {area}, bbox (xmin,ymin,xmax,ymax): {bbox}, centroid: (x={cx}, y={cy}), area_frac: {area / float(H*W):.3f}')
    # Save mask for inspection
    out = (m * 255.0).astype('uint8')
    Image.fromarray(out, mode='L').save('tmp_debug_mask.png')
    print('wrote tmp_debug_mask.png')
