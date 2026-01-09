import os
import torch
from toolkit.print import print_acc

from toolkit.losses import masked_mse, luminance_mask_from_images

# Image / op helpers used by mask generation
from PIL import Image
import numpy as np
from torchvision.transforms import functional as TF
import torch.nn.functional as F
try:
    import cv2
except Exception:
    cv2 = None



def build_control_mask(ctrl, train_config, target_size, device_torch):
    """Build a processed mask from control tensor(s) that matches the masked-recon pipeline.

    Args:
        ctrl: Tensor (B,C,H,W) or (C,H,W) or list of such Tensors
        train_config: config object with masked_recon_* tuning params
        target_size: (H, W) desired output spatial size
        device_torch: torch.device or device string

    Returns:
        mask tensor [B,1,H,W] with values in [0,1] or None on failure
    """
    if ctrl is None:
        return None
    # Normalize and coerce to tensor batch [B,C,H,W]
    if isinstance(ctrl, list):
        if all(isinstance(x, torch.Tensor) for x in ctrl):
            normalized = []
            for x in ctrl:
                if x.dim() == 3:
                    normalized.append(x.unsqueeze(0))
                else:
                    normalized.append(x)
            try:
                ctrl_tensor = torch.cat(normalized, dim=0).to(device_torch)
            except Exception:
                return None
        else:
            return None
    elif isinstance(ctrl, torch.Tensor):
        ctrl_tensor = ctrl.to(device_torch)
        if ctrl_tensor.dim() == 3:
            ctrl_tensor = ctrl_tensor.unsqueeze(0)
    else:
        return None

    try:
        # Delegate to unified generator so preview and training masks match exactly
        Ht, Wt = int(target_size[0]), int(target_size[1])
        t = float(getattr(train_config, 'masked_recon_control_threshold', 0.05))
        base_d = int(getattr(train_config, 'masked_recon_control_dilate', 7))
        do_auto = bool(getattr(train_config, 'masked_recon_control_dilate_auto', True))
        scale_factor = float(getattr(train_config, 'masked_recon_control_dilate_scale_factor', 4.0))
        b = int(getattr(train_config, 'masked_recon_control_blur', 9))
        ref_max = int(getattr(train_config, 'masked_recon_control_ref_max_size', max(Ht, Wt)))

        mask = generate_control_mask(ctrl_tensor, base_img_path=None, target_size=(Ht, Wt), threshold=t, base_dilate=base_d, auto_scale_factor=scale_factor, blur=b, ref_max_dim=ref_max)
        return mask
    except Exception:
        return None


def generate_control_mask(control, base_img_path=None, target_size=(256,256), threshold=0.05, base_dilate=7, auto_scale_factor=4.0, blur=9, ref_max_dim: int = 1024, max_dilate: int = 200, device: str = None, use_edges: bool = False):
    """Unified control mask generator used by preview and training.

    Args:
        control: torch Tensor (B,C,H,W) or (C,H,W) or single-file path (string) or PIL Image
        base_img_path: optional path to base image (used for edge extraction)
        target_size: (H,W) tuple for output size
        use_edges: if True, attempt to use base image edges and connected-component constrained
                   fill to refine mask coverage; when False, use simple dilation/blur of skeleton
                   (pre-refactor behavior).
        other params tune the behaviour
    Returns:
        torch.Tensor of shape [B,1,H,W] with values in [0,1]
    """
    """Unified control mask generator used by preview and training.

    Args:
        control: torch Tensor (B,C,H,W) or (C,H,W) or single-file path (string) or PIL Image
        base_img_path: optional path to base image (used for edge extraction)
        target_size: (H,W) tuple for output size
        other params tune the behaviour
    Returns:
        torch.Tensor of shape [B,1,H,W] with values in [0,1]
    """
    # If a file path to a base image was provided (not in `_controls`), attempt to find
    # a matching control image in a sibling `_controls/` folder. If found, use that control
    # image and set base_img_path to the original path so edge extraction has access.
    if isinstance(control, str) and not os.path.basename(os.path.dirname(control)).startswith('_controls'):
        dirname = os.path.dirname(control)
        base_name = os.path.basename(control).rsplit('.', 1)[0]
        controls_dir = os.path.join(dirname, '_controls')
        found = None
        for ext in ('.pose.jpg', '.pose.png', '.jpg', '.png'):
            cand = os.path.join(controls_dir, base_name + ext)
            if os.path.exists(cand):
                found = cand
                break
        if found is not None:
            base_img_path = control
            control = found

    # normalize control input into a batch tensor on CPU (we do CPU numpy ops for image processing)
    is_tensor = isinstance(control, torch.Tensor)
    batch = None
    if is_tensor:
        ctrl_tensor = control
        if ctrl_tensor.dim() == 3:
            ctrl_tensor = ctrl_tensor.unsqueeze(0)
        if ctrl_tensor.dim() != 4:
            raise ValueError('control tensor must be 3 or 4 dims')
        batch = ctrl_tensor.detach().cpu()
    else:
        # control is a path or PIL image
        if isinstance(control, str):
            ctrl_pil = Image.open(control).convert('RGB')
        elif isinstance(control, Image.Image):
            ctrl_pil = control.convert('RGB')
        else:
            raise ValueError('control must be tensor, path, or PIL Image')
        ctrl_t = TF.to_tensor(ctrl_pil).unsqueeze(0)
        batch = ctrl_t.detach().cpu()

    # per-sample processing
    out_masks = []
    Bs = batch.shape[0]
    Ht, Wt = int(target_size[0]), int(target_size[1])
    ref_max = int(ref_max_dim) if ref_max_dim is not None else max(Ht, Wt)
    scale = float(ref_max) / float(max(Ht, Wt)) if max(Ht, Wt) > 0 else 1.0
    ref_H = max(1, int(round(Ht * scale)))
    ref_W = max(1, int(round(Wt * scale)))

    for bi in range(Bs):
        ctrl_i = batch[bi:bi+1]
        act = ctrl_i.abs().sum(dim=1, keepdim=True)  # [1,1,Hc,Wc]
        act = F.interpolate(act, size=(ref_H, ref_W), mode='bilinear', align_corners=False)
        max_per_sample = act.view(act.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
        act = act / (max_per_sample + 1e-9)
        mask0 = (act > threshold).float()

        # bbox and r_px
        nz = (mask0[0, 0] > 0.0).cpu().numpy().astype('uint8')
        ys, xs = np.nonzero(nz)
        if ys.size == 0:
            # fallback to simple control mask behaviour
            # small dilate and blur
            d = max(1, base_dilate)
            if d > 1:
                mask1 = F.max_pool2d(mask0, kernel_size=d, stride=1, padding=d // 2)
            else:
                mask1 = mask0
            if blur and blur > 1:
                bsz = blur
                mask_ref = F.avg_pool2d(mask1, kernel_size=bsz, stride=1, padding=bsz // 2)
            else:
                mask_ref = mask1
            max_per_sample = mask_ref.view(mask_ref.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
            mask_ref = mask_ref / (max_per_sample + 1e-9)
            mask_ref = mask_ref.clamp(0.0, 1.0)
            mask = F.interpolate(mask_ref, size=(Ht, Wt), mode='area')
            out_masks.append(mask)
            continue

        h_bbox = int(ys.max() - ys.min() + 1)
        w_bbox = int(xs.max() - xs.min() + 1)
        frac = max(h_bbox, w_bbox) / float(max(ref_H, ref_W))
        r_px = int(round(base_dilate * (1.0 + frac * auto_scale_factor)))
        if max_dilate is not None:
            r_px = min(r_px, int(max_dilate))
        # conservative cap relative to reference size to avoid runaway dilation on small images
        r_px = min(r_px, max(1, max(ref_H, ref_W) // 8))
        if r_px < 1:
            r_px = 1

        # initial mask_np
        mask_np = (mask0[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
        nonzero_frac = float(np.count_nonzero(mask_np)) / float(mask_np.size)

        # If edges disabled and control appears dense, construct a small seed from top-percentile
        # activations to avoid the whole-image mask blowup.
        if not use_edges and nonzero_frac > 0.35:
            arr_act = act[0, 0].cpu().numpy()
            seed_small = None
            for pct in (99, 98, 95, 90):
                p = np.percentile(arr_act, pct)
                s = (arr_act >= p).astype(np.uint8) * 255
                if np.count_nonzero(s) > 0:
                    seed_small = s
                    break
            if seed_small is None:
                # last resort erosion until small
                m = mask_np.copy()
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) if cv2 is not None else None
                for it in range(10):
                    if cv2 is not None:
                        m2 = cv2.erode(m, kern, iterations=1)
                    else:
                        mt = torch.from_numpy(m.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0)
                        mt = F.max_pool2d(mt, kernel_size=3, stride=1, padding=1)
                        m2 = (mt[0,0].cpu().numpy() * 255.0).astype(np.uint8)
                    if np.count_nonzero(m2) == 0:
                        break
                    m = m2
                    frac_now = float(np.count_nonzero(m)) / float(m.size)
                    if frac_now < 0.02:
                        break
                seed_small = m
            if np.count_nonzero(seed_small) > 0:
                mask_np = seed_small
                r_px = max(1, int(base_dilate // 2))

        if not use_edges:
            # short-circuit: simple dilation+blur path (pre-refactor behavior)
            if cv2 is not None:
                r_px = min(r_px, int(max_dilate) if max_dilate is not None else r_px, max(ref_H, ref_W)//2)
                k = max(1, 2 * r_px + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                dilated = cv2.dilate(mask_np, kernel, iterations=1)
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 3), max(3, k // 3)))
                closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=1)
                if blur and blur > 0:
                    bsz = max(1, int(blur) * 2 + 1)
                    blurred = cv2.GaussianBlur(closed, (bsz, bsz), 0)
                else:
                    blurred = closed
                mask_ref = torch.from_numpy(blurred.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            else:
                r_px = min(r_px, int(max_dilate) if max_dilate is not None else r_px, max(ref_H, ref_W)//2)
                d = max(1, 2 * r_px + 1)
                mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                mask_tensor = F.max_pool2d(mask_tensor, kernel_size=d, stride=1, padding=d // 2)
                mask_tensor = F.max_pool2d(mask_tensor, kernel_size=3, stride=1, padding=1)
                mask_ref = mask_tensor

            max_per_sample = mask_ref.view(mask_ref.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
            mask_ref = mask_ref / (max_per_sample + 1e-9)
            mask_ref = mask_ref.clamp(0.0, 1.0)
            mask = F.interpolate(mask_ref, size=(Ht, Wt), mode='area')
            mask = mask.clamp(0.0, 1.0)
            out_masks.append(mask)
            continue

        # compute edges from base image if supplied, else from control image
        edges = None
        g = None
        if base_img_path is not None and os.path.exists(base_img_path):
            img_for_edges = Image.open(base_img_path).convert('L')
            img_res = img_for_edges.resize((ref_W, ref_H), resample=Image.BILINEAR)
            g = np.array(img_res)
            if cv2 is not None:
                med = np.median(g)
                lower = max(1, int(0.66 * med))
                upper = min(255, int(1.33 * med))
                edges = cv2.Canny(g, lower, upper)
        # if edges still None or control is dense, fallback to control-based edges or thinning
        if edges is None and nonzero_frac > 0.35:
            ctrl_pil = TF.to_pil_image(ctrl_i.squeeze(0))
            img_res = ctrl_pil.resize((ref_W, ref_H), resample=Image.BILINEAR)
            g = np.array(img_res)
            if cv2 is not None:
                med = np.median(g)
                lower = max(1, int(0.66 * med))
                upper = min(255, int(1.33 * med))
                edges = cv2.Canny(g, lower, upper)
            if edges is None or np.count_nonzero(edges) == 0:
                # gradient-based fallback
                arrf = g.astype(np.float32)
                gx = np.abs(np.gradient(arrf, axis=1))
                gy = np.abs(np.gradient(arrf, axis=0))
                grad = gx + gy
                for pct in (90,85,75,60,40):
                    th = np.percentile(grad, pct)
                    edges_try = (grad > th).astype(np.uint8) * 255
                    if np.count_nonzero(edges_try) > 0:
                        edges = edges_try
                        break
                # last-resort: iterative erosion thinning
                if edges is None or np.count_nonzero(edges) == 0:
                    m = mask_np.copy()
                    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) if cv2 is not None else None
                    for it in range(10):
                        if cv2 is not None:
                            m2 = cv2.erode(m, kern, iterations=1)
                        else:
                            mt = torch.from_numpy(m.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0)
                            mt = F.max_pool2d(mt, kernel_size=3, stride=1, padding=1)
                            m2 = (mt[0,0].cpu().numpy() * 255.0).astype(np.uint8)
                        if np.count_nonzero(m2) == 0:
                            break
                        m = m2
                        frac_now = float(np.count_nonzero(m)) / float(m.size)
                        if frac_now < 0.02:
                            break
                    if np.count_nonzero(m) > 0:
                        mask_np = m
                        r_px = min(r_px, max(3, base_dilate * 2, max(ref_H, ref_W)//8))

        # If edges present, compute edges_bin and use connected component constrained fill
        if edges is not None and np.count_nonzero(edges) > 0:
            # ensure edges is single-channel 2D
            if edges.ndim == 3:
                edges_2 = edges[...,0]
            else:
                edges_2 = edges
            edges_bin = (edges_2 > 0).astype(np.uint8) * 255
            ys_e, xs_e = np.nonzero(edges_bin)
            if ys_e.size > 0:
                h_be = int(ys_e.max() - ys_e.min() + 1)
                w_be = int(xs_e.max() - xs_e.min() + 1)
                frac_edges = max(h_be, w_be) / float(max(ref_H, ref_W))
                r_px = int(round(base_dilate * (1.0 + frac_edges * auto_scale_factor)))
                # cap r_px relative to both reference and edge bbox to avoid over-expansion
                r_px = min(r_px, max(1, max(ref_H, ref_W) // 8, max(h_be, w_be) // 6))
            # dilate edges to close small gaps
            k_close = max(3, int(round(r_px / 6)) * 2 + 1)
            edges_dil = cv2.dilate(edges_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)), iterations=1) if cv2 is not None else edges_bin
            inverted = (edges_dil == 0).astype(np.uint8) * 255
            # prepare skeleton seed
            skeleton_np = (mask0[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
            seed_k = 3
            seed = cv2.dilate(skeleton_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_k, seed_k)), iterations=1) if cv2 is not None else skeleton_np

            # Use connectedComponents and pick the component that best overlaps the seed
            try:
                num, labels = cv2.connectedComponents(inverted, connectivity=8)
            except Exception:
                num, labels = 0, None

            ys_s, xs_s = np.nonzero(seed)
            if ys_s.size == 0 or labels is None:
                mask_np = mask0[0, 0].cpu().numpy().astype(np.uint8) * 255
            else:
                cy = int(np.round(ys_s.mean()))
                cx = int(np.round(xs_s.mean()))
                if not (0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]):
                    mask_np = mask0[0, 0].cpu().numpy().astype(np.uint8) * 255
                else:
                    # Choose the component that has the greatest overlap with the skeleton seed.
                    overlap_counts = {}
                    for y, x in zip(ys_s, xs_s):
                        l = int(labels[y, x])
                        if l != 0:
                            overlap_counts[l] = overlap_counts.get(l, 0) + 1

                    # compute areas for labeled components
                    areas = {}
                    for lab in range(1, int(num) if num is not None else 1):
                        areas[lab] = int((labels == lab).sum())

                    if len(overlap_counts) > 0:
                        # choose the component with the highest overlap fraction (overlap / component_area)
                        # tie-break by absolute overlap count
                        def score_for_label(l):
                            a = max(1, areas.get(l, 1))
                            return (overlap_counts.get(l, 0) / float(a), overlap_counts.get(l, 0))
                        best_label = max(overlap_counts.keys(), key=score_for_label)
                        mask_np = (labels == best_label).astype(np.uint8) * 255
                    else:
                        # if seed doesn't overlap any labeled component, pick the largest non-background component
                        if len(areas) > 0:
                            best_label = max(areas.keys(), key=lambda k: areas[k])
                            mask_np = (labels == best_label).astype(np.uint8) * 255
                        else:
                            mask_np = mask0[0, 0].cpu().numpy().astype(np.uint8) * 255
            # fallback if component tiny vs allowable area
            area_mask = np.count_nonzero(mask_np)
            area_inv = np.count_nonzero(inverted) if 'inverted' in locals() else 0
            if area_inv > 0 and area_mask < max(10, int(area_inv * 0.25)):
                # keep the component but expand a bit to be less tiny
                if cv2 is not None:
                    mask_np = cv2.dilate(mask_np, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, seed_k), max(3, seed_k))), iterations=1)
                else:
                    mt = torch.from_numpy((mask_np.astype(np.float32)/255.0)).unsqueeze(0).unsqueeze(0)
                    mt = F.max_pool2d(mt, kernel_size=3, stride=1, padding=1)
                    mask_np = (mt[0,0].cpu().numpy() * 255.0).astype(np.uint8)

        # safeguard: if mask currently covers almost entire image and we don't have base image to constrain it,
        # fall back to a conservative small dilation of the original activation mask to avoid full-image masks.
        area_frac_now = float(np.count_nonzero(mask_np)) / float(mask_np.size)
        if area_frac_now > 0.95 and (base_img_path is None or not os.path.exists(base_img_path)):
            # dense control fallback: construct a small seed from the strongest activations (top-percentile)
            arr_act = act[0, 0].cpu().numpy()
            for pct in (99, 98, 95):
                p = np.percentile(arr_act, pct)
                seed_small = (arr_act >= p).astype(np.uint8) * 255
                if np.count_nonzero(seed_small) > 0:
                    break
            if np.count_nonzero(seed_small) == 0:
                # as a last resort, erode the original mask until small
                m = mask_np.copy()
                kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)) if cv2 is not None else None
                for it in range(10):
                    if cv2 is not None:
                        m2 = cv2.erode(m, kern, iterations=1)
                    else:
                        mt = torch.from_numpy(m.astype(np.float32)/255.0).unsqueeze(0).unsqueeze(0)
                        mt = F.max_pool2d(mt, kernel_size=3, stride=1, padding=1)
                        m2 = (mt[0,0].cpu().numpy() * 255.0).astype(np.uint8)
                    if np.count_nonzero(m2) == 0:
                        break
                    m = m2
                    frac_now = float(np.count_nonzero(m)) / float(m.size)
                    if frac_now < 0.02:
                        break
                seed_small = m
            mask_np = seed_small

            # use a much smaller dilation radius for dense fallback to avoid full-image growth
            r_px = max(1, int(max(1, base_dilate // 2)))

        # final dilation and blur using cv2 or torch fallback
        if cv2 is not None:
            r_px = min(r_px, int(max_dilate) if max_dilate is not None else r_px, max(ref_H, ref_W)//2)
            k = max(1, 2 * r_px + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dilated = cv2.dilate(mask_np, kernel, iterations=1)
            # if we computed an inverted allowable area (from edges), constrain dilation to it
            if 'inverted' in locals():
                dilated = cv2.bitwise_and(dilated, inverted)
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, k // 3), max(3, k // 3)))
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            if blur and blur > 0:
                bsz = max(1, int(blur) * 2 + 1)
                blurred = cv2.GaussianBlur(closed, (bsz, bsz), 0)
            else:
                blurred = closed
            mask_ref = torch.from_numpy(blurred.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        else:
            r_px = min(r_px, int(max_dilate) if max_dilate is not None else r_px, max(ref_H, ref_W)//2)
            d = max(1, 2 * r_px + 1)
            mask_tensor = torch.from_numpy(mask_np.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
            mask_tensor = F.max_pool2d(mask_tensor, kernel_size=d, stride=1, padding=d // 2)
            # constrain to inverted (if present) by multiplying by its mask
            if 'inverted' in locals():
                inv_t = torch.from_numpy((inverted.astype(np.float32)/255.0)).unsqueeze(0).unsqueeze(0)
                mask_tensor = mask_tensor * inv_t
            mask_tensor = F.max_pool2d(mask_tensor, kernel_size=3, stride=1, padding=1)
            mask_ref = mask_tensor

        max_per_sample = mask_ref.view(mask_ref.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
        mask_ref = mask_ref / (max_per_sample + 1e-9)
        mask_ref = mask_ref.clamp(0.0, 1.0)
        mask = F.interpolate(mask_ref, size=(Ht, Wt), mode='area')
        mask = mask.clamp(0.0, 1.0)
        out_masks.append(mask)

    out = torch.cat(out_masks, dim=0)
    return out


def apply_masked_recon_loss(current_loss, train_config, sd, noisy_latents, imgs, batch, dtype, device_torch):
    """Compute masked reconstruction loss and add to current_loss.
    Returns (loss, mloss_tensor_or_None)
    This function mirrors the logic originally in SDTrainer._compute_and_apply_masked_recon_loss.
    """
    if not getattr(train_config, 'masked_recon_weight', 0.0):
        return current_loss, None

    # choose target image
    target_img = None
    if getattr(batch, 'unaugmented_tensor', None) is not None:
        target_img = batch.unaugmented_tensor
    elif imgs is not None:
        target_img = imgs
    else:
        return current_loss, None

    target_img = target_img.to(device_torch)

    pred_img = None
    # Ensure we have a VAE to decode latents; raise on decode errors so callers can log and handle them.
    if not hasattr(sd, 'vae') or sd.vae is None:
        return current_loss, None

    try:
        vae_device = next(sd.vae.parameters(), torch.tensor(0)).device if hasattr(sd.vae, 'parameters') else device_torch
        decoded = sd.vae.decode(noisy_latents.to(vae_device))
        pred_img = decoded.sample if hasattr(decoded, 'sample') else decoded
        pred_img = pred_img.to(device_torch)
        if pred_img.min() < 0.0:
            pred_img = (pred_img + 1.0) / 2.0
        try:
            tgt_c = target_img.shape[1]
            if pred_img.shape[1] != tgt_c:
                if pred_img.shape[1] > tgt_c:
                    pred_img = pred_img[:, :tgt_c, :, :].contiguous()
                else:
                    reps = (tgt_c + pred_img.shape[1] - 1) // pred_img.shape[1]
                    pred_img = pred_img.repeat(1, reps, 1, 1)[:, :tgt_c, :, :].contiguous()
        except Exception:
            pass
    except Exception as e:
        # Propagate error so the caller (trainer) can catch and log it explicitly
        raise RuntimeError(f"Masked recon VAE decode or processing failed: {e}") from e

    if target_img.min() < 0.0:
        target_img = (target_img + 1.0) / 2.0

    mtype = getattr(train_config, 'masked_recon_type', 'illum')
    mask = None
    # Generate mask according to selected mode. Let internal errors propagate so callers can log them.
    if mtype == 'illum':
        mask = luminance_mask_from_images(pred_img, target_img, blur_kernel=9)
    elif mtype == 'edge':
        lum = 0.299 * target_img[:, 0:1] + 0.587 * target_img[:, 1:2] + 0.114 * target_img[:, 2:3]
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=lum.dtype, device=lum.device).view(1, 1, 3, 3)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=lum.dtype, device=lum.device).view(1, 1, 3, 3)
        grad_x = torch.nn.functional.conv2d(lum, kx, padding=1)
        grad_y = torch.nn.functional.conv2d(lum, ky, padding=1)
        mag = (grad_x.abs() + grad_y.abs())
        bmax = mag.view(mag.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
        mag = mag / (bmax + 1e-9)
        mask = (1.0 - mag).clamp(0.0, 1.0)
    elif mtype == 'control':
        ctrl = getattr(batch, 'control_tensor', None)
        mask = None
        if ctrl is not None:
            # Use the shared helper to build a control-derived mask at the prediction image size
            Ht = pred_img.shape[2]
            Wt = pred_img.shape[3]
            mask = build_control_mask(ctrl, train_config, target_size=(Ht, Wt), device_torch=device_torch)

    if mask is None:
        return current_loss, None

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    try:
        mloss = masked_mse(pred_img, target_img, mask)
    except Exception:
        mloss = None

    # NOTE: per-step mask preview saving was removed in favor of a single per-job preview run.
    # The new approach generates one mask + overlay per dataset item once at job start (see `save_mask_previews`).
    pass

    weight = float(getattr(train_config, 'masked_recon_weight', 0.0))
    if weight != 0.0 and mloss is not None:
        current_loss = current_loss + (weight * mloss)
        return current_loss, mloss.detach()
    return current_loss, None


def save_mask_previews(datasets, train_config, sd, save_path_tpl: str, overwrite: bool = False, overlay: bool = True):
    """Generate one mask + optional overlay per dataset item and save to disk.

    Args:
        datasets: iterable of dataset objects exposing `file_list` (list of FileItemDTO-like objects).
        train_config: training config (used for mask generation params).
        sd: StableDiffusion instance (used only for job name lookup if needed).
        save_path_tpl: directory path template already formatted with job_name (e.g., 'output/myjob/masks').
        overwrite: whether to overwrite existing files.
        overlay: whether to create overlay PNGs (default True).

    Returns:
        index: list of records {'src': original_path, 'mask': mask_path, 'overlay': overlay_path}
    """
    import json

    try:
        from PIL import Image as PILImage
        from PIL import ImageOps
        from PIL.ImageOps import exif_transpose as _exif_transpose
        import numpy as _np
        from toolkit.visualization import save_mask_preview
    except Exception:
        PILImage = None

    out_dir = save_path_tpl
    os.makedirs(out_dir, exist_ok=True)

    index = []
    for ds in datasets:
        file_list = getattr(ds, 'file_list', None)
        if file_list is None:
            continue
        for fi in file_list:
            try:
                # determine target size
                Ht = int(getattr(fi, 'crop_height', getattr(fi, 'scale_to_height', 256)))
                Wt = int(getattr(fi, 'crop_width', getattr(fi, 'scale_to_width', 256)))

                # find control source
                ctrl = None
                if getattr(fi, 'control_path', None) is not None:
                    cp = fi.control_path
                    if isinstance(cp, list):
                        tensors = []
                        for p in cp:
                            try:
                                img = PILImage.open(p).convert('RGB')
                                t = TF.to_tensor(img)
                                tensors.append(t)
                            except Exception:
                                continue
                        if len(tensors) == 0:
                            continue
                        ctrl = tensors
                    else:
                        # single path
                        ctrl = cp
                elif getattr(fi, 'control_tensor', None) is not None:
                    ctrl = fi.control_tensor
                elif getattr(fi, 'control_tensor_list', None) is not None:
                    ctrl = fi.control_tensor_list
                else:
                    # nothing to build a mask from
                    continue

                # build mask
                mask = None
                try:
                    mask = build_control_mask(ctrl, train_config, target_size=(Ht, Wt), device_torch='cpu')
                except Exception:
                    mask = None
                if mask is None:
                    continue
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)

                # single-file only (expect batch dim = 1)
                if mask.shape[0] > 1:
                    use_mask = mask[0, 0]
                else:
                    use_mask = mask[0, 0]

                # file paths
                basename = os.path.splitext(os.path.basename(getattr(fi, 'path', 'unknown')))[0]
                mask_path = os.path.join(out_dir, f"{basename}_mask.png")
                overlay_path = os.path.join(out_dir, f"{basename}_overlay.png") if overlay else None

                if (not overwrite) and os.path.exists(mask_path):
                    # skip write but still add to index
                    index.append({'src': getattr(fi, 'path', None), 'mask': mask_path, 'overlay': overlay_path})
                    continue

                try:
                    save_mask_preview(use_mask, mask_path)
                except Exception:
                    # best-effort; skip failures
                    continue

                if overlay and PILImage is not None:
                    try:
                        base_img_path = getattr(fi, 'path', None)
                        if base_img_path is not None and os.path.exists(base_img_path):
                            base_img = PILImage.open(base_img_path)
                            base_img = _exif_transpose(base_img).convert('RGBA')
                            base_img = base_img.resize((Wt, Ht), PILImage.BICUBIC)
                            # mask array
                            m = (use_mask.detach().cpu().numpy() * 255.0).astype(_np.uint8)
                            alpha = PILImage.fromarray(m).convert('L')
                            red = PILImage.new('RGBA', base_img.size, (255, 0, 0, 0))
                            # build overlay by tinting red where mask > 0
                            # create colored overlay with alpha proportional to mask
                            alpha_rgba = alpha.point(lambda x: int(x))
                            colored = PILImage.new('RGBA', base_img.size, (255, 0, 0, 0))
                            colored.putalpha(alpha_rgba)
                            composed = PILImage.alpha_composite(base_img, colored)
                            composed.save(overlay_path)
                        else:
                            # cannot form overlay without base image; create an empty placeholder
                            pass
                    except Exception:
                        pass

                index.append({'src': getattr(fi, 'path', None), 'mask': mask_path, 'overlay': overlay_path})

            except Exception:
                # ignore per-file failures
                continue

    # write index
    try:
        idx_path = os.path.join(out_dir, 'index.json')
        with open(idx_path, 'w') as f:
            json.dump(index, f, indent=2)
    except Exception:
        pass

    return index