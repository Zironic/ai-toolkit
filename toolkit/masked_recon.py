import os
import torch
from toolkit.print import print_acc

from toolkit.losses import masked_mse, luminance_mask_from_images


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
        act = ctrl_tensor.abs().sum(dim=1, keepdim=True)
        Ht, Wt = int(target_size[0]), int(target_size[1])
        ref_max = int(getattr(train_config, 'masked_recon_control_ref_max_size', max(Ht, Wt)))
        scale = float(ref_max) / float(max(Ht, Wt)) if max(Ht, Wt) > 0 else 1.0
        ref_H = max(1, int(round(Ht * scale)))
        ref_W = max(1, int(round(Wt * scale)))
        act = torch.nn.functional.interpolate(act, size=(ref_H, ref_W), mode='bilinear', align_corners=False)
        max_per_sample = act.view(act.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
        act = act / (max_per_sample + 1e-9)
        t = float(getattr(train_config, 'masked_recon_control_threshold', 0.05))
        mask0 = (act > t).float()
        base_d = int(getattr(train_config, 'masked_recon_control_dilate', 7))
        do_auto = bool(getattr(train_config, 'masked_recon_control_dilate_auto', True))
        scale_factor = float(getattr(train_config, 'masked_recon_control_dilate_scale_factor', 4.0))

        if do_auto:
            bs, _, Hm, Wm = mask0.shape
            per_sample_ds = []
            for si in range(bs):
                m = mask0[si, 0]
                nz = torch.nonzero(m, as_tuple=False)
                if nz.numel() == 0:
                    frac = 0.0
                else:
                    ys = nz[:, 0]
                    xs = nz[:, 1]
                    h_bbox = int(ys.max() - ys.min() + 1)
                    w_bbox = int(xs.max() - xs.min() + 1)
                    frac = max(h_bbox, w_bbox) / float(max(Hm, Wm))
                d_auto = max(1, int(round(base_d * (1.0 + frac * scale_factor))))
                per_sample_ds.append(d_auto)
            if len(set(per_sample_ds)) == 1:
                d_use = per_sample_ds[0]
                if d_use > 1:
                    mask1 = torch.nn.functional.max_pool2d(mask0, kernel_size=d_use, stride=1, padding=d_use // 2)
                else:
                    mask1 = mask0
            else:
                out_masks = []
                for si, k in enumerate(per_sample_ds):
                    mi = mask0[si:si+1]
                    if k > 1:
                        out_masks.append(torch.nn.functional.max_pool2d(mi, kernel_size=k, stride=1, padding=k // 2))
                    else:
                        out_masks.append(mi)
                mask1 = torch.cat(out_masks, dim=0)
        else:
            d = base_d
            if d > 1:
                mask1 = torch.nn.functional.max_pool2d(mask0, kernel_size=d, stride=1, padding=d // 2)
            else:
                mask1 = mask0

        b = int(getattr(train_config, 'masked_recon_control_blur', 9))
        if b > 1:
            mask_ref = torch.nn.AvgPool2d(kernel_size=b, stride=1, padding=b // 2)(mask1)
        else:
            mask_ref = mask1

        max_per_sample = mask_ref.view(mask_ref.shape[0], -1).amax(dim=1).view(-1, 1, 1, 1)
        mask_ref = mask_ref / (max_per_sample + 1e-9)
        mask_ref = mask_ref.clamp(0.0, 1.0)

        mask = torch.nn.functional.interpolate(mask_ref, size=(Ht, Wt), mode='area')
        mask = mask.clamp(0.0, 1.0)
        return mask
    except Exception:
        return None


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
    try:
        if hasattr(sd, 'vae') and sd.vae is not None:
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
        else:
            return current_loss, None
    except Exception:
        return current_loss, None

    if target_img.min() < 0.0:
        target_img = (target_img + 1.0) / 2.0

    mtype = getattr(train_config, 'masked_recon_type', 'illum')
    mask = None
    try:
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
                try:
                    # Use the shared helper to build a control-derived mask at the prediction image size
                    Ht = pred_img.shape[2]
                    Wt = pred_img.shape[3]
                    mask = build_control_mask(ctrl, train_config, target_size=(Ht, Wt), device_torch=device_torch)
                except Exception:
                    mask = None
    except Exception:
        mask = None

    if mask is None:
        return current_loss, None

    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    try:
        mloss = masked_mse(pred_img, target_img, mask)
    except Exception:
        mloss = None

    # optionally save mask preview images for debugging
    try:
        if getattr(train_config, 'mask_preview_enabled', False):
            max_steps = int(getattr(train_config, 'mask_preview_max_steps', 10))
            samples_per_step = int(getattr(train_config, 'mask_preview_samples_per_step', 2))
            save_path_tpl = getattr(train_config, 'mask_preview_save_path', 'output/{job_name}/masks')
            overwrite = bool(getattr(train_config, 'mask_preview_overwrite', False))
            step = int(getattr(getattr(sd, 'trainer', None), '_total_batch_count', 0)) if getattr(getattr(sd, 'trainer', None), '_total_batch_count', None) is not None else 0
            if step <= max_steps:
                job_name = getattr(getattr(sd, 'trainer', None), 'job', None)
                job_name = getattr(job_name, 'name', 'job') if job_name is not None else 'job'
                save_dir = save_path_tpl.format(job_name=job_name)
                b = mask.shape[0]
                n_save = min(b, samples_per_step)
                from toolkit.visualization import save_mask_preview
                for i in range(n_save):
                    suffix = f"step_{step:06d}_{i}.png"
                    full_path = os.path.join(save_dir, suffix)
                    if (os.path.exists(full_path) and not overwrite):
                        continue
                    try:
                        save_mask_preview(mask[i, 0], full_path)
                    except Exception as e:
                        print(f"[MASK_PREVIEW] failed to save preview: {e}")
    except Exception:
        pass

    weight = float(getattr(train_config, 'masked_recon_weight', 0.0))
    if weight != 0.0 and mloss is not None:
        current_loss = current_loss + (weight * mloss)
        return current_loss, mloss.detach()
    return current_loss, None