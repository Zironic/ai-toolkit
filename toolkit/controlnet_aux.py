import torch
import torch.nn.functional as F


def compute_control_edge_loss(batch_tensor: torch.Tensor, control_tensor: torch.Tensor, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Compute a simple edge-based auxiliary loss between an image batch and a control tensor (e.g., Canny edges).

    Args:
        batch_tensor: Tensor of shape [B, C, H, W], in 0..1 range or similar
        control_tensor: Tensor of shape [B, Cc, Hc, Wc] (Cc usually 1 or 3)
        device: device to perform computation on

    Returns:
        scalar Tensor (mean over batch) representing the L1 edge loss
    """
    eps = 1e-6
    # Move tensors to device
    img = batch_tensor.to(device)
    ctrl = control_tensor.to(device)

    # Convert image to grayscale
    if img.shape[1] == 3:
        r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        gray = img.mean(dim=1, keepdim=True)

    # Sobel kernels
    kernel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=gray.dtype, device=device)
    kernel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=gray.dtype, device=device)

    pad = 1
    gx = F.conv2d(gray, kernel_x, padding=pad)
    gy = F.conv2d(gray, kernel_y, padding=pad)
    grad = torch.sqrt(gx * gx + gy * gy + eps)

    # normalize gradient to 0..1 per-sample
    b, c, h, w = grad.shape
    maxv = grad.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
    grad_norm = grad / (maxv + eps)

    # Prepare control to match shape
    if ctrl.shape[1] > 1:
        ctrl_gray = ctrl.mean(dim=1, keepdim=True)
    else:
        ctrl_gray = ctrl

    if ctrl_gray.shape[2] != grad_norm.shape[2] or ctrl_gray.shape[3] != grad_norm.shape[3]:
        ctrl_gray = F.interpolate(ctrl_gray, size=(grad_norm.shape[2], grad_norm.shape[3]), mode='bilinear', align_corners=False)

    # L1 loss per sample
    loss_per_sample = F.l1_loss(grad_norm, ctrl_gray, reduction='none')
    loss_per_sample = loss_per_sample.mean(dim=[1, 2, 3])

    return loss_per_sample.mean()
