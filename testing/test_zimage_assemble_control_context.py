import torch
import torch.nn.functional as F
import pytest
from toolkit.control_channels import assemble_zimage_control_context


def test_pass_through_when_control_in_dim_matches():
    B, C, H, W = 2, 16, 8, 8
    x = torch.randn(B, C, H, W)
    out = assemble_zimage_control_context(x, control_in_dim=C)
    assert out is not None
    assert out.ndim == 4
    assert out.shape == x.shape
    assert torch.allclose(out, x)


def test_assemble_concat_basic():
    B, C, H, W = 1, 16, 8, 8
    in_h, in_w = 4, 4
    control_latents = torch.full((B, C, H, W), 2.0)
    inpaint_latent = torch.full((B, C, in_h, in_w), 5.0)
    # mask zeros -> mask_single = 1 - 0 = 1
    mask_condition = torch.zeros((B, 3, H, W))

    out = assemble_zimage_control_context(control_latents, inpaint_latent=inpaint_latent, mask_condition=mask_condition, control_in_dim=33)
    assert out is not None
    # Expect 5D tensor [B, 33, 1, in_h, in_w]
    assert out.ndim == 5
    assert out.shape == (B, 33, 1, in_h, in_w)

    # Check first C channels are control latents resized to inpaint size
    ctl_resized = F.interpolate(control_latents, size=(in_h, in_w), mode='bilinear', align_corners=False)
    assert torch.allclose(out[:, :C, 0, :, :], ctl_resized)

    # Next channel is mask_single (all ones)
    assert torch.allclose(out[:, C:C+1, 0, :, :], torch.ones((B, 1, in_h, in_w)))

    # Last C channels are inpaint latent
    assert torch.allclose(out[:, C+1:, 0, :, :], inpaint_latent)


def test_assemble_defaults_when_missing_mask_and_inpaint():
    B, C, H, W = 2, 16, 8, 8
    control_latents = torch.randn(B, C, H, W)
    # control_in_dim desired 33
    out = assemble_zimage_control_context(control_latents, inpaint_latent=None, mask_condition=None, control_in_dim=33)
    assert out is not None
    assert out.ndim == 5
    # Check shape
    tgt_h = control_latents.shape[-2]
    tgt_w = control_latents.shape[-1]
    assert out.shape == (B, 33, 1, tgt_h, tgt_w)
    # Check that inpaint (last C channels) are zeros
    assert torch.allclose(out[:, -C:, 0, :, :], torch.zeros((B, C, tgt_h, tgt_w)))
    # Check mask channel is zeros inverted -> ones
    assert torch.allclose(out[:, C:C+1, 0, :, :], torch.ones((B, 1, tgt_h, tgt_w)))


def test_assemble_detect_mask_from_black_background():
    B, C, H, W = 1, 16, 8, 8
    control_latents = torch.randn(B, C, H, W)
    # Black background with a small white square in the middle
    img = torch.zeros((B, 3, H, W))
    img[:, :, 2:6, 2:6] = 1.0
    out = assemble_zimage_control_context(control_latents, inpaint_latent=None, mask_condition=None, control_in_dim=33, mask_from=img)
    # mask_single is placed at channel index C
    mask_ch = out[:, C:C+1, 0, :, :]
    # corners are background -> mask_single should be 0 (since bg_mask=1 -> mask_single=1-bg_mask)
    assert torch.allclose(mask_ch[0, :, 0:2, 0:2], torch.zeros((1, 1, 2, 2)))
    # center region (foreground) should be 1
    assert torch.allclose(mask_ch[0, :, 3:5, 3:5], torch.ones((1, 1, 2, 2)))


def test_assemble_detect_mask_from_white_background():
    B, C, H, W = 1, 16, 8, 8
    control_latents = torch.randn(B, C, H, W)
    # White background with a small black square in the middle
    img = torch.ones((B, 3, H, W))
    img[:, :, 2:6, 2:6] = 0.0
    out = assemble_zimage_control_context(control_latents, inpaint_latent=None, mask_condition=None, control_in_dim=33, mask_from=img)
    mask_ch = out[:, C:C+1, 0, :, :]
    # corners background -> mask_single should be 0
    assert torch.allclose(mask_ch[0, :, 0:2, 0:2], torch.zeros((1, 1, 2, 2)))
    # center region (foreground) should be 1
    assert torch.allclose(mask_ch[0, :, 3:5, 3:5], torch.ones((1, 1, 2, 2)))


def test_assemble_rejects_3_channel_raw_image():
    B, C, H, W = 1, 3, 8, 8
    control_latents = torch.full((B, C, H, W), 2.0)
    in_h, in_w = 4, 4
    inpaint_latent = torch.full((B, C, in_h, in_w), 5.0)
    # mask zeros -> mask_single = 1 - 0 = 1
    mask_condition = torch.zeros((B, 3, H, W))

    with pytest.raises(RuntimeError) as exc:
        assemble_zimage_control_context(control_latents, inpaint_latent=inpaint_latent, mask_condition=mask_condition, control_in_dim=33)
    msg = str(exc.value)
    assert 'Unsupported control_latents channels' in msg or 'raw' in msg


def test_pass_through_preassembled_33():
    B, C, H, W = 1, 33, 8, 8
    x = torch.randn((B, C, H, W))
    out = assemble_zimage_control_context(x, control_in_dim=33)
    assert out is x


def test_rejects_other_channel_counts():
    B, C, H, W = 1, 8, 8, 8
    x = torch.randn((B, C, H, W))
    with pytest.raises(RuntimeError) as exc:
        assemble_zimage_control_context(x, control_in_dim=33)
    assert 'Unsupported control_latents channels' in str(exc.value)
