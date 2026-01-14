import pytest
# Robust torch import: try importing torch, and if an inspect-related AttributeError occurs
# (modules with malformed __file__ attributes), clean sys.modules and retry before skipping.
try:
    import torch
except Exception as e:
    import sys
    msg = str(e)
    bad = []
    # if inspect-related __file__ bug: enumerate bad modules, remove and retry
    if isinstance(e, AttributeError) and "'__file__' has no attribute 'endswith'" in msg:
        for name, mod in list(sys.modules.items()):
            try:
                if hasattr(mod, '__file__'):
                    f = getattr(mod, '__file__')
                    if f is not None and not isinstance(f, str):
                        bad.append(name)
            except Exception:
                bad.append(name)
        # Remove suspicious modules and retry import
        for name in bad:
            sys.modules.pop(name, None)
        try:
            import importlib
            importlib.invalidate_caches()
            import torch
        except Exception as e2:
            pytest.skip(f"Skipping: torch import failed after sanitization; last exception: {e2}; bad modules: {bad}", allow_module_level=True)
    else:
        pytest.skip(f"Skipping: torch import failed: {e}", allow_module_level=True)

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer


def test_mask_strength_zero_returns_ones():
    B, C, H, W = 2, 4, 8, 8
    mask = torch.zeros((B, 1, H, W), dtype=torch.float32)
    out = SDTrainer.compute_mask_multiplier(mask, (B, C, H, W), device='cpu', dtype=torch.float32, mask_strength=0.0)
    assert out.shape == (B, 1, 1, 1) or out.shape == (B, C, H, W)
    # After normalization, mean per-sample should be 1
    mean = out.mean(dim=[1,2,3])
    assert torch.allclose(mean, torch.ones_like(mean), atol=1e-6)
    # All ones
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)


def test_mask_strength_one_normalized_mean():
    B, C, H, W = 1, 3, 4, 4
    mask = torch.zeros((B, 1, H, W), dtype=torch.float32)
    mask[0, 0, 1:3, 1:3] = 1.0
    out = SDTrainer.compute_mask_multiplier(mask, (B, C, H, W), device='cpu', dtype=torch.float32, mask_strength=1.0)
    # mean per-sample should be 1
    mean = out.mean(dim=[1,2,3])
    assert torch.allclose(mean, torch.ones_like(mean), atol=1e-6)
    # Outside masked area should be zero (within tolerance)
    outside = out[0, 0, 0, 0]
    assert outside < 1e-6


def test_rgb_mask_input_and_shape_expansion():
    B, C, H, W = 2, 4, 6, 6
    # Simulate a 3-channel mask image (RGB) where channels may differ
    mask_rgb = torch.rand((B, 3, H, W), dtype=torch.float32)
    out = SDTrainer.compute_mask_multiplier(mask_rgb, (B, C, H, W), device='cpu', dtype=torch.float32, mask_strength=0.5)
    assert out.shape == (B, C, H, W)
    mean = out.mean(dim=[1,2,3])
    assert torch.allclose(mean, torch.ones_like(mean), atol=1e-6)


def test_prior_mask_normalization_and_turbo_slice():
    B, C, H, W = 1, 4, 4, 4
    prior_mask = torch.zeros((B, 1, H, W), dtype=torch.float32)
    prior_mask[0, 0, 0:2, 0:2] = 1.0
    prior_mult = 1.0 - prior_mask
    out = SDTrainer.compute_prior_mask_multiplier(prior_mult)
    assert out.shape == prior_mult.shape
    mean = out.mean(dim=[1,2,3])
    assert torch.allclose(mean, torch.ones_like(mean), atol=1e-6)

    # Turbo slicing: ensure channel 3 is selected if available
    mask_multi = torch.rand((B, 4, H, W), dtype=torch.float32)
    sliced = SDTrainer.apply_turbo_channel_slice(mask_multi)
    assert sliced.shape[1] == 1
    # if only 1 channel, returns that
    mask_single = torch.rand((B, 1, H, W), dtype=torch.float32)
    sliced2 = SDTrainer.apply_turbo_channel_slice(mask_single)
    assert sliced2.shape[1] == 1
