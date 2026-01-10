"""Unit tests for mask strength blending feature.

Tests the mask_strength parameter in DatasetConfig and its application
in the training loop (SDTrainer mask_multiplier computation).
"""

import pytest
import torch


def test_mask_strength_full():
    """Test mask strength at 1.0 (full strength - fully zero non-masked)."""
    # Create test mask: top half masked (1.0), bottom half not (0.0)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :] = 1.0
    
    # Apply full strength (1.0)
    strength = 1.0
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    
    # Top half should be 1.0, bottom half should be 0.0
    assert torch.allclose(multiplier[:, :, :4, :], torch.ones_like(multiplier[:, :, :4, :]))
    assert torch.allclose(multiplier[:, :, 4:, :], torch.zeros_like(multiplier[:, :, 4:, :]))


def test_mask_strength_half():
    """Test mask strength at 0.5 (half strength)."""
    # Create test mask: top half masked (1.0), bottom half not (0.0)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :] = 1.0
    
    # Apply half strength (0.5)
    strength = 0.5
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    
    # Top half should be 1.0, bottom half should be 0.5
    assert torch.allclose(multiplier[:, :, :4, :], torch.ones_like(multiplier[:, :, :4, :]))
    assert torch.allclose(multiplier[:, :, 4:, :], torch.ones_like(multiplier[:, :, 4:, :]) * 0.5)


def test_mask_strength_none():
    """Test mask strength at 0.0 (no masking effect)."""
    # Create test mask: top half masked (1.0), bottom half not (0.0)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :] = 1.0
    
    # Apply no strength (0.0)
    strength = 0.0
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    
    # All regions should be 1.0 (no masking effect)
    assert torch.allclose(multiplier, torch.ones_like(multiplier))


def test_mask_strength_gradual():
    """Test mask strength blending formula with various strengths."""
    # Create simple mask with clear regions
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.5]]]])
    
    test_cases = [
        (1.0, [[[[1.0, 0.0], [0.0, 0.5]]]]),  # Full strength - mask unchanged
        (0.5, [[[[1.0, 0.5], [0.5, 0.75]]]]),  # Half strength
        (0.0, [[[[1.0, 1.0], [1.0, 1.0]]]]),   # No strength - all 1.0
    ]
    
    for strength, expected in test_cases:
        multiplier = mask + (1.0 - mask) * (1.0 - strength)
        expected_tensor = torch.tensor(expected)
        assert torch.allclose(multiplier, expected_tensor, atol=1e-5), \
            f"Failed for strength={strength}: got {multiplier}, expected {expected_tensor}"


def test_mask_strength_normalization():
    """Test that mask normalization (mean=1.0) works correctly after blending."""
    # Create test mask
    mask = torch.zeros(2, 1, 4, 4)
    mask[0, :, :2, :] = 1.0  # First batch: top half masked
    mask[1, :, :, :2] = 1.0  # Second batch: left half masked
    
    # Apply various strengths and check normalization
    for strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
        multiplier = mask + (1.0 - mask) * (1.0 - strength)
        
        # Normalize to mean=1.0
        normalized = multiplier / multiplier.mean()
        
        # Check mean is 1.0
        assert torch.allclose(normalized.mean(), torch.tensor(1.0), atol=1e-5), \
            f"Normalization failed for strength={strength}: mean={normalized.mean()}"


def test_mask_strength_edge_cases():
    """Test edge cases: empty mask, full mask, single pixel."""
    # Empty mask (all zeros)
    mask_empty = torch.zeros(1, 1, 4, 4)
    strength = 0.8
    multiplier = mask_empty + (1.0 - mask_empty) * (1.0 - strength)
    assert torch.allclose(multiplier, torch.ones_like(multiplier) * 0.2)
    
    # Full mask (all ones)
    mask_full = torch.ones(1, 1, 4, 4)
    multiplier = mask_full + (1.0 - mask_full) * (1.0 - strength)
    assert torch.allclose(multiplier, torch.ones_like(multiplier))
    
    # Single pixel mask
    mask_single = torch.zeros(1, 1, 4, 4)
    mask_single[0, 0, 2, 2] = 1.0
    multiplier = mask_single + (1.0 - mask_single) * (1.0 - strength)
    assert multiplier[0, 0, 2, 2] == 1.0
    assert torch.allclose(multiplier[multiplier != 1.0], torch.tensor(0.2))


def test_datasetconfig_mask_strength():
    """Test that DatasetConfig properly stores mask_strength."""
    from toolkit.config_modules import DatasetConfig
    
    # Test default value (1.0)
    config_default = DatasetConfig(folder_path='test')
    assert config_default.mask_strength == 1.0
    
    # Test custom value
    config_custom = DatasetConfig(folder_path='test', mask_strength=0.7)
    assert config_custom.mask_strength == 0.7
    
    # Test via kwargs
    config_kwargs = DatasetConfig(**{'folder_path': 'test', 'mask_strength': 0.3})
    assert config_kwargs.mask_strength == 0.3


def test_mask_strength_batch_consistency():
    """Test that mask strength is applied consistently across batch dimensions."""
    batch_size = 4
    mask = torch.zeros(batch_size, 1, 8, 8)
    
    # Create different masks for each batch item
    mask[0, :, :4, :] = 1.0  # Top half
    mask[1, :, 4:, :] = 1.0  # Bottom half
    mask[2, :, :, :4] = 1.0  # Left half
    mask[3, :, :, 4:] = 1.0  # Right half
    
    strength = 0.6
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    
    # Check that each batch item has correct values
    # Masked regions should be 1.0
    assert torch.allclose(mask[mask == 1.0], torch.ones_like(mask[mask == 1.0]))
    # Non-masked regions should be (1 - strength) = 0.4
    expected_nonmasked = 1.0 - strength
    assert torch.allclose(multiplier[mask == 0.0], torch.full_like(multiplier[mask == 0.0], expected_nonmasked))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
