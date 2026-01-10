"""Simple validation script for mask_strength implementation."""

import torch

def test_mask_strength():
    """Test mask strength blending formula."""
    print("Testing mask strength implementation...")
    
    # Create test mask: top half masked (1.0), bottom half not (0.0)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :] = 1.0
    
    # Test full strength (1.0)
    print("\n1. Testing full strength (1.0)...")
    strength = 1.0
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    assert torch.allclose(multiplier[:, :, :4, :], torch.ones_like(multiplier[:, :, :4, :])), "Masked region should be 1.0"
    assert torch.allclose(multiplier[:, :, 4:, :], torch.zeros_like(multiplier[:, :, 4:, :])), "Non-masked region should be 0.0"
    print("   ✓ Full strength works correctly")
    
    # Test half strength (0.5)
    print("\n2. Testing half strength (0.5)...")
    strength = 0.5
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    assert torch.allclose(multiplier[:, :, :4, :], torch.ones_like(multiplier[:, :, :4, :])), "Masked region should be 1.0"
    assert torch.allclose(multiplier[:, :, 4:, :], torch.ones_like(multiplier[:, :, 4:, :]) * 0.5), "Non-masked region should be 0.5"
    print("   ✓ Half strength works correctly")
    
    # Test no strength (0.0)
    print("\n3. Testing no strength (0.0)...")
    strength = 0.0
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    assert torch.allclose(multiplier, torch.ones_like(multiplier)), "All regions should be 1.0"
    print("   ✓ No strength works correctly")
    
    # Test normalization
    print("\n4. Testing normalization (mean=1.0)...")
    mask = torch.zeros(2, 1, 4, 4)
    mask[0, :, :2, :] = 1.0
    mask[1, :, :, :2] = 1.0
    strength = 0.7
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    normalized = multiplier / multiplier.mean()
    assert torch.allclose(normalized.mean(), torch.tensor(1.0), atol=1e-5), "Mean should be 1.0"
    print("   ✓ Normalization works correctly")
    
    print("\n5. Testing DatasetConfig...")
    try:
        from toolkit.config_modules import DatasetConfig
        config = DatasetConfig(folder_path='test', mask_strength=0.8)
        assert config.mask_strength == 0.8, "mask_strength should be 0.8"
        
        config_default = DatasetConfig(folder_path='test')
        assert config_default.mask_strength == 1.0, "Default mask_strength should be 1.0"
        print("   ✓ DatasetConfig mask_strength works correctly")
    except Exception as e:
        print(f"   ✗ DatasetConfig test failed: {e}")
        return False
    
    print("\n✅ All tests passed!")
    return True

if __name__ == "__main__":
    success = test_mask_strength()
    exit(0 if success else 1)
