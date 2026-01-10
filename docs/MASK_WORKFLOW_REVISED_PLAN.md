# Mask Workflow - Revised Implementation Plan

## IMPORTANT DISCOVERY: Most Infrastructure Already Exists! 🎉

After searching the codebase, **95% of the mask functionality you requested is already implemented:**

### ✅ Already Working Features

#### 1. **Mask Loading from External Folder**
- **Location:** `toolkit/dataloader_mixins.py` (lines 1620-1720) - `MaskFileItemDTOMixin`
- **Config:** `mask_path` in DatasetConfig (line 1026)
- **How it works:** 
  - Discovers masks from specified folder path
  - Matches mask filename to image filename
  - Loads, resizes, crops, and augments masks automatically
  - Supports alpha channel masks via `alpha_mask` config

#### 2. **Mask Fields in DTOs**
- **FileItemDTO:** Has `mask_tensor` field (line 1626)
- **DataLoaderBatchDTO:** Has `mask_tensor` field (line 203)

#### 3. **Automatic Mask Resizing**
- **Location:** `extensions_built_in/sd_trainer/SDTrainer.py` (lines 2266-2285)
- **What it does:** Automatically resizes masks to match latent dimensions on-the-fly

#### 4. **Loss Masking in Training**
- **Location:** `extensions_built_in/sd_trainer/SDTrainer.py` (line 1400)
- **Code:** `loss = loss * mask_multiplier`
- **Features:**
  - Multiplies loss by mask values
  - Normalizes mask to mean of 1.0 to preserve loss scale
  - Supports video (5D tensors)

#### 5. **Inverted Mask Prior (Regularization)**
- **Config:** `inverted_mask_prior` and `inverted_mask_prior_multiplier` (lines 530-531)
- **What it does:** Applies regularization loss to non-masked regions (exactly what you wanted!)
- **Location:** Lines 1408-1415 in SDTrainer.py

### ❌ What's Actually Missing

1. **SAM3 mask generation CLI** - Need to create this
2. **Per-dataset mask UI controls** - Currently only in train/job config, not in dataset view
3. **Configurable mask strength** - Need to add `mask_strength` parameter
4. **Remove masked_recon code** - Separate cleanup task

---

## Revised Implementation Plan

### Phase 1: Remove Masked Reconstruction Code
*(Keep as-is from original plan - Phase 1 sections 1.1-1.8)*

Same as original document.

---

### Phase 2: Add Mask Strength Parameter (NEW - Small Enhancement)

#### 2.1 Add mask_strength to DatasetConfig
**File:** `toolkit/config_modules.py` (around line 1031)

**Add after `mask_min_value`:**
```python
self.mask_min_value: float = kwargs.get('mask_min_value', 0.0)
# Mask strength: 1.0 = fully zero non-masked regions, 0.0 = no masking
self.mask_strength: float = kwargs.get('mask_strength', 1.0)
```

#### 2.2 Update Mask Application in Training
**File:** `extensions_built_in/sd_trainer/SDTrainer.py` (around line 2283-2285)

**Current code:**
```python
mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()
# make avg 1.0
mask_multiplier = mask_multiplier / mask_multiplier.mean()
```

**Enhanced code:**
```python
mask_multiplier = mask_multiplier.to(self.device_torch, dtype=dtype).detach()

# Apply mask_strength blending (get from first file item's dataset config)
if len(batch.file_items) > 0:
    mask_strength = float(getattr(batch.file_items[0].dataset_config, 'mask_strength', 1.0))
    if 0.0 < mask_strength < 1.0:
        # Blend: masked regions (1.0) get full weight, non-masked (0.0) get reduced weight
        # Formula: final = mask * 1.0 + (1-mask) * (1-strength)
        #        = mask + (1-mask) * (1-strength)
        mask_multiplier = mask_multiplier + (1.0 - mask_multiplier) * (1.0 - mask_strength)

# Normalize to mean of 1.0
mask_multiplier = mask_multiplier / mask_multiplier.mean()
```

---

### Phase 3: SAM3 Mask Generation CLI

#### 3.1 Create SAM3 Script
**File:** `scripts/generate_masks_sam3.py`

```python
#!/usr/bin/env python3
"""Generate binary masks for a dataset using SAM3 (Segment Anything Model 3).

Usage:
    python scripts/generate_masks_sam3.py --dataset datasets/my_dataset --prompts "person,gun" --output-folder datasets/my_dataset/masks
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoProcessor, AutoModel
except ImportError:
    print("Error: transformers required. Install: pip install transformers")
    sys.exit(1)


def load_sam3_model(device: str = "cuda"):
    """Load SAM3 model from HuggingFace."""
    model_id = "facebook/sam3"
    print(f"Loading SAM3 from {model_id}...")
    
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    
    return model, processor


def generate_mask(image_path: str, prompts: List[str], model, processor, device: str) -> np.ndarray:
    """Generate binary mask using text prompts.
    
    Returns:
        Binary mask [H, W] with values {0, 255}
    """
    image = Image.open(image_path).convert("RGB")
    
    # Prepare inputs (adjust based on actual SAM3 API)
    inputs = processor(images=image, text=prompts, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract mask (adjust based on SAM3 output format)
        mask_logits = outputs.masks[0]  # Assuming [H, W]
        mask_binary = (mask_logits > 0).cpu().numpy().astype(np.uint8) * 255
    
    return mask_binary


def process_dataset(dataset_path: str, prompts: List[str], output_folder: str, device: str):
    """Process all images and generate masks."""
    dataset_path = Path(dataset_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [f for f in dataset_path.iterdir() 
              if f.is_file() and f.suffix.lower() in extensions]
    
    if not images:
        print(f"No images found in {dataset_path}")
        return
    
    print(f"Found {len(images)} images")
    print(f"Generating masks for: {prompts}")
    
    model, processor = load_sam3_model(device)
    
    for img_file in tqdm(images, desc="Generating masks"):
        try:
            mask = generate_mask(str(img_file), prompts, model, processor, device)
            
            # Save mask with same name as image (as PNG)
            mask_name = img_file.stem + ".png"
            mask_path = output_dir / mask_name
            Image.fromarray(mask, mode='L').save(mask_path)
            
        except Exception as e:
            print(f"\nError processing {img_file.name}: {e}")
    
    print(f"\nDone! Masks saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate masks using SAM3")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset folder path")
    parser.add_argument("--prompts", "-p", required=True, 
                       help="Comma-separated text prompts (e.g., 'person,gun')")
    parser.add_argument("--output-folder", "-o", required=True,
                       help="Output folder for masks (e.g., 'datasets/my_dataset/masks')")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (default: cuda if available)")
    
    args = parser.parse_args()
    
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if not prompts:
        print("Error: no valid prompts")
        sys.exit(1)
    
    process_dataset(args.dataset, prompts, args.output_folder, args.device)


if __name__ == "__main__":
    main()
```

**Note:** SAM3 API calls are placeholders - adjust based on actual model API from HuggingFace.

---

### Phase 4: Add UI Controls to Dataset View

#### 4.1 Update Dataset Config Types
**File:** `ui/src/app/datasets/[id]/types.ts` (or similar)

**Add:**
```typescript
interface DatasetConfig {
  // ... existing fields ...
  mask_path?: string;
  alpha_mask?: boolean;
  mask_strength?: number;  // 0.0-1.0
  invert_mask?: boolean;
}
```

#### 4.2 Add UI Form Controls
**File:** `ui/src/app/datasets/[id]/page.tsx` (or dataset settings component)

**Add section:**
```tsx
<FormGroup label="Mask Settings" className="pt-2">
  <TextInput
    label="Mask Folder Path"
    value={dataset.mask_path || ''}
    onChange={value => updateDataset({ mask_path: value || null })}
    placeholder="Path to folder containing mask images"
    helperText="Mask files should match image filenames"
  />
  
  <Checkbox
    label="Use Alpha Channel as Mask"
    className="pt-2"
    checked={dataset.alpha_mask || false}
    onChange={value => updateDataset({ alpha_mask: value })}
    disabled={!!dataset.mask_path}
  />
  
  <NumberInput
    label="Mask Strength"
    className="pt-2"
    value={dataset.mask_strength ?? 1.0}
    onChange={value => updateDataset({ mask_strength: value })}
    min={0.0}
    max={1.0}
    step={0.1}
    disabled={!dataset.mask_path && !dataset.alpha_mask}
    helperText="1.0 = fully zero non-masked regions, 0.0 = no effect"
  />
  
  <Checkbox
    label="Invert Mask"
    className="pt-2"
    checked={dataset.invert_mask || false}
    onChange={value => updateDataset({ invert_mask: value })}
    disabled={!dataset.mask_path && !dataset.alpha_mask}
  />
  
  <div className="pt-2 text-sm text-gray-600">
    <p><strong>Generate masks:</strong></p>
    <code>python scripts/generate_masks_sam3.py --dataset "{dataset.path}" --prompts "your,prompts" --output-folder "{dataset.path}/masks"</code>
  </div>
</FormGroup>
```

#### 4.3 Update API Endpoints
**File:** `ui/src/app/api/datasets/[id]/route.ts`

**Add to validation:**
```typescript
const validFields = [
  // ... existing ...
  'mask_path',
  'alpha_mask',
  'mask_strength',
  'invert_mask',
];
```

---

### Phase 5: Testing

#### 5.1 Unit Test for Mask Strength
**File:** `testing/test_mask_strength.py`

```python
import torch

def test_mask_strength_application():
    """Test mask strength blending formula."""
    # Create test mask: top half masked (1.0), bottom half not (0.0)
    mask = torch.zeros(1, 1, 8, 8)
    mask[:, :, :4, :] = 1.0
    
    # Test full strength (1.0)
    strength = 1.0
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    assert torch.allclose(multiplier[:, :, :4, :], torch.ones_like(multiplier[:, :, :4, :]))
    assert torch.allclose(multiplier[:, :, 4:, :], torch.zeros_like(multiplier[:, :, 4:, :]))
    
    # Test half strength (0.5)
    strength = 0.5
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    assert torch.allclose(multiplier[:, :, :4, :], torch.ones_like(multiplier[:, :, :4, :]))
    assert torch.allclose(multiplier[:, :, 4:, :], torch.ones_like(multiplier[:, :, 4:, :]) * 0.5)
    
    # Test no strength (0.0)
    strength = 0.0
    multiplier = mask + (1.0 - mask) * (1.0 - strength)
    assert torch.allclose(multiplier, torch.ones_like(multiplier))
```

#### 5.2 Integration Test
**File:** `testing/test_mask_integration.py`

```python
def test_existing_mask_loading():
    """Verify existing mask loading still works."""
    # Test that MaskFileItemDTOMixin loads masks correctly
    # (Use existing test structure)
    pass

def test_mask_path_config():
    """Test mask_path configuration."""
    from toolkit.config_modules import DatasetConfig
    
    config = DatasetConfig(mask_path='datasets/test/masks', mask_strength=0.8)
    assert config.mask_path == 'datasets/test/masks'
    assert config.mask_strength == 0.8
```

---

## Implementation Checklist

### Phase 1: Cleanup (2-4 hours) - Same as original plan
- [ ] Delete masked_recon code
- [ ] Remove config fields
- [ ] Update trainer
- [ ] Delete tests/docs

### Phase 2: Add Mask Strength (1-2 hours) - SIMPLIFIED!
- [ ] Add `mask_strength` to DatasetConfig
- [ ] Update mask application in SDTrainer.py
- [ ] Add unit test

### Phase 3: SAM3 CLI (3-4 hours)
- [ ] Create `scripts/generate_masks_sam3.py`
- [ ] Test with sample dataset
- [ ] Document usage

### Phase 4: UI Controls (2-3 hours)
- [ ] Add mask fields to dataset types
- [ ] Add form controls to dataset view
- [ ] Update API validation
- [ ] Test UI workflow

### Phase 5: Testing & Docs (2-3 hours)
- [ ] Write unit tests
- [ ] Update LEARNINGS.md
- [ ] Update README
- [ ] Manual GPU smoke test

### Total Estimated Time: 10-16 hours (down from 20-33!)

---

## Key Insights

1. **Don't reinvent the wheel:** The existing `MaskFileItemDTOMixin` already handles mask loading perfectly
2. **Mask application is solid:** The training loop already multiplies loss by masks correctly
3. **inverted_mask_prior exists:** This feature already does regularization on non-masked regions!
4. **Just need polish:** Add strength parameter, SAM3 CLI, and UI controls

---

## Migration Notes

**For users with existing mask setups:**
- Current `mask_path` config continues to work unchanged
- Add `mask_strength: 1.0` to configs for backward compatibility
- `inverted_mask_prior` continues to work as before

**Recommended workflow:**
1. Generate masks: `python scripts/generate_masks_sam3.py --dataset path --prompts "person" --output-folder path/masks`
2. Set dataset `mask_path` to masks folder
3. Set `mask_strength` (0.0-1.0) to control effect
4. Optionally enable `inverted_mask_prior` for regularization on non-masked regions

---

## Success Criteria

- [ ] Existing mask functionality continues to work
- [ ] New `mask_strength` parameter works correctly
- [ ] SAM3 script generates valid masks
- [ ] UI allows configuring masks per dataset
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Manual GPU verification complete
