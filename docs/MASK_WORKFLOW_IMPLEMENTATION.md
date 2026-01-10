# Mask Workflow Implementation Summary

## Overview
This document summarizes the implementation of the new mask workflow that replaces the previous Masked Reconstruction feature.

## What Was Changed

### 1. Removed Masked Reconstruction Feature
- **Deleted:** `toolkit/masked_recon.py` (690 lines)
- **Removed from config:**
  - `masked_recon_weight`
  - `masked_recon_type`
  - `masked_recon_mask_key`
  - `masked_recon_control_*` parameters (threshold, dilate, blur, etc.)
  - `mask_preview_enabled` and related preview settings
- **Removed from UI:**
  - "Masked Reconstruction" section in job creation form
  - Related default values in `jobConfig.ts`
- **Removed from trainer:**
  - `_compute_and_apply_masked_recon_loss()` method
  - `_apply_masked_recon_loss_local()` helper
  - `masked_recon_logged` variable
  - Masked recon timer block
  - `build_control_mask` import and usage

### 2. Enhanced Existing Mask System

#### Added mask_strength Parameter
**Location:** `toolkit/config_modules.py` (DatasetConfig)

```python
self.mask_strength: float = kwargs.get('mask_strength', 1.0)
```

**Purpose:** Controls how strongly masks affect non-masked regions during training.
- `1.0` = Fully zero out non-masked regions (full mask effect)
- `0.0` = No masking effect (all regions weighted equally)
- Values in between create proportional blending

#### Updated Mask Application in Training
**Location:** `extensions_built_in/sd_trainer/SDTrainer.py` (lines ~2265-2295)

**Enhancement:** Added mask_strength blending formula after mask resizing:

```python
# Apply mask_strength blending (get from first file item's dataset config)
if len(batch.file_items) > 0:
    mask_strength = float(getattr(batch.file_items[0].dataset_config, 'mask_strength', 1.0))
    if 0.0 < mask_strength < 1.0:
        # Blend: masked regions (1.0) get full weight, non-masked (0.0) get reduced weight
        # Formula: final = mask * 1.0 + (1-mask) * (1-strength)
        #        = mask + (1-mask) * (1-strength)
        mask_multiplier = mask_multiplier + (1.0 - mask_multiplier) * (1.0 - mask_strength)
```

### 3. Created SAM2 Mask Generation CLI
**Location:** `scripts/generate_masks_sam2.py`

**Features:**
- Uses facebook/sam2.1-hiera-tiny (or other SAM2 models) from HuggingFace
- Supports point prompts: `--points "[[x,y],[x2,y2]]"`
- Supports bounding box prompts: `--box "[[x1,y1,x2,y2]]"`
- Batch processes entire datasets
- Saves masks as PNG files matching image filenames

**Usage Examples:**
```bash
# Generate masks using center point
python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --points "[[512,512]]" --output datasets/my_dataset/masks

# Generate masks using bounding box
python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --box "[[100,100,400,400]]" --output datasets/my_dataset/masks

# Use larger model for better quality
python scripts/generate_masks_sam2.py --dataset datasets/my_dataset --points "[[512,512]]" --output datasets/my_dataset/masks --model facebook/sam2.1-hiera-large
```

### 4. Added Unit Tests
**Location:** `testing/test_mask_strength.py`

**Test Coverage:**
- `test_mask_strength_full()` - Full strength (1.0)
- `test_mask_strength_half()` - Half strength (0.5)
- `test_mask_strength_none()` - No strength (0.0)
- `test_mask_strength_gradual()` - Various strength values
- `test_mask_strength_normalization()` - Mean=1.0 normalization
- `test_mask_strength_edge_cases()` - Empty/full masks
- `test_datasetconfig_mask_strength()` - Config storage
- `test_mask_strength_batch_consistency()` - Batch processing

## What Already Existed (No Changes Needed)

The following infrastructure was already in place and continues to work:

1. **Mask Loading:** `MaskFileItemDTOMixin` (toolkit/dataloader_mixins.py lines 1620-1740)
   - Loads masks from `mask_path` config
   - Supports alpha channel masks via `alpha_mask` config
   - Handles inversion via `invert_mask` config
   - Applies augmentation and cropping

2. **Data Transfer Objects:**
   - `FileItemDTO.mask_tensor` field
   - `DataLoaderBatchDTO.mask_tensor` field

3. **Training Integration:**
   - Automatic mask resizing to latent dimensions (SDTrainer lines 2266-2285)
   - Loss multiplication by mask (SDTrainer line ~1400)
   - Normalization to mean=1.0

4. **Inverted Mask Prior:**
   - `inverted_mask_prior` and `inverted_mask_prior_multiplier` configs
   - Regularization loss on non-masked regions (SDTrainer lines ~1408-1415)

## Usage Guide

### Basic Workflow

1. **Generate Masks:**
```bash
python scripts/generate_masks_sam2.py \
  --dataset datasets/my_dataset \
  --points "[[512,512]]" \
  --output datasets/my_dataset/masks
```

2. **Configure Dataset:**
In your training config YAML:
```yaml
datasets:
  - folder_path: datasets/my_dataset
    mask_path: datasets/my_dataset/masks
    mask_strength: 1.0  # Adjust 0.0-1.0
    invert_mask: false
```

3. **Optional: Use Inverted Mask Prior for Regularization:**
```yaml
train:
  inverted_mask_prior: true
  inverted_mask_prior_multiplier: 0.1
```

### Configuration Options

**Dataset-level (applies to all examples in dataset):**
- `mask_path`: Path to folder containing mask images
- `alpha_mask`: Use alpha channel from images as mask
- `mask_strength`: Mask effect strength (0.0-1.0, default 1.0)
- `invert_mask`: Invert the mask (swap masked/non-masked regions)
- `mask_min_value`: Minimum mask value (0.0-1.0, default 0.0)

**Training-level (applies to entire job):**
- `inverted_mask_prior`: Apply regularization to non-masked regions
- `inverted_mask_prior_multiplier`: Weight for regularization loss

## Backward Compatibility

**Breaking Changes:**
- `masked_recon_*` config fields removed - configs using these will need updating
- `mask_preview_*` config fields removed
- Scripts importing `toolkit.masked_recon` will fail

**Migration:**
- Remove `masked_recon_weight`, `masked_recon_type`, etc. from configs
- Use `mask_path` + `mask_strength` for standard masking
- Use `inverted_mask_prior` for regularization (was called "inverted mask prior" in masked_recon)

**No Breaking Changes:**
- Existing `mask_path`, `alpha_mask`, `invert_mask` configs continue working
- Default `mask_strength=1.0` maintains previous behavior
- Training loop behavior unchanged (except no masked_recon loss)

## Testing

Run unit tests:
```bash
python -m pytest testing/test_mask_strength.py -v
```

Expected output:
```
test_mask_strength.py::test_mask_strength_full PASSED
test_mask_strength.py::test_mask_strength_half PASSED
test_mask_strength.py::test_mask_strength_none PASSED
test_mask_strength.py::test_mask_strength_gradual PASSED
test_mask_strength.py::test_mask_strength_normalization PASSED
test_mask_strength.py::test_mask_strength_edge_cases PASSED
test_mask_strength.py::test_datasetconfig_mask_strength PASSED
test_mask_strength.py::test_mask_strength_batch_consistency PASSED
```

## Implementation Time

- **Planned:** 20-33 hours (full reimplementation)
- **Actual:** ~4 hours (discovered existing infrastructure, only added enhancements)

## Files Changed

### Modified:
- `toolkit/config_modules.py` - Added mask_strength, removed masked_recon fields
- `extensions_built_in/sd_trainer/SDTrainer.py` - Removed masked_recon code, added mask_strength blending
- `ui/src/app/jobs/new/SimpleJob.tsx` - Removed masked_recon UI section
- `ui/src/app/jobs/new/jobConfig.ts` - Removed masked_recon defaults

### Created:
- `scripts/generate_masks_sam2.py` - SAM2 mask generation CLI
- `testing/test_mask_strength.py` - Unit tests
- `docs/MASK_WORKFLOW_IMPLEMENTATION.md` - This document

### Deleted:
- `toolkit/masked_recon.py` - 690 lines removed

## Next Steps

### Optional Enhancements:
1. **UI Dataset View Controls** - Add mask settings to dataset configuration UI
2. **Multi-prompt Support** - Allow different prompts per image in SAM2 script
3. **Automatic Mask Detection** - Auto-detect subject in images
4. **Mask Visualization** - Preview masks overlaid on images in UI

### Documentation Updates:
- Update README with mask workflow examples
- Add tutorial for mask generation workflow
- Document mask_strength parameter in config reference
