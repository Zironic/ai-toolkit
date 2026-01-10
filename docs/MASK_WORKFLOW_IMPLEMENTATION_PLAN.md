# Mask Workflow Implementation Plan

## IMPORTANT DISCOVERY: Most Infrastructure Already Exists! 🎉

The codebase **already has comprehensive mask support** that matches your requirements:
- ✅ Mask loading from external folder (via `mask_path` config)
- ✅ `FileItemDTO.mask_tensor` and `DataLoaderBatchDTO.mask_tensor` fields
- ✅ Automatic mask resizing to latent dimensions
- ✅ Loss multiplication by mask in training loop
- ✅ Support for alpha channel masks and mask inversion
- ✅ `inverted_mask_prior` for regularization on non-masked regions

**What's Actually Missing:**
1. SAM3 mask generation CLI script
2. Per-dataset mask UI controls (mask settings are currently in job/train config)
3. Mask strength parameter (currently uses fixed mask or inverted_mask_prior_multiplier)
4. Removal of masked_recon code (separate feature)

## Overview
Leverage existing mask infrastructure, add SAM3 generation CLI, enhance UI with per-dataset controls, and remove the separate masked_recon feature.

---

## Phase 1: Remove Old Masked Reconstruction Code

### 1.1 Delete Masked Reconstruction Implementation
**File:** `toolkit/masked_recon.py`
**Action:** Delete entire file (690 lines)

### 1.2 Remove Config Fields
**File:** `toolkit/config_modules.py`
**Changes:**
- Remove all `masked_recon_*` fields (lines ~449-460)
- Remove all `mask_preview_*` fields
- Remove `controlnet_frozen` field (line ~469)
- Update `allowed_aux` set: `{'none', 'edge', 'masked_recon'}` → `{'none', 'edge'}`

### 1.3 Remove Trainer Methods
**File:** `extensions_built_in/sd_trainer/SDTrainer.py`
**Remove:**
- Method `generate_mask_previews_if_enabled()` (lines 997-1030)
- Method `_compute_and_apply_masked_recon_loss()` (lines 3646-3664)
- Local helper `_apply_masked_recon_loss_local()` in training loop (lines 2293-2304)
- Variable `masked_recon_logged` initialization (line 2306)
- Timer block calling masked recon (lines 3365-3368)
- Call to `self.generate_mask_previews_if_enabled()` (line 832)

### 1.4 Update Base Training Process
**File:** `jobs/process/BaseSDTrainProcess.py`
**Changes:**
- Remove `mr_weight` detection logic (lines 2232-2243)
- Always detach `noisy_latents` (no conditional grad preservation)

### 1.5 Remove/Update Scripts
**Files:**
- `scripts/preview_mask.py` (imports masked_recon)
- `scripts/debug_mask_dense.py` (imports masked_recon)
- `scripts/debug_inspect_mask.py` (imports masked_recon)
- `scripts/debug_combined_case.py` (imports masked_recon)

**Action:** Delete these debug scripts (they're masked_recon-specific)

### 1.6 Remove UI Components
**File:** `ui/src/app/jobs/new/jobConfig.ts`
**Remove:**
- `masked_recon_weight`, `masked_recon_type`, `masked_recon_mask_key`
- `mask_preview_enabled`, `mask_preview_save_path`, `mask_preview_overwrite`, `mask_preview_overlay`
- `controlnet_frozen`

**File:** `ui/src/app/jobs/new/SimpleJob.tsx`
**Remove:**
- "Masked Reconstruction" FormGroup (lines 679-714)
- "Mask Preview (debug)" FormGroup (lines 721-751)

### 1.7 Remove Tests
**Delete:**
- `testing/test_masked_recon_default.py`
- `testing/test_masked_recon_exception_logging.py`
- `testing/test_masked_recon_helper.py`
- `testing/test_masked_recon_no_mutation.py`
- `testing/test_mask_preview.py`
- `testing/test_masked_reconstruction_integration.py`
- `testing/test_masked_recon_control.py`
- `testing/test_sdtrainer_mask_preview_integration.py`

### 1.8 Remove Documentation
**Delete/Archive:**
- `docs/MASKED_RECONSTRUCTION_IMPLEMENTATION.md`

**Update:**
- `LEARNINGS.md` - add entry documenting removal and new workflow

---

## Phase 2: Implement New Mask Loading System

### 2.1 Add Mask Discovery to Dataset Config
**File:** `toolkit/config_modules.py` (DatasetConfig class)
**Add fields:**
```python
# Mask support (new workflow)
self.mask_enabled: bool = kwargs.get('mask_enabled', False)
self.mask_strength: float = kwargs.get('mask_strength', 1.0)  # 0.0-1.0; 1.0 = fully zero non-masked regions
self.mask_subfolder: str = kwargs.get('mask_subfolder', '_masks')  # relative to dataset folder
```

**Validation:**
```python
# In validate_dataset_config or similar
if dataset_config.mask_enabled:
    if not (0.0 <= dataset_config.mask_strength <= 1.0):
        raise ValueError(f"mask_strength must be in [0.0, 1.0], got {dataset_config.mask_strength}")
```

### 2.2 Add Mask Loading to FileItemDTO
**File:** `toolkit/data_loader.py` (or wherever FileItemDTO is defined)
**Add field:**
```python
@dataclass
class FileItemDTO:
    # ... existing fields ...
    mask: Optional[torch.Tensor] = None  # [1, H, W] binary mask if mask_enabled
```

### 2.3 Implement Mask Discovery and Loading
**File:** `toolkit/dataloader_mixins.py` (or new `toolkit/mask_utils.py`)
**Add helper:**
```python
def load_mask_for_image(image_path: str, mask_subfolder: str = '_masks') -> Optional[torch.Tensor]:
    """Load binary mask for given image from _masks subfolder.
    
    Args:
        image_path: Full path to image file
        mask_subfolder: Name of subfolder containing masks (relative to image directory)
    
    Returns:
        Binary mask tensor [1, H, W] with values {0.0, 1.0}, or None if not found
    """
    from PIL import Image
    import os
    
    # Construct mask path: replace image directory with mask subfolder, keep filename
    img_dir = os.path.dirname(image_path)
    img_name = os.path.basename(image_path)
    # Try common extensions
    base_name = os.path.splitext(img_name)[0]
    mask_dir = os.path.join(img_dir, mask_subfolder)
    
    for ext in ['.png', '.jpg', '.jpeg']:
        mask_path = os.path.join(mask_dir, base_name + ext)
        if os.path.exists(mask_path):
            try:
                mask_pil = Image.open(mask_path).convert('L')  # grayscale
                mask_np = np.array(mask_pil)
                # Binarize: >127 = 1.0, <=127 = 0.0
                mask_binary = (mask_np > 127).astype(np.float32)
                mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0)  # [1, H, W]
                return mask_tensor
            except Exception as e:
                print(f"Warning: failed to load mask {mask_path}: {e}")
                return None
    
    return None  # No mask found
```

### 2.4 Integrate Mask Loading into Dataset Processing
**File:** `toolkit/dataloader_mixins.py` (or dataset loading code)
**Update image loading:**
```python
# In the dataset's __getitem__ or file loading method
if self.dataset_config.mask_enabled:
    mask = load_mask_for_image(file_item.path, self.dataset_config.mask_subfolder)
    file_item.mask = mask
    if mask is None:
        # Optional: log warning or raise error if masks are required but missing
        print(f"Warning: mask enabled but not found for {file_item.path}")
```

---

## Phase 3: Implement Mask Application in Training Loop

### 3.1 Add Mask Tensor to Batch DTO
**File:** `toolkit/data_loader.py` (DataLoaderBatchDTO)
**Add field:**
```python
@dataclass
class DataLoaderBatchDTO:
    # ... existing fields ...
    mask_multiplier_tensor: Optional[torch.Tensor] = None  # [B, 1, H, W] for loss masking
```

### 3.2 Build Mask Multiplier in Dataloader Collation
**File:** Dataset collation code (likely in dataloader or dataset class)
**Add logic:**
```python
# During batch collation (after collecting file_items)
masks = []
for fi in batch_file_items:
    if fi.mask is not None:
        masks.append(fi.mask)
    else:
        # Default to all-ones (no masking)
        masks.append(torch.ones(1, fi.original_height, fi.original_width))

if len(masks) > 0:
    mask_batch = torch.stack(masks, dim=0)  # [B, 1, H, W]
    batch.mask_multiplier_tensor = mask_batch
```

### 3.3 Apply Mask in Training Loop
**File:** `extensions_built_in/sd_trainer/SDTrainer.py`
**Location:** After computing main loss, before preservation losses (around line 3300-3320)

**Add new method:**
```python
def _apply_mask_to_loss(self, loss: torch.Tensor, batch: 'DataLoaderBatchDTO', 
                        noisy_latents: torch.Tensor) -> torch.Tensor:
    """Apply dataset-specific mask multiplier to loss.
    
    Masks are inverted: regions under mask get full loss, regions outside mask
    get reduced loss based on mask_strength. At strength=1.0, non-masked regions
    are zeroed; at strength=0.0, no masking is applied.
    
    Args:
        loss: Loss tensor (scalar or per-element)
        batch: Batch with mask_multiplier_tensor [B,1,H,W]
        noisy_latents: Noisy latents [B,C,H,W] for shape reference
        
    Returns:
        Masked loss tensor
    """
    if batch.mask_multiplier_tensor is None:
        return loss
    
    # Get mask strength from dataset config (assume single dataset per batch for now)
    # TODO: support per-sample mask_strength if mixed datasets in batch
    mask_strength = 1.0
    if hasattr(batch, 'dataset_config') and batch.dataset_config is not None:
        mask_strength = float(getattr(batch.dataset_config, 'mask_strength', 1.0))
    
    if mask_strength == 0.0:
        return loss  # No masking
    
    # Resize mask to match latent dimensions
    B, C, H, W = noisy_latents.shape
    mask = batch.mask_multiplier_tensor.to(loss.device, dtype=loss.dtype)
    
    if mask.shape[-2:] != (H, W):
        mask = torch.nn.functional.interpolate(
            mask, size=(H, W), mode='bilinear', align_corners=False
        )
    
    # Invert mask: 1.0 where mask is present, (1-strength) where mask is absent
    # mask values are {0, 1}, so:
    # multiplier = mask + (1 - mask) * (1 - mask_strength)
    #            = mask + (1 - mask) * (1 - mask_strength)
    #            = mask * mask_strength + (1 - mask_strength)
    # Wait, let me recalculate:
    # - mask=1 (masked region): multiplier=1.0 (full loss)
    # - mask=0 (non-masked region): multiplier=(1-mask_strength)
    # So: multiplier = mask + (1 - mask) * (1 - mask_strength)
    multiplier = mask + (1.0 - mask) * (1.0 - mask_strength)
    
    # Expand to match loss dimensions if needed
    if loss.dim() == 0:
        # Scalar loss: cannot apply spatial mask, return unchanged
        return loss
    elif loss.dim() == 4:  # [B, C, H, W]
        multiplier = multiplier.expand(-1, C, -1, -1)
        return loss * multiplier
    elif loss.dim() == 2:  # [B, spatial_dims]
        multiplier_flat = multiplier.view(B, -1)
        return loss * multiplier_flat
    else:
        # Unsupported loss shape; return unchanged
        return loss
```

**Insert call in training loop:**
```python
# After: loss = F.mse_loss(...) or similar
# Before: preservation loss branch

# Apply mask if present (only to main training loss, not preservation)
if batch.mask_multiplier_tensor is not None:
    loss = self._apply_mask_to_loss(loss, batch, noisy_latents)
```

**Important:** Ensure masking is NOT applied to:
- Preservation losses (DOP, blank prompt preservation)
- Regularization losses
- Any auxiliary losses

---

## Phase 4: Implement SAM3 Mask Generation CLI

### 4.1 Add SAM3 Dependencies
**File:** `requirements.txt`
**Add:**
```
# For SAM3 mask generation
transformers>=4.30.0  # (likely already present)
# SAM3 may require specific torch/transformers versions - check HF model card
```

### 4.2 Create SAM3 Mask Generation Script
**File:** `scripts/generate_masks_sam3.py`
**Content:**
```python
#!/usr/bin/env python3
"""Generate binary masks for a dataset using SAM3 (Segment Anything Model 3).

Usage:
    python scripts/generate_masks_sam3.py --dataset datasets/my_dataset --prompts "person,gun" --output-subfolder _masks

This will:
1. Load images from the dataset folder
2. Use SAM3 to segment objects matching the text prompts
3. Save binary masks to dataset/_masks/ with matching filenames
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

# SAM3 imports (adjust based on actual SAM3 API)
try:
    from transformers import AutoProcessor, AutoModelForMaskGeneration
except ImportError:
    print("Error: transformers library required. Install: pip install transformers")
    sys.exit(1)


def load_sam3_model(device: str = "cuda"):
    """Load SAM3 model and processor from HuggingFace."""
    model_id = "facebook/sam3"  # Adjust if needed
    
    print(f"Loading SAM3 model from {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMaskGeneration.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    
    return model, processor


def generate_mask_for_image(image_path: str, prompts: List[str], model, processor, device: str) -> np.ndarray:
    """Generate binary mask for image using text prompts.
    
    Args:
        image_path: Path to input image
        prompts: List of text prompts (e.g., ["person", "gun"])
        model: SAM3 model
        processor: SAM3 processor
        device: Device string
        
    Returns:
        Binary mask as numpy array [H, W] with values {0, 255}
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Prepare inputs (this is pseudocode - adjust for actual SAM3 API)
    inputs = processor(images=image, text=prompts, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate mask
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract binary mask (adjust based on SAM3 output format)
        # Assuming outputs.masks is [1, H, W] with logits
        mask_logits = outputs.masks[0]  # [H, W]
        mask_binary = (mask_logits > 0).cpu().numpy().astype(np.uint8) * 255
    
    return mask_binary


def process_dataset(dataset_path: str, prompts: List[str], output_subfolder: str, device: str):
    """Process all images in dataset and generate masks.
    
    Args:
        dataset_path: Path to dataset folder
        prompts: List of text prompts for segmentation
        output_subfolder: Name of subfolder to save masks (e.g., "_masks")
        device: Device string
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: dataset path {dataset_path} does not exist")
        sys.exit(1)
    
    # Create output directory
    output_dir = dataset_path / output_subfolder
    output_dir.mkdir(exist_ok=True)
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = [
        f for f in dataset_path.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if len(image_files) == 0:
        print(f"Warning: no images found in {dataset_path}")
        return
    
    print(f"Found {len(image_files)} images in {dataset_path}")
    print(f"Generating masks for prompts: {prompts}")
    
    # Load model
    model, processor = load_sam3_model(device=device)
    
    # Process each image
    for image_file in tqdm(image_files, desc="Generating masks"):
        try:
            # Generate mask
            mask = generate_mask_for_image(
                str(image_file), prompts, model, processor, device
            )
            
            # Save mask with same filename (but as PNG)
            mask_filename = image_file.stem + ".png"
            mask_path = output_dir / mask_filename
            
            mask_pil = Image.fromarray(mask, mode='L')
            mask_pil.save(mask_path)
            
        except Exception as e:
            print(f"\nError processing {image_file.name}: {e}")
    
    print(f"\nDone! Masks saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate binary masks using SAM3 for dataset images"
    )
    parser.add_argument(
        "--dataset", "-d", required=True,
        help="Path to dataset folder"
    )
    parser.add_argument(
        "--prompts", "-p", required=True,
        help="Comma-separated text prompts for segmentation (e.g., 'person,gun')"
    )
    parser.add_argument(
        "--output-subfolder", "-o", default="_masks",
        help="Name of subfolder to save masks (default: _masks)"
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on (default: cuda if available)"
    )
    
    args = parser.parse_args()
    
    # Parse prompts
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    if len(prompts) == 0:
        print("Error: no valid prompts provided")
        sys.exit(1)
    
    process_dataset(
        dataset_path=args.dataset,
        prompts=prompts,
        output_subfolder=args.output_subfolder,
        device=args.device
    )


if __name__ == "__main__":
    main()
```

**Note:** The SAM3 API calls above are placeholders - adjust based on actual HuggingFace model API.

### 4.3 Add CLI Command to Runner
**File:** `run.py`
**Add subcommand:**
```python
# Add to argument parser
subparsers = parser.add_subparsers(dest='command')

# ... existing subparsers ...

# New: generate-masks subcommand
mask_parser = subparsers.add_parser('generate-masks', help='Generate masks using SAM3')
mask_parser.add_argument('--dataset', '-d', required=True, help='Dataset path')
mask_parser.add_argument('--prompts', '-p', required=True, help='Comma-separated prompts')
mask_parser.add_argument('--output-subfolder', '-o', default='_masks', help='Output subfolder')

# In main():
if args.command == 'generate-masks':
    from scripts.generate_masks_sam3 import process_dataset
    process_dataset(args.dataset, args.prompts.split(','), args.output_subfolder, 'cuda')
    sys.exit(0)
```

---

## Phase 5: Implement UI for Per-Dataset Mask Controls

### 5.1 Update Dataset Config Schema
**File:** `ui/src/app/datasets/[id]/types.ts` (or wherever dataset config types are)
**Add fields:**
```typescript
interface DatasetConfig {
  // ... existing fields ...
  mask_enabled?: boolean;
  mask_strength?: number;  // 0.0-1.0
  mask_subfolder?: string;
}
```

### 5.2 Add UI Controls to Dataset View
**File:** `ui/src/app/datasets/[id]/page.tsx` (or dataset settings component)
**Add form section:**
```tsx
<FormGroup label="Mask Settings" className="pt-2">
  <Checkbox
    label="Enable Masks"
    checked={dataset.mask_enabled || false}
    onChange={value => updateDataset({ mask_enabled: value })}
  />
  <NumberInput
    label="Mask Strength"
    className="pt-2"
    value={dataset.mask_strength ?? 1.0}
    onChange={value => updateDataset({ mask_strength: value })}
    min={0.0}
    max={1.0}
    step={0.1}
    disabled={!dataset.mask_enabled}
    helperText="1.0 = fully zero non-masked regions, 0.0 = no effect"
  />
  <TextInput
    label="Mask Subfolder"
    className="pt-2"
    value={dataset.mask_subfolder || '_masks'}
    onChange={value => updateDataset({ mask_subfolder: value })}
    disabled={!dataset.mask_enabled}
    helperText="Subfolder containing mask images"
  />
  <div className="pt-2 text-sm text-gray-600">
    <p>Masks should be binary PNG images in the '{dataset.mask_subfolder || '_masks'}' subfolder.</p>
    <p>Generate masks using: <code>python scripts/generate_masks_sam3.py --dataset {dataset.path} --prompts "your,prompts"</code></p>
  </div>
</FormGroup>
```

### 5.3 Update Dataset API Endpoints
**File:** `ui/src/app/api/datasets/[id]/route.ts`
**Update PATCH handler to accept mask fields:**
```typescript
// In PATCH handler body validation
const validFields = [
  // ... existing fields ...
  'mask_enabled',
  'mask_strength',
  'mask_subfolder',
];
```

---

## Phase 6: Testing and Validation

### 6.1 Unit Tests for Mask Loading
**File:** `testing/test_mask_loading.py`
```python
import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os

def test_load_mask_for_image_found():
    """Test mask loading when mask file exists."""
    from toolkit.mask_utils import load_mask_for_image
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create image
        img_path = os.path.join(tmpdir, "test.jpg")
        Image.new('RGB', (256, 256)).save(img_path)
        
        # Create mask
        mask_dir = os.path.join(tmpdir, "_masks")
        os.makedirs(mask_dir)
        mask_path = os.path.join(mask_dir, "test.png")
        mask_np = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        Image.fromarray(mask_np, mode='L').save(mask_path)
        
        # Load
        mask = load_mask_for_image(img_path, "_masks")
        
        assert mask is not None
        assert mask.shape == (1, 256, 256)
        assert mask.dtype == torch.float32
        assert torch.all((mask == 0.0) | (mask == 1.0))

def test_load_mask_for_image_not_found():
    """Test mask loading when mask file does not exist."""
    from toolkit.mask_utils import load_mask_for_image
    
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.jpg")
        Image.new('RGB', (256, 256)).save(img_path)
        
        mask = load_mask_for_image(img_path, "_masks")
        assert mask is None
```

### 6.2 Unit Tests for Mask Application
**File:** `testing/test_mask_application.py`
```python
import pytest
import torch

def test_apply_mask_to_loss_full_strength():
    """Test mask application at full strength (1.0)."""
    # Setup
    loss = torch.ones(2, 4, 8, 8)  # [B, C, H, W]
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :4, :] = 1.0  # Mask top half
    
    # Mock batch
    class MockBatch:
        mask_multiplier_tensor = mask
        class dataset_config:
            mask_strength = 1.0
    
    batch = MockBatch()
    noisy_latents = torch.zeros(2, 4, 8, 8)
    
    # Apply mask (need to import the method)
    # trainer._apply_mask_to_loss(loss, batch, noisy_latents)
    # For now, test the formula directly:
    mask_strength = 1.0
    multiplier = mask + (1.0 - mask) * (1.0 - mask_strength)
    multiplier = multiplier.expand(-1, 4, -1, -1)
    masked_loss = loss * multiplier
    
    # Verify
    assert torch.allclose(masked_loss[:, :, :4, :], loss[:, :, :4, :])  # Masked region unchanged
    assert torch.allclose(masked_loss[:, :, 4:, :], torch.zeros_like(loss[:, :, 4:, :]))  # Non-masked zeroed

def test_apply_mask_to_loss_half_strength():
    """Test mask application at 50% strength."""
    loss = torch.ones(2, 4, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    mask[:, :, :4, :] = 1.0
    
    mask_strength = 0.5
    multiplier = mask + (1.0 - mask) * (1.0 - mask_strength)
    multiplier = multiplier.expand(-1, 4, -1, -1)
    masked_loss = loss * multiplier
    
    # Masked region: full loss
    assert torch.allclose(masked_loss[:, :, :4, :], loss[:, :, :4, :])
    # Non-masked region: 50% loss
    assert torch.allclose(masked_loss[:, :, 4:, :], loss[:, :, 4:, :] * 0.5)
```

### 6.3 Integration Test
**File:** `testing/test_mask_training_integration.py`
```python
def test_training_with_masks_enabled():
    """Test that training runs successfully with masks enabled."""
    # Setup minimal trainer with mask-enabled dataset
    # Run a few training steps
    # Verify loss is computed and masks are applied
    # (Implementation details depend on test harness)
    pass
```

### 6.4 SAM3 Script Test
**File:** `testing/test_generate_masks_sam3.py`
```python
def test_generate_masks_sam3_creates_files():
    """Test that SAM3 script creates mask files."""
    # Create temp dataset with images
    # Run generate_masks_sam3.process_dataset()
    # Verify masks are created in _masks subfolder
    # (May need to mock SAM3 model to avoid heavy download)
    pass
```

---

## Phase 7: Documentation and Migration

### 7.1 Update LEARNINGS.md
**File:** `LEARNINGS.md`
**Add entry:**
```markdown
## Mask Workflow Replacement (2026-01-10)

**What changed:**
- Removed masked reconstruction feature (toolkit/masked_recon.py and all related code)
- Implemented new mask workflow: masks are pre-generated and stored in _masks subfolder
- Masks are applied as negative multipliers during training (configurable per-dataset)

**Why:**
- Simplify mask generation (separate from training)
- More flexible mask sources (SAM3, manual drawing, etc.)
- Clearer separation of concerns

**How to migrate:**
1. Remove all `masked_recon_*` and `mask_preview_*` config keys from job configs
2. Generate masks using `scripts/generate_masks_sam3.py`
3. Enable masks per-dataset in UI with `mask_enabled` and `mask_strength`

**Gotchas:**
- Masks are NOT applied to preservation or regularization losses (by design)
- Mask files must match image filenames (different extension OK)
- Binary masks only (0/255); grayscale will be thresholded at 127

**PR:** #XXX
**Tests:** Added test_mask_loading.py, test_mask_application.py
```

### 7.2 Update README
**File:** `README.md`
**Add section:**
```markdown
## Using Masks

Masks allow you to focus training on specific regions of your images. Masks are pre-generated and stored alongside your dataset.

### Generating Masks

Use the SAM3-based mask generator:

```bash
python scripts/generate_masks_sam3.py --dataset datasets/my_dataset --prompts "person,gun"
```

This creates binary masks in `datasets/my_dataset/_masks/`.

### Enabling Masks

1. Open the dataset in the UI
2. Check "Enable Masks"
3. Set "Mask Strength" (1.0 = fully ignore non-masked regions, 0.0 = no effect)

Masks are applied to the main training loss only (not preservation/regularization).

### Manual Mask Creation (Coming Soon)

A UI for drawing masks manually is planned for a future release.
```

### 7.3 Update User Documentation
**File:** `docs/training_guide.md` (or similar)
**Add mask workflow section with examples and screenshots**

---

## Phase 8: Future Enhancements (Not Part of Initial Implementation)

### 8.1 Manual Mask Drawing UI
- Add canvas-based mask editor to dataset view
- Allow brush/eraser tools with size controls
- Save directly to _masks subfolder

### 8.2 Per-Sample Mask Strength
- Support mixed datasets with different mask_strength values
- Pass mask_strength per sample in batch

### 8.3 Alternative Mask Generation Methods
- Add support for other segmentation models (YOLO, Detectron2)
- Add point/box prompting UI for SAM3

### 8.4 Mask Visualization
- Show mask overlay on dataset preview images
- Display mask coverage statistics

---

## Implementation Checklist

### Phase 1: Cleanup (Estimated: 2-4 hours)
- [ ] Delete `toolkit/masked_recon.py`
- [ ] Remove config fields from `toolkit/config_modules.py`
- [ ] Remove trainer methods from `extensions_built_in/sd_trainer/SDTrainer.py`
- [ ] Update `jobs/process/BaseSDTrainProcess.py`
- [ ] Delete debug scripts
- [ ] Remove UI components from `ui/src/app/jobs/new/`
- [ ] Delete test files
- [ ] Archive/delete `docs/MASKED_RECONSTRUCTION_IMPLEMENTATION.md`

### Phase 2: Mask Loading (Estimated: 3-5 hours)
- [ ] Add config fields to DatasetConfig
- [ ] Add mask field to FileItemDTO
- [ ] Implement `load_mask_for_image()` helper
- [ ] Integrate into dataset loading
- [ ] Add mask_multiplier_tensor to DataLoaderBatchDTO
- [ ] Build mask batch in collation

### Phase 3: Mask Application (Estimated: 2-4 hours)
- [ ] Implement `_apply_mask_to_loss()` method
- [ ] Insert call in training loop
- [ ] Verify preservation losses are NOT masked

### Phase 4: SAM3 CLI (Estimated: 4-6 hours)
- [ ] Create `scripts/generate_masks_sam3.py`
- [ ] Test with sample dataset
- [ ] Add to `run.py` subcommands
- [ ] Update requirements.txt if needed

### Phase 5: UI (Estimated: 3-5 hours)
- [ ] Add mask fields to dataset config types
- [ ] Add UI controls to dataset view
- [ ] Update API endpoints
- [ ] Test UI workflow

### Phase 6: Testing (Estimated: 4-6 hours)
- [ ] Write unit tests for mask loading
- [ ] Write unit tests for mask application
- [ ] Write integration test
- [ ] Test SAM3 script
- [ ] Manual GPU smoke test

### Phase 7: Documentation (Estimated: 2-3 hours)
- [ ] Update LEARNINGS.md
- [ ] Update README.md
- [ ] Update training guide
- [ ] Add migration notes

### Total Estimated Time: 20-33 hours

---

## Risk Mitigation

### Risk: SAM3 API Changes
**Mitigation:** Pin transformers version; add fallback to SAM2 if SAM3 unavailable

### Risk: Performance Impact of On-the-Fly Mask Resizing
**Mitigation:** Profile mask resizing; consider caching resized masks if bottleneck

### Risk: Mixed Datasets with Different Mask Settings
**Mitigation:** Document limitation; add per-sample mask_strength support in Phase 8

### Risk: Breaking Existing Configs
**Mitigation:** Removal of old keys is clean break; document migration clearly

---

## Success Criteria

- [ ] All old masked_recon code removed
- [ ] SAM3 mask generation script works on sample dataset
- [ ] Masks load correctly and apply to loss
- [ ] UI allows enabling/configuring masks per dataset
- [ ] All tests pass (unit, integration, smoke)
- [ ] Documentation updated
- [ ] No performance regression on non-masked training
- [ ] GPU manual verification shows masks reduce loss in non-masked regions

---

## Open Questions

1. Should we support multiple mask files per image (e.g., different prompts → separate masks)?
   - **Decision:** Not in initial implementation; user can combine masks externally

2. Should masks be cached/preloaded for performance?
   - **Decision:** Load on-the-fly initially; optimize if needed

3. Should we validate that masks exist when mask_enabled=True?
   - **Decision:** Warn but don't fail; allow datasets with partial mask coverage
