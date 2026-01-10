# Mask Workflow - Quick Start Guide

This guide explains how to use the new mask workflow in AI-Toolkit for training with focused regions.

## Quick Start

### 1. Generate Masks

Use the SAM2-based mask generation script:

```bash
# Generate masks using a center point
python scripts/generate_masks_sam2.py \
  --dataset datasets/my_dataset \
  --points "[[512,512]]" \
  --output datasets/my_dataset/masks

# Generate masks using a bounding box
python scripts/generate_masks_sam2.py \
  --dataset datasets/my_dataset \
  --box "[[100,100,400,400]]" \
  --output datasets/my_dataset/masks

# Use a larger model for better quality
python scripts/generate_masks_sam2.py \
  --dataset datasets/my_dataset \
  --points "[[512,512]]" \
  --output datasets/my_dataset/masks \
  --model facebook/sam2.1-hiera-large
```

### 2. Configure Training

Add mask settings to your dataset configuration:

```yaml
datasets:
  - folder_path: datasets/my_dataset
    mask_path: datasets/my_dataset/masks  # Path to mask folder
    mask_strength: 1.0                     # 0.0-1.0, controls masking effect
    invert_mask: false                     # Swap masked/non-masked regions
```

### 3. Run Training

```bash
python run.py config/my_config.yaml
```

## Configuration Options

### Dataset-Level Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask_path` | string | None | Path to folder containing mask images |
| `alpha_mask` | bool | false | Use alpha channel from images as mask |
| `mask_strength` | float | 1.0 | Mask effect strength (0.0-1.0) |
| `invert_mask` | bool | false | Invert the mask |
| `mask_min_value` | float | 0.0 | Minimum mask value (0.0-1.0) |

### Training-Level Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inverted_mask_prior` | bool | false | Apply regularization to non-masked regions |
| `inverted_mask_prior_multiplier` | float | 0.1 | Weight for regularization loss |

## Mask Strength Explained

The `mask_strength` parameter controls how strongly the mask affects training:

- **1.0** (default): Fully zero out non-masked regions
  - Loss in non-masked regions = 0.0
  - Full masking effect

- **0.5**: Moderate masking
  - Loss in non-masked regions = 0.5 × normal loss
  - Balanced focus on masked regions

- **0.0**: No masking effect
  - Loss in non-masked regions = 1.0 × normal loss
  - Mask has no effect (all regions trained equally)

**Formula:** `final_weight = mask + (1-mask) * (1-strength)`

## SAM2 Mask Generation

### Available Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `facebook/sam2.1-hiera-tiny` | 39M | Fastest | Good |
| `facebook/sam2.1-hiera-small` | - | Fast | Better |
| `facebook/sam2.1-hiera-base-plus` | - | Medium | Very Good |
| `facebook/sam2.1-hiera-large` | - | Slow | Best |

### Prompt Types

#### Point Prompts
Specify points [x, y] to segment:

```bash
# Single point (center of 1024x1024 image)
--points "[[512,512]]"

# Multiple points
--points "[[100,100],[200,200],[300,300]]"

# With labels (1=foreground, 0=background)
--points "[[100,100],[200,200]]" --labels "[1,0]"
```

#### Bounding Box Prompts
Specify boxes [x1, y1, x2, y2]:

```bash
# Single box
--box "[[100,100,400,400]]"

# Multiple boxes
--box "[[100,100,200,200],[300,300,500,500]]"
```

## Advanced Usage

### Use Alpha Channel as Mask

If your images have transparency:

```yaml
datasets:
  - folder_path: datasets/my_dataset
    alpha_mask: true
    mask_strength: 0.8
```

### Inverted Mask Prior (Regularization)

Apply regularization to non-masked regions:

```yaml
datasets:
  - folder_path: datasets/my_dataset
    mask_path: datasets/my_dataset/masks

train:
  inverted_mask_prior: true
  inverted_mask_prior_multiplier: 0.1  # Adjust regularization strength
```

### Per-Image Masks

Masks are matched by filename:

```
datasets/my_dataset/
  ├── image001.jpg
  ├── image002.jpg
  └── masks/
      ├── image001.png  ← Matches image001.jpg
      └── image002.png  ← Matches image002.jpg
```

## Troubleshooting

### Issue: Masks not loading

**Solution:** Check that:
1. Mask filenames match image filenames (excluding extension)
2. `mask_path` points to the correct folder
3. Masks are grayscale PNG files (values 0-255)

### Issue: Training focuses too much on masked region

**Solution:** Reduce `mask_strength` to 0.5-0.7

### Issue: Training still learns background features

**Solution:** 
1. Increase `mask_strength` closer to 1.0
2. Or use `inverted_mask_prior` for regularization

### Issue: SAM2 script fails with OOM

**Solution:** Use a smaller model or reduce image resolution:
```bash
--model facebook/sam2.1-hiera-tiny
```

## Migration from Masked Reconstruction

If you previously used the `masked_recon_*` feature:

**Before:**
```yaml
train:
  masked_recon_weight: 0.5
  masked_recon_type: illum
```

**After:**
```yaml
datasets:
  - folder_path: datasets/my_dataset
    mask_path: datasets/my_dataset/masks
    mask_strength: 1.0
```

**Key Changes:**
- Generate masks externally using SAM2 script
- Configure masks per-dataset instead of per-job
- Use `mask_strength` for fine control
- Use `inverted_mask_prior` for regularization (was part of masked_recon)

## Examples

### Example 1: Focus on Character

Train a LoRA focusing only on a character (ignore background):

```bash
# 1. Generate masks for the character
python scripts/generate_masks_sam2.py \
  --dataset datasets/character \
  --points "[[512,512]]" \
  --output datasets/character/masks

# 2. Configure training
# config.yaml:
#   datasets:
#     - folder_path: datasets/character
#       mask_path: datasets/character/masks
#       mask_strength: 1.0

# 3. Train
python run.py config/character_lora.yaml
```

### Example 2: Focus on Gun with Background Regularization

```yaml
datasets:
  - folder_path: datasets/gun
    mask_path: datasets/gun/masks
    mask_strength: 0.9  # Strong focus on gun

train:
  inverted_mask_prior: true
  inverted_mask_prior_multiplier: 0.05  # Light regularization on background
```

### Example 3: Partial Masking

Train with 70% focus on masked region, 30% on background:

```yaml
datasets:
  - folder_path: datasets/subject
    mask_path: datasets/subject/masks
    mask_strength: 0.7  # Partial masking
```

## See Also

- [MASK_WORKFLOW_IMPLEMENTATION.md](MASK_WORKFLOW_IMPLEMENTATION.md) - Full implementation details
- [MASK_WORKFLOW_REVISED_PLAN.md](MASK_WORKFLOW_REVISED_PLAN.md) - Original implementation plan
- [LEARNINGS.md](../LEARNINGS.md) - Project learnings including mask workflow

## Support

For issues or questions:
1. Check existing GitHub issues
2. Review the implementation documentation
3. Open a new issue with reproduction steps
